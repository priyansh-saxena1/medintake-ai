import os
import json
from typing import Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.llm import get_llm, CombinedOutput, HPI_FIELDS, ROS_REQUIRED
from app.schemas import ClinicalBrief, HPI, ClinicalStateExtraction

_MOCK = lambda: os.environ.get("MOCK_LLM", "true").lower() == "true"


def add_messages(left: list[dict], right: list[dict]) -> list[dict]:
    return left + right


class IntakeState(TypedDict):
    messages: Annotated[list[dict], add_messages]
    clinical_state: str          # JSON of CombinedOutput (accumulated clinical data)
    missing_fields: list[str]
    current_node: str
    clinical_brief: Optional[dict]
    frontend_stage: str          # 'intake', 'hpi', 'ros', 'done'


# HPI_FIELDS and ROS_REQUIRED imported from app.llm

EMERGENCY_PHRASES = [
    "crushing chest pain", "can't breathe", "cannot breathe",
    "heart attack", "suicide", "kill myself", "can't move", "dying"
]


# ------------------------------------------------------------------ helpers --

def format_transcript(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = "AI" if m["role"] == "assistant" else "Patient"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def compute_stage(state: CombinedOutput) -> str:
    if not state.chief_complaint:
        return "intake"
    for f in HPI_FIELDS:
        if not getattr(state, f):
            return "hpi"
    if len(state.ros) < ROS_REQUIRED:
        return "ros"
    return "done"


def missing_from(state: CombinedOutput) -> list[str]:
    missing = []
    if not state.chief_complaint:
        missing.append("chief complaint")
        return missing
    for f in HPI_FIELDS:
        if not getattr(state, f):
            missing.append(f"HPI:{f}")
    if len(state.ros) < ROS_REQUIRED:
        missing.append(f"ROS ({ROS_REQUIRED - len(state.ros)} more systems needed)")
    return missing


def _detect_repeat(state) -> bool:
    """Return True if the last two assistant replies are identical."""
    msgs = state.get("messages", [])
    assistant_replies = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
    return len(assistant_replies) >= 2 and assistant_replies[-1] == assistant_replies[-2]


# ------------------------------------------------------------------- nodes ---

def triage_node(state: IntakeState) -> dict:
    """Fast keyword check — no LLM call. Abort immediately on emergency phrases."""
    msgs = state.get("messages", [])
    if msgs and msgs[-1]["role"] == "user":
        content = msgs[-1]["content"].lower()
        if any(p in content for p in EMERGENCY_PHRASES):
            return {
                "messages": [{
                    "role": "assistant",
                    "content": (
                        "🚨 EMERGENCY: Your symptoms require immediate attention. "
                        "Please call 911 or go to your nearest emergency room right away."
                    )
                }],
                "current_node": "done",
                "frontend_stage": "done",
            }
    return {"current_node": "agent"}


def agent_node(state: IntakeState) -> dict:
    """
    Core agent node — ONE combined LLM call per turn:
    1. Extracts any new clinical data from the transcript.
    2. Generates the next conversational question.
    3. If all data is collected, builds the ClinicalBrief inline (no separate scribe node).
    """
    msgs = state.get("messages", [])

    # On first call with no messages, return opening greeting
    if not msgs or (len(msgs) == 1 and msgs[0]["role"] == "assistant"):
        return {
            "messages": [{"role": "assistant", "content": "Hello, I'm conducting your pre-visit clinical intake. What brings you in today?"}],
            "clinical_state": CombinedOutput().model_dump_json(),
            "frontend_stage": "intake",
            "current_node": "agent",
        }

    if msgs[-1]["role"] == "assistant":
        return {"current_node": "agent"}

    current_json = state.get("clinical_state") or CombinedOutput().model_dump_json()
    transcript = format_transcript(msgs)

    # Compute the current stage BEFORE the LLM call so we can pick the right prompt
    try:
        pre_state = CombinedOutput.model_validate_json(current_json)
        current_stage = compute_stage(pre_state)
    except Exception:
        current_stage = "intake"

    import time
    t_agent = time.time()
    print(f"[{time.time():.3f}] [Graph Node] Requesting LLM inference (stage={current_stage})...")

    llm = get_llm()
    result: CombinedOutput = llm.combined_call(transcript, current_json, stage=current_stage)

    # ── Loop Guard: if LLM returned same reply as last turn, force-fill stuck field ──
    if _detect_repeat({"messages": msgs + [{"role": "assistant", "content": result.reply}]}):
        hpi_filled = all(getattr(result, f, None) for f in HPI_FIELDS)

        if not hpi_filled:
            # Still in HPI — force-fill the first empty HPI field
            for stuck_field in HPI_FIELDS:
                if getattr(result, stuck_field, None) is None:
                    object.__setattr__(result, stuck_field, "not specified")
                    print(f"[LoopGuard] Force-filled HPI '{stuck_field}' = 'not specified' to break repeat loop")
                    new_missing = missing_from(result)
                    if new_missing:
                        object.__setattr__(result, "reply", f"Thank you. Now, could you tell me about {new_missing[0].replace('HPI:', '')}?")
                    else:
                        object.__setattr__(result, "reply", "Thank you — I have everything I need.")
                    break
        else:
            # In ROS stage — force-fill the current ROS system with patient's last answer
            patient_answer = ""
            for m in reversed(msgs):
                if m.get("role") == "user":
                    patient_answer = m.get("content", "denied")
                    break
            patient_answer = patient_answer or "denied"

            # Find which ROS system the LLM was asking about (from its repeated reply)
            ros = dict(result.ros) if result.ros else {}
            ros_label = f"patient_reported_{len(ros) + 1}"
            ros[ros_label] = [patient_answer]
            object.__setattr__(result, "ros", ros)
            print(f"[LoopGuard] Force-filled ROS '{ros_label}' = ['{patient_answer}'] to break ROS repeat loop")

            if len(ros) < ROS_REQUIRED:
                object.__setattr__(result, "reply", f"Thank you. Are there any other symptoms you've been experiencing?")
            else:
                object.__setattr__(result, "reply", "Thank you — I have everything I need.")

    # ── ROS Hallucination Guard: LLM can only ADD one new ROS system per turn ──
    try:
        prev_state = json.loads(current_json)
        prev_ros = prev_state.get("ros") or {}
    except Exception:
        prev_ros = {}
    new_ros_keys = [k for k in result.ros if k not in prev_ros]
    if len(new_ros_keys) > 1:
        print(f"[ROSGuard] LLM added {len(new_ros_keys)} new ROS systems in one turn: {new_ros_keys}. Keeping only first.")
        allowed_ros = dict(prev_ros)
        allowed_ros[new_ros_keys[0]] = result.ros[new_ros_keys[0]]
        object.__setattr__(result, "ros", allowed_ros)

    print(f"[{time.time():.3f}] [Graph Node] LLM returned. Preparing node dictionaries...")

    stage = compute_stage(result)
    missing = missing_from(result)
    reply = result.reply or "Could you tell me more?"

    # All fields complete — build the brief inline so it's available this turn
    if stage == "done":
        from datetime import datetime, timezone
        brief = ClinicalBrief(
            chief_complaint=result.chief_complaint or "Not specified",
            hpi=HPI(
                onset=result.onset or "Not specified",
                location=result.location or "Not specified",
                duration=result.duration or "Not specified",
                character=result.character or "Not specified",
                severity=result.severity or "Not specified",
                aggravating=result.aggravating or "Not specified",
                relieving=result.relieving or "Not specified",
            ),
            ros=result.ros,
            generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        return {
            "messages": [{"role": "assistant", "content": "Your clinical summary is ready. Please wait for the doctor."}],
            "clinical_state": result.model_dump_json(),
            "missing_fields": [],
            "frontend_stage": "done",
            "current_node": "done",
            "clinical_brief": brief.model_dump(),
        }

    return {
        "messages": [{"role": "assistant", "content": reply}],
        "clinical_state": result.model_dump_json(),
        "missing_fields": missing,
        "frontend_stage": stage,
        "current_node": "agent",
    }


# -------------------------------------------------------------- graph build --

def build_graph():
    workflow = StateGraph(IntakeState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("agent", agent_node)

    def route_triage(state: IntakeState) -> str:
        return state.get("current_node", "agent")

    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges("triage", route_triage, {"done": END, "agent": "agent"})
    workflow.add_edge("agent", END)

    checkpointer = MemorySaver()
    # Interrupt after agent so it pauses for user input each turn
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["agent"]
    )

    return graph, checkpointer