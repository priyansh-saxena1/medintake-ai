# app/graph.py  — complete file

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


EMERGENCY_PHRASES = [
    "crushing chest pain", "can't breathe", "cannot breathe",
    "heart attack", "suicide", "kill myself", "can't move", "dying"
]

# ── Natural-language question for each HPI field (used by LoopGuard reply-fix) ──
_HPI_NEXT_Q = {
    "onset":       "when did the symptom start?",
    "location":    "where exactly in your body do you feel it?",
    "duration":    "how long has it been lasting?",
    "character":   "how would you describe the quality of the pain (sharp, dull, pressure...)?",
    "severity":    "how severe is it on a scale of 1 to 10?",
    "aggravating": "what makes it worse?",
    "relieving":   "what makes it better?",
}


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


def _count_hpi_filled(cs: CombinedOutput) -> int:
    """Count how many of chief_complaint + HPI_FIELDS are non-None."""
    return sum(1 for f in ["chief_complaint"] + HPI_FIELDS if getattr(cs, f))


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
    3. If all data is collected, builds the ClinicalBrief inline.
    """
    msgs = state.get("messages", [])

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

    try:
        pre_state = CombinedOutput.model_validate_json(current_json)
        current_stage = compute_stage(pre_state)
    except Exception:
        pre_state = CombinedOutput()
        current_stage = "intake"

    import time
    print(f"[{time.time():.3f}] [Graph Node] Requesting LLM inference (stage={current_stage})...")

    llm = get_llm()
    result: CombinedOutput = llm.combined_call(transcript, current_json, stage=current_stage)

    # ── Load previous state once — used by LoopGuard AND ROSGuard ──
    try:
        prev_cs = CombinedOutput.model_validate_json(current_json)
        prev_ros = prev_cs.ros or {}
        prev_hpi_count = _count_hpi_filled(prev_cs)
    except Exception:
        prev_cs = CombinedOutput()
        prev_ros = {}
        prev_hpi_count = 0

    curr_hpi_count = _count_hpi_filled(result)
    new_hpi_extracted = curr_hpi_count > prev_hpi_count   # ← KEY: did this turn add new data?

    # ── Loop Guard ──────────────────────────────────────────────────────────
    if _detect_repeat({"messages": msgs + [{"role": "assistant", "content": result.reply}]}):

        if new_hpi_extracted:
            # ----------------------------------------------------------------
            # LLM correctly extracted data BUT gave a stale/repeated reply.
            # FIX: advance the reply to the next missing field — do NOT force-fill.
            # ----------------------------------------------------------------
            new_missing = missing_from(result)
            if new_missing:
                m = new_missing[0]
                if m.startswith("HPI:"):
                    next_field = m.replace("HPI:", "")
                    next_q = _HPI_NEXT_Q.get(next_field, f"tell me about {next_field}")
                    fixed_reply = f"Thank you. {next_q.capitalize()}"
                else:
                    # Transition to ROS
                    fixed_reply = "Thank you — that covers the history. Now I'd like to ask about a few related body systems."
            else:
                fixed_reply = "Thank you — I have everything I need."

            object.__setattr__(result, "reply", fixed_reply)
            print(f"[LoopGuard] Reply-only fix applied (new data was extracted). New reply: '{fixed_reply}'")

        else:
            # ----------------------------------------------------------------
            # LLM extracted nothing new AND repeated itself — truly stuck.
            # Force-fill the first missing HPI field, then move on.
            # ----------------------------------------------------------------
            hpi_filled = all(getattr(result, f, None) for f in HPI_FIELDS)

            if not hpi_filled:
                for stuck_field in HPI_FIELDS:
                    if getattr(result, stuck_field, None) is None:
                        object.__setattr__(result, stuck_field, "not specified")
                        print(f"[LoopGuard] Force-filled HPI '{stuck_field}' = 'not specified' to break repeat loop")
                        new_missing = missing_from(result)
                        if new_missing:
                            m = new_missing[0]
                            if m.startswith("HPI:"):
                                next_field = m.replace("HPI:", "")
                                next_q = _HPI_NEXT_Q.get(next_field, f"tell me about {next_field}")
                                object.__setattr__(result, "reply", f"Understood. {next_q.capitalize()}")
                            else:
                                object.__setattr__(result, "reply", "Thank you — that covers the history. Now I'd like to ask about a few related body systems.")
                        else:
                            object.__setattr__(result, "reply", "Thank you — I have everything I need.")
                        break
            else:
                # ROS stuck — use the patient's last answer as a finding
                patient_answer = ""
                for m in reversed(msgs):
                    if m.get("role") == "user":
                        patient_answer = m.get("content", "denied")
                        break
                patient_answer = patient_answer or "denied"

                ros = dict(prev_ros)
                ros_label = f"patient_reported_{len(ros) + 1}"
                ros[ros_label] = [patient_answer]
                object.__setattr__(result, "ros", ros)
                print(f"[LoopGuard] Force-filled ROS '{ros_label}' = ['{patient_answer}'] to break ROS repeat loop")

                if len(ros) < ROS_REQUIRED:
                    object.__setattr__(result, "reply", "Thank you. Are there any other symptoms you've been experiencing?")
                else:
                    object.__setattr__(result, "reply", "Thank you — I have all the information I need.")

    # ── ROS Hallucination Guard: LLM can only ADD one new ROS system per turn ──
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
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["agent"]
    )

    return graph, checkpointer