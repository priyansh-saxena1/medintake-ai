# app/graph.py

import os
import json
from typing import Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.llm import (
    get_llm, CombinedOutput, HPI_FIELDS, ROS_REQUIRED,
    _fallback_reply, _next_ros_system, _ROS_QUESTIONS
)
from app.schemas import ClinicalBrief, HPI

_MOCK = lambda: os.environ.get("MOCK_LLM", "true").lower() == "true"

_ROS_SKIP_WORDS = frozenset({
    "next", "skip", "ok", "okay", "yes", "no", "sure", "continue",
    "move on", "go on", "proceed", "nothing", "none", "nope", "yep",
    "yes i am", "yes i do", "i am", "i do"
})


def add_messages(left: list[dict], right: list[dict]) -> list[dict]:
    return left + right


class IntakeState(TypedDict):
    messages: Annotated[list[dict], add_messages]
    clinical_state: str
    missing_fields: list[str]
    current_node: str
    clinical_brief: Optional[dict]
    frontend_stage: str


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
    real_ros = {k: v for k, v in state.ros.items() if not k.startswith("patient_reported_")}
    if len(real_ros) < ROS_REQUIRED:
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
    real_ros = {k: v for k, v in state.ros.items() if not k.startswith("patient_reported_")}
    if len(real_ros) < ROS_REQUIRED:
        missing.append(f"ROS ({ROS_REQUIRED - len(real_ros)} more systems needed)")
    return missing


def _count_hpi_filled(cs: CombinedOutput) -> int:
    return sum(1 for f in ["chief_complaint"] + HPI_FIELDS if getattr(cs, f))


def _detect_repeat(state) -> bool:
    msgs = state.get("messages", [])
    replies = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
    return len(replies) >= 2 and replies[-1] == replies[-2]


def _is_valid_ros_finding(text: str) -> bool:
    t = text.strip().lower()
    if t in _ROS_SKIP_WORDS:
        return False
    return len(t.split()) >= 2


def _clean_ros_for_brief(ros: dict) -> dict:
    """Remove patient_reported_N placeholder keys before writing to the clinical brief."""
    return {k: v for k, v in ros.items() if not k.startswith("patient_reported_")}


# ------------------------------------------------------------------- nodes ---

def triage_node(state: IntakeState) -> dict:
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
    import time
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

    print(f"[{time.time():.3f}] [Graph Node] Requesting LLM inference (stage={current_stage})...")

    llm = get_llm()
    result: CombinedOutput = llm.combined_call(transcript, current_json, stage=current_stage)

    try:
        prev_cs = CombinedOutput.model_validate_json(current_json)
        prev_ros = prev_cs.ros or {}
        prev_hpi_count = _count_hpi_filled(prev_cs)
    except Exception:
        prev_cs = CombinedOutput()
        prev_ros = {}
        prev_hpi_count = 0

    curr_hpi_count = _count_hpi_filled(result)
    new_hpi_extracted = curr_hpi_count > prev_hpi_count

    # ── Loop Guard ──────────────────────────────────────────────────────────
    if _detect_repeat({"messages": msgs + [{"role": "assistant", "content": result.reply}]}):

        if new_hpi_extracted:
            fixed_reply = _fallback_reply(result)
            object.__setattr__(result, "reply", fixed_reply)
            print(f"[LoopGuard] Reply-only fix: '{fixed_reply}'")

        else:
            hpi_filled = all(getattr(result, f, None) for f in HPI_FIELDS)

            if not hpi_filled:
                for stuck_field in HPI_FIELDS:
                    if getattr(result, stuck_field, None) is None:
                        object.__setattr__(result, stuck_field, "not specified")
                        print(f"[LoopGuard] Force-filled HPI '{stuck_field}' = 'not specified'")
                        fixed_reply = _fallback_reply(result)
                        object.__setattr__(result, "reply", fixed_reply)
                        break
            else:
                # ── ROS stuck: use deterministic next system ──────────────
                patient_answer = ""
                for m in reversed(msgs):
                    if m.get("role") == "user":
                        patient_answer = m.get("content", "").strip()
                        break

                real_prev_ros = {k: v for k, v in prev_ros.items() if not k.startswith("patient_reported_")}
                next_sys = _next_ros_system(result.chief_complaint or "", set(result.ros.keys()))

                if _is_valid_ros_finding(patient_answer) and next_sys:
                    ros = dict(prev_ros)
                    existing = ros.get(next_sys, [])
                    if patient_answer not in existing:
                        ros[next_sys] = existing + [patient_answer]
                    object.__setattr__(result, "ros", ros)
                    print(f"[LoopGuard] Force-filled ROS '{next_sys}' += ['{patient_answer}']")
                else:
                    print(f"[LoopGuard] Skipping non-answer '{patient_answer}'")

                fixed_reply = _fallback_reply(result)
                object.__setattr__(result, "reply", fixed_reply)

                if len({k for k in result.ros if not k.startswith("patient_reported_")}) >= ROS_REQUIRED:
                    object.__setattr__(result, "reply", "Thank you — I have all the information I need.")

    # ── ROS Hallucination Guard ──────────────────────────────────────────────
    new_ros_keys = [k for k in result.ros if k not in prev_ros]
    if len(new_ros_keys) > 1:
        print(f"[ROSGuard] LLM added {len(new_ros_keys)} systems at once. Keeping first only.")
        allowed_ros = dict(prev_ros)
        allowed_ros[new_ros_keys[0]] = result.ros[new_ros_keys[0]]
        object.__setattr__(result, "ros", allowed_ros)

    # ── Filter bare-affirmative ROS findings ─────────────────────────────────
    cleaned_ros = {}
    for sys_name, findings in result.ros.items():
        valid = [f for f in findings if _is_valid_ros_finding(f)]
        if valid or sys_name in prev_ros:
            cleaned_ros[sys_name] = valid if valid else findings
    object.__setattr__(result, "ros", cleaned_ros)

    print(f"[{time.time():.3f}] [Graph Node] LLM returned.")

    stage = compute_stage(result)
    missing = missing_from(result)
    reply = result.reply or _fallback_reply(result)

    if stage == "done":
        from datetime import datetime, timezone

        hpi_obj = HPI(
            onset=result.onset or "Not specified",
            location=result.location or "Not specified",
            duration=result.duration or "Not specified",
            character=result.character or "Not specified",
            severity=result.severity or "Not specified",
            aggravating=result.aggravating or "Not specified",
            relieving=result.relieving or "Not specified",
        )

        # Strip placeholder keys before writing to brief
        clean_ros = _clean_ros_for_brief(result.ros)

        brief_data = {
            "chief_complaint": result.chief_complaint or "Not specified",
            "hpi": hpi_obj.model_dump(),
            "ros": clean_ros,
        }

        narrative = llm.generate_brief_narrative(brief_data)

        brief = ClinicalBrief(
            chief_complaint=result.chief_complaint or "Not specified",
            hpi=hpi_obj,
            ros=clean_ros,
            generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            narrative=narrative,
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
    graph = workflow.compile(checkpointer=checkpointer, interrupt_after=["agent"])
    return graph, checkpointer