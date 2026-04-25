from typing import Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import os
from app.llm import get_llm

_MOCK = lambda: os.environ.get("MOCK_LLM", "true").lower() == "true"

SYSTEM_PROMPT = """
You are a clinical intake assistant.

Rules:
- Ask exactly ONE question at a time
- Keep responses under 20 words
- Be clear and direct
- No explanations unless asked
"""


def _ask(prompt: str) -> str:
    llm = get_llm()
    try:
        return llm.ask(prompt, system=SYSTEM_PROMPT)
    except TypeError:
        # fallback if system param not supported
        return llm.ask(prompt)


def add_messages(left: list[dict], right: list[dict]) -> list[dict]:
    return left + right


class IntakeState(TypedDict):
    messages: Annotated[list[dict], add_messages]
    chief_complaint: str
    hpi: dict
    ros: dict[str, list[str]]
    current_node: str
    clinical_brief: Optional[dict]
    ros_systems: list[str]
    ros_current_index: int
    ros_pending_system: Optional[str]
    last_processed_message_index: int
    vague_retry_field: Optional[str]


HPI_FIELDS = ["onset", "location", "duration", "character", "severity", "aggravating", "relieving"]

HPI_QUESTIONS = {
    "onset": "When did your symptoms first start?",
    "location": "Where exactly do you feel the pain or discomfort?",
    "duration": "How long does each episode last? Is it constant or intermittent?",
    "character": "Can you describe what the pain feels like?",
    "severity": "On a scale of 1 to 10, how severe is your pain?",
    "aggravating": "What makes your symptoms worse?",
    "relieving": "What helps relieve your symptoms?"
}

HPI_FIELD_CONTEXT = {
    "onset": "when your symptoms first started",
    "location": "where exactly you feel it",
    "duration": "how long each episode lasts",
    "character": "what the pain feels like",
    "severity": "pain severity (1-10)",
    "aggravating": "what makes symptoms worse",
    "relieving": "what relieves symptoms",
}

CC_KEYWORDS_TO_ROS = {
    "chest": ["cardiac", "respiratory", "gi"],
    "pain": ["cardiac", "respiratory", "gi"],
    "headache": ["neuro", "ent", "vision"],
    "head": ["neuro", "ent", "vision"],
    "breath": ["respiratory", "cardiac"],
    "shortness": ["respiratory", "cardiac"],
    "cough": ["respiratory", "ent"],
    "dizzy": ["neuro", "cardiac"],
    "nausea": ["gi", "constitutional"],
    "vomiting": ["gi", "constitutional"],
}

DEFAULT_ROS = ["constitutional", "cardiac", "respiratory"]


def get_relevant_ros_systems(cc: str) -> list[str]:
    cc_lower = cc.lower()
    for keyword, systems in CC_KEYWORDS_TO_ROS.items():
        if keyword in cc_lower:
            return systems
    return DEFAULT_ROS


import re


def extract_hpi_value(answer: str, field: str) -> str:
    answer = answer.strip()
    if field == "severity":
        match = re.search(r'(\d{1,2})\s*(?:out of|/)?\s*10', answer, re.IGNORECASE)
        if match:
            return f"{match.group(1)}/10"
    return answer


def _is_vague_answer(answer: str) -> bool:
    vague_phrases = ["i don't know", "not sure", "dont know", "idk", "maybe", "i guess"]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in vague_phrases)


# -------------------- NODES --------------------

GREETINGS = {"hello", "hi", "hey", "start", "begin", "ok", "okay", "yes", "sure"}

def intake_node(state: IntakeState) -> dict:
    messages = state.get("messages", [])
    last_idx = state.get("last_processed_message_index", 0)
    cc = state.get("chief_complaint", "")

    if cc:
        return {"current_node": "hpi"}

    has_new_user_msg = len(messages) > last_idx
    greeting_reply = "Hello, I'm conducting your pre-visit clinical intake. What brings you in today?"

    if has_new_user_msg:
        user_msg = next((m for m in messages[last_idx:] if m["role"] == "user"), None)
        if user_msg:
            content = user_msg["content"].strip()

            if content.lower() in GREETINGS or len(content) <= 4:
                return {
                    "messages": [{"role": "assistant", "content": greeting_reply}],
                    "chief_complaint": "",
                    "current_node": "intake",
                    "last_processed_message_index": len(messages),
                    "vague_retry_field": None,
                }

            cc = content
            if _MOCK():
                reply = f"I understand you're experiencing {cc}. Let me ask a few questions."
            else:
                reply = _ask(
                    f"Patient's chief complaint is: '{cc}'. "
                    "Acknowledge it in one sentence and say you'll ask a few questions."
                )
            return {
                "messages": [{"role": "assistant", "content": reply}],
                "chief_complaint": cc,
                "current_node": "hpi",
                "last_processed_message_index": len(messages),
                "vague_retry_field": None,
            }

    return {
        "messages": [{"role": "assistant", "content": greeting_reply}],
        "chief_complaint": "",
        "current_node": "intake",
        "last_processed_message_index": last_idx,
        "vague_retry_field": None,
    }


def hpi_node(state: IntakeState) -> dict:
    messages = state.get("messages", [])
    last_idx = state.get("last_processed_message_index", 0)
    hpi = dict(state.get("hpi", {}))
    vague_retry_field = state.get("vague_retry_field")
    cc = state.get("chief_complaint", "")

    next_field = vague_retry_field
    if not next_field:
        for field in HPI_FIELDS:
            if field not in hpi or not hpi.get(field):
                next_field = field
                break

    if next_field is None:
        return {
            "messages": [{"role": "assistant", "content": "Now I’ll ask about other symptoms."}],
            "current_node": "ros",
            "last_processed_message_index": len(messages),
            "vague_retry_field": None,
        }

    has_new_user_msg = len(messages) > last_idx

    if has_new_user_msg:
        user_msg = next((m for m in messages[last_idx:] if m["role"] == "user"), None)

        if user_msg:
            answer = user_msg["content"]

            if _is_vague_answer(answer):
                field_context = HPI_FIELD_CONTEXT[next_field]

                if _MOCK():
                    reply = f"Please be more specific about {field_context}."
                else:
                    reply = _ask(
                        f"Patient response about {field_context} was vague. "
                        "Ask for clarification in one short sentence."
                    )

                return {
                    "messages": [{"role": "assistant", "content": reply}],
                    "current_node": "hpi",
                    "last_processed_message_index": last_idx,
                    "vague_retry_field": next_field,
                }

            hpi[next_field] = extract_hpi_value(answer, next_field)

            next_idx = HPI_FIELDS.index(next_field)
            if next_idx < len(HPI_FIELDS) - 1:
                next_field = HPI_FIELDS[next_idx + 1]

                if _MOCK():
                    reply = HPI_QUESTIONS[next_field]
                else:
                    reply = _ask(
                        f"Complaint: {cc}. Known info: {hpi}. "
                        f"Ask ONE question about {HPI_FIELD_CONTEXT[next_field]}."
                    )

                return {
                    "messages": [{"role": "assistant", "content": reply}],
                    "hpi": hpi,
                    "current_node": "hpi",
                    "last_processed_message_index": len(messages),
                    "vague_retry_field": None,
                }

            return {
                "messages": [{"role": "assistant", "content": "Now I’ll ask about other symptoms."}],
                "hpi": hpi,
                "current_node": "ros",
                "last_processed_message_index": len(messages),
                "vague_retry_field": None,
            }

    if _MOCK():
        reply = HPI_QUESTIONS[next_field]
    else:
        reply = _ask(
            f"Complaint: {cc}. Known info: {hpi}. "
            f"Ask ONE question about {HPI_FIELD_CONTEXT[next_field]}."
        )

    return {
        "messages": [{"role": "assistant", "content": reply}],
        "current_node": "hpi",
        "last_processed_message_index": last_idx,
        "vague_retry_field": None,
    }


def ros_node(state: IntakeState) -> dict:
    messages = state.get("messages", [])
    last_idx = state.get("last_processed_message_index", 0)
    ros = dict(state.get("ros", {}))
    cc = state.get("chief_complaint", "")

    ros_systems = state.get("ros_systems") or get_relevant_ros_systems(cc)
    current_idx = state.get("ros_current_index", 0)
    pending = state.get("ros_pending_system")

    if current_idx >= len(ros_systems):
        return {
            "messages": [{"role": "assistant", "content": "I have enough information."}],
            "current_node": "brief_generator",
            "last_processed_message_index": len(messages),
        }

    has_new_user_msg = len(messages) > last_idx

    if has_new_user_msg and pending:
        answer = messages[-1]["content"]
        ros[pending] = [f.strip() for f in answer.split(",")]

    next_system = ros_systems[current_idx]

    if _MOCK():
        reply = f"Any {next_system} symptoms? Mention present and absent."
    else:
        reply = _ask(
            f"Ask about {next_system} symptoms. One short question. "
            "Ask for both present and absent symptoms."
        )

    return {
        "messages": [{"role": "assistant", "content": reply}],
        "ros": ros,
        "current_node": "ros",
        "ros_systems": ros_systems,
        "ros_current_index": current_idx + 1,
        "ros_pending_system": next_system,
        "last_processed_message_index": len(messages),
    }


# -------------------- FINAL --------------------

from datetime import datetime, timezone
from app.schemas import HPI as HPIModel, ClinicalBrief as ClinicalBriefModel


def brief_generator_node(state: IntakeState) -> dict:
    hpi_obj = HPIModel(**{f: state.get("hpi", {}).get(f) or "not specified" for f in HPI_FIELDS})

    brief = ClinicalBriefModel(
        chief_complaint=state.get("chief_complaint", ""),
        hpi=hpi_obj,
        ros=state.get("ros", {}),
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    return {
        "messages": [{"role": "assistant", "content": "Intake complete. Here is your summary."}],
        "current_node": "done",
        "clinical_brief": brief.model_dump(),
    }


def build_graph():
    workflow = StateGraph(IntakeState)

    workflow.add_node("intake", intake_node)
    workflow.add_node("hpi", hpi_node)
    workflow.add_node("ros", ros_node)
    workflow.add_node("brief_generator", brief_generator_node)

    def route(state: IntakeState) -> str:
        return state.get("current_node", "intake")

    workflow.add_edge(START, "intake")

    workflow.add_conditional_edges(
        "intake", route, {"intake": "intake", "hpi": "hpi"}
    )
    workflow.add_conditional_edges(
        "hpi", route, {"hpi": "hpi", "ros": "ros"}
    )
    workflow.add_conditional_edges(
        "ros", route, {"ros": "ros", "brief_generator": "brief_generator"}
    )
    workflow.add_edge("brief_generator", END)

    checkpointer = MemorySaver()
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["intake", "hpi", "ros"]
    )

    return graph, checkpointer