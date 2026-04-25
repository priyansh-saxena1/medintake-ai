from typing import Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import os
import re

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
    from app.llm import get_llm
    llm = get_llm()
    try:
        return llm.ask(prompt, system=SYSTEM_PROMPT)
    except TypeError:
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

# Questions are templated — {cc} will be replaced with chief complaint
HPI_QUESTIONS = {
    "onset": "When did {cc} start?",
    "location": "Where exactly do you feel {cc}?",
    "duration": "Is {cc} constant or does it come and go? How long does each episode last?",
    "character": "How would you describe {cc} — sharp, dull, pressure, burning?",
    "severity": "On a 1–10 scale, how severe is your {cc} right now?",
    "aggravating": "Does anything make {cc} worse, like activity or certain foods?",
    "relieving": "What helps relieve your {cc}?"
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

ROS_SYSTEM_QUESTIONS = {
    "cardiac": "Any palpitations, fluttering, or swelling in your legs or ankles?",
    "respiratory": "Any shortness of breath, wheezing, or cough?",
    "gi": "Any nausea, vomiting, heartburn, or abdominal pain?",
    "neuro": "Any headaches, dizziness, numbness, or vision changes?",
    "ent": "Any ear pain, sore throat, or sinus pressure?",
    "vision": "Any blurry vision, double vision, or eye pain?",
    "constitutional": "Any fever, chills, unexplained weight loss, or fatigue?",
}


def get_relevant_ros_systems(cc: str) -> list[str]:
    cc_lower = cc.lower()
    seen = []
    for keyword, systems in CC_KEYWORDS_TO_ROS.items():
        if keyword in cc_lower:
            for s in systems:
                if s not in seen:
                    seen.append(s)
    return seen if seen else DEFAULT_ROS


def _fmt_question(field: str, cc: str) -> str:
    """Format an HPI question, injecting the chief complaint naturally."""
    q = HPI_QUESTIONS[field]
    cc_short = cc.split()[0:4]  # first few words of complaint
    cc_str = " ".join(cc_short).lower() if cc_short else "this"
    return q.format(cc=cc_str)


def extract_hpi_value(answer: str, field: str) -> str:
    answer = answer.strip()
    if field == "severity":
        match = re.search(r'(\d{1,2})\s*(?:out of|/|over)?\s*10', answer, re.IGNORECASE)
        if match:
            return f"{match.group(1)}/10"
        # also handle bare numbers 1-10
        match2 = re.search(r'\b([1-9]|10)\b', answer)
        if match2:
            return f"{match2.group(1)}/10"
    return answer


def _is_vague_answer(answer: str) -> bool:
    vague_phrases = ["i don't know", "not sure", "dont know", "idk", "maybe", "i guess", "not really", "not sure"]
    return any(phrase in answer.lower() for phrase in vague_phrases)


def _parse_ros_answer(answer: str) -> list[str]:
    """
    Parse a free-text ROS answer into a list of individual findings.
    Handles comma-separated, 'and'-joined, and 'no X' style negative findings.
    """
    # Split on commas, semicolons, and 'and'
    parts = re.split(r'[,;]|\band\b', answer, flags=re.IGNORECASE)
    findings = []
    for part in parts:
        part = part.strip()
        if part:
            findings.append(part)
    return findings if findings else [answer.strip()]


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
                reply = f"Got it — {cc}. I'll ask a few quick questions to document your visit."
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
            "messages": [{"role": "assistant", "content": "Thank you. Now I'll ask about a few other symptoms."}],
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
                    reply = f"Could you be more specific? I need to know {field_context}."
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
                    reply = _fmt_question(next_field, cc)
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
                "messages": [{"role": "assistant", "content": "Thank you. Now I'll ask about a few other symptoms."}],
                "hpi": hpi,
                "current_node": "ros",
                "last_processed_message_index": len(messages),
                "vague_retry_field": None,
            }

    if _MOCK():
        reply = _fmt_question(next_field, cc)
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
            "messages": [{"role": "assistant", "content": "Thank you — I have everything I need."}],
            "current_node": "brief_generator",
            "last_processed_message_index": len(messages),
        }

    has_new_user_msg = len(messages) > last_idx

    if has_new_user_msg and pending:
        answer = messages[-1]["content"]
        ros[pending] = _parse_ros_answer(answer)

    next_system = ros_systems[current_idx]

    if _MOCK():
        reply = ROS_SYSTEM_QUESTIONS.get(next_system, f"Any {next_system} symptoms? Mention present and absent.")
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


def _clean_hpi_value(field: str, raw: str) -> str:
    """
    Convert a raw patient answer into a clean clinical phrase.
    Removes filler words and informal language.
    """
    raw = raw.strip()

    # Remove filler starters
    fillers = [
        r'^(yeah|yes|no|well|so|like|um|uh|i mean|i guess),?\s*',
        r'^(it\'?s?\s+)',
        r'^(the\s+)',
    ]
    for pattern in fillers:
        raw = re.sub(pattern, '', raw, flags=re.IGNORECASE).strip()

    if not raw:
        return "not specified"

    # Capitalize first letter
    return raw[0].upper() + raw[1:]


def brief_generator_node(state: IntakeState) -> dict:
    raw_hpi = state.get("hpi", {})

    # Clean each HPI field
    cleaned_hpi = {f: _clean_hpi_value(f, raw_hpi.get(f) or "not specified") for f in HPI_FIELDS}

    hpi_obj = HPIModel(**cleaned_hpi)

    # Clean ROS — ensure each system has a proper list of findings
    raw_ros = state.get("ros", {})
    cleaned_ros: dict[str, list[str]] = {}
    for system, findings in raw_ros.items():
        clean_findings = []
        for f in findings:
            f = f.strip()
            if f:
                # Capitalize
                f = f[0].upper() + f[1:]
                clean_findings.append(f)
        if clean_findings:
            cleaned_ros[system] = clean_findings

    brief = ClinicalBriefModel(
        chief_complaint=state.get("chief_complaint", ""),
        hpi=hpi_obj,
        ros=cleaned_ros,
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    return {
        "messages": [{"role": "assistant", "content": "Intake complete. Your clinical summary is ready."}],
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