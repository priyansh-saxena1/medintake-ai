from typing import Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


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
    "severity": "how severe the pain is on a 1-10 scale",
    "aggravating": "what makes your symptoms worse",
    "relieving": "what helps relieve your symptoms",
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


def intake_node(state: IntakeState) -> dict:
    messages = state.get("messages", [])
    last_idx = state.get("last_processed_message_index", 0)
    cc = state.get("chief_complaint", "")

    has_new_user_msg = len(messages) > last_idx

    if not cc and has_new_user_msg:
        user_msg = messages[-1]
        if user_msg.get("role") == "user":
            cc = user_msg.get("content", "")
            reply = f"I understand you're experiencing {cc}. Let me ask you some questions about this."
        else:
            reply = "Hello, I'm conducting your pre-visit clinical intake. What brings you in today?"
    elif not cc:
        reply = "Hello, I'm conducting your pre-visit clinical intake. What brings you in today?"
    else:
        return {
            "current_node": "hpi",
        }

    return {
        "messages": [{"role": "assistant", "content": reply}],
        "chief_complaint": cc,
        "current_node": "hpi",
        "ros_systems": state.get("ros_systems", []),
        "ros_current_index": state.get("ros_current_index", 0),
        "ros_pending_system": state.get("ros_pending_system"),
        "last_processed_message_index": len(messages) if has_new_user_msg else last_idx,
        "vague_retry_field": None,
    }


def hpi_node(state: IntakeState) -> dict:
    messages = state.get("messages", [])
    last_idx = state.get("last_processed_message_index", 0)
    hpi = dict(state.get("hpi", {}))
    vague_retry_field = state.get("vague_retry_field")

    next_field = vague_retry_field
    if not next_field:
        for field in HPI_FIELDS:
            if field not in hpi or not hpi.get(field):
                next_field = field
                break

    if next_field is None:
        reply = "Thank you for providing that information. Now let me ask about other symptoms."
        return {
            "messages": [{"role": "assistant", "content": reply}],
            "current_node": "ros",
            "last_processed_message_index": len(messages),
            "vague_retry_field": None,
        }

    has_new_user_msg = len(messages) > last_idx
    
    if has_new_user_msg:
        user_msg = None
        for i in range(last_idx, len(messages)):
            if messages[i].get("role") == "user":
                user_msg = messages[i]
                break
        
        if user_msg:
            answer = user_msg.get("content", "")

            if _is_vague_answer(answer):
                field_context = HPI_FIELD_CONTEXT.get(next_field, "your symptoms")
                reply = f"Could you be more specific about {field_context}?"
                return {
                    "messages": [{"role": "assistant", "content": reply}],
                    "current_node": "hpi",
                    "last_processed_message_index": last_idx,
                    "vague_retry_field": next_field,
                }

            hpi[next_field] = extract_hpi_value(answer, next_field)

            next_idx = HPI_FIELDS.index(next_field)
            if next_idx < len(HPI_FIELDS) - 1:
                next_q = HPI_FIELDS[next_idx + 1]
                reply = HPI_QUESTIONS[next_q]
                next_node = "hpi"
            else:
                reply = "Thank you. Now let me ask about other associated symptoms."
                next_node = "ros"

            return {
                "messages": [{"role": "assistant", "content": reply}],
                "hpi": hpi,
                "current_node": next_node,
                "last_processed_message_index": len(messages),
                "vague_retry_field": None,
            }

    reply = HPI_QUESTIONS[next_field]
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

    ros_systems = state.get("ros_systems", [])
    if not ros_systems:
        ros_systems = get_relevant_ros_systems(cc)

    current_idx = state.get("ros_current_index", 0)
    pending_system = state.get("ros_pending_system")

    if current_idx >= len(ros_systems):
        reply = "Thank you. I have enough information to generate your clinical brief."
        return {
            "messages": [{"role": "assistant", "content": reply}],
            "current_node": "brief_generator",
            "ros_systems": ros_systems,
            "ros_current_index": current_idx,
            "ros_pending_system": None,
            "last_processed_message_index": len(messages),
            "vague_retry_field": None,
        }

    has_new_user_msg = len(messages) > last_idx

    if has_new_user_msg:
        user_msg = messages[-1]
        if user_msg.get("role") == "user":
            answer = user_msg.get("content", "")

            if pending_system:
                positive_findings = []
                negative_findings = []

                findings = [f.strip() for f in answer.split(",")]
                for f in findings:
                    f_lower = f.lower()
                    if "no " in f_lower or "none" in f_lower:
                        negative_findings.append(f)
                    else:
                        positive_findings.append(f)

                ros[pending_system] = positive_findings + negative_findings

            if current_idx < len(ros_systems):
                next_system = ros_systems[current_idx]
                reply = f"Let's review your {next_system} system. Any {next_system} symptoms? Please mention what's present and what's not."
                return {
                    "messages": [{"role": "assistant", "content": reply}],
                    "ros": ros,
                    "current_node": "ros",
                    "ros_systems": ros_systems,
                    "ros_current_index": current_idx + 1,
                    "ros_pending_system": next_system,
                    "last_processed_message_index": len(messages),
                    "vague_retry_field": None,
                }
            else:
                reply = "Thank you. I have enough information."
                return {
                    "messages": [{"role": "assistant", "content": reply}],
                    "ros": ros,
                    "current_node": "brief_generator",
                    "ros_systems": ros_systems,
                    "ros_current_index": current_idx,
                    "ros_pending_system": None,
                    "last_processed_message_index": len(messages),
                    "vague_retry_field": None,
                }

    if current_idx < len(ros_systems):
        next_system = ros_systems[current_idx]
        reply = f"Let's start with your {next_system} system. Any {next_system} symptoms? Please mention what's present and what's not."
        return {
            "messages": [{"role": "assistant", "content": reply}],
            "current_node": "ros",
            "ros_systems": ros_systems,
            "ros_current_index": current_idx + 1,
            "ros_pending_system": next_system,
            "last_processed_message_index": last_idx,
            "vague_retry_field": None,
        }

    reply = "Continuing review of systems..."
    return {
        "messages": [{"role": "assistant", "content": reply}],
        "current_node": "ros",
        "ros_systems": ros_systems,
        "ros_current_index": current_idx,
        "ros_pending_system": None,
        "last_processed_message_index": last_idx,
        "vague_retry_field": None,
    }


from datetime import datetime, timezone
from app.schemas import HPI as HPIModel, ClinicalBrief as ClinicalBriefModel


def brief_generator_node(state: IntakeState) -> dict:
    ros = state.get("ros", {})
    hpi_data = state.get("hpi", {})

    hpi_obj = HPIModel(
        onset=hpi_data.get("onset") or "not specified",
        location=hpi_data.get("location") or "not specified",
        duration=hpi_data.get("duration") or "not specified",
        character=hpi_data.get("character") or "not specified",
        severity=hpi_data.get("severity") or "not specified",
        aggravating=hpi_data.get("aggravating") or "not specified",
        relieving=hpi_data.get("relieving") or "not specified",
    )

    brief = ClinicalBriefModel(
        chief_complaint=state.get("chief_complaint", ""),
        hpi=hpi_obj,
        ros=ros,
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    reply = "Your clinical intake is complete. Here is your summary."
    return {
        "messages": [{"role": "assistant", "content": reply}],
        "current_node": "done",
        "clinical_brief": brief.model_dump(),
        "ros_systems": state.get("ros_systems", []),
        "ros_current_index": state.get("ros_current_index", 0),
        "ros_pending_system": None,
        "last_processed_message_index": len(state.get("messages", [])),
        "vague_retry_field": None,
    }


def route_from_intake(state: IntakeState) -> str:
    return "hpi"


def route_from_hpi(state: IntakeState) -> str:
    hpi = state.get("hpi", {})
    all_filled = all(hpi.get(f) for f in HPI_FIELDS)
    return "ros" if all_filled else "hpi"


def route_from_ros(state: IntakeState) -> str:
    ros_systems = state.get("ros_systems", [])
    current_index = state.get("ros_current_index", 0)
    all_processed = current_index >= len(ros_systems)
    return "brief_generator" if all_processed else "ros"


def build_graph() -> tuple:
    workflow = StateGraph(IntakeState)

    workflow.add_node("intake", intake_node)
    workflow.add_node("hpi", hpi_node)
    workflow.add_node("ros", ros_node)
    workflow.add_node("brief_generator", brief_generator_node)

    workflow.add_edge(START, "intake")
    workflow.add_conditional_edges("intake", route_from_intake, {"hpi": "hpi"})
    workflow.add_conditional_edges("hpi", route_from_hpi, {"hpi": "hpi", "ros": "ros"})
    workflow.add_conditional_edges("ros", route_from_ros, {"ros": "ros", "brief_generator": "brief_generator"})
    workflow.add_edge("brief_generator", END)

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer, interrupt_after=["intake", "hpi", "ros"])

    return graph, checkpointer
