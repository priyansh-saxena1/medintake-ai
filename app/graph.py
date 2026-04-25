import os
import json
from typing import Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.llm import get_llm
from app.schemas import ClinicalStateExtraction, ClinicalBrief, HPI

_MOCK = lambda: os.environ.get("MOCK_LLM", "true").lower() == "true"

def add_messages(left: list[dict], right: list[dict]) -> list[dict]:
    return left + right

class IntakeState(TypedDict):
    messages: Annotated[list[dict], add_messages]
    clinical_state: str  # JSON representation of ClinicalStateExtraction
    missing_fields: list[str]
    current_node: str
    clinical_brief: Optional[dict]
    frontend_stage: str # 'intake', 'hpi', 'ros', or 'done'

# -------------------- HELPER FUNCTIONS --------------------

HPI_REQUIRED = ["onset", "location", "duration", "character", "severity", "aggravating", "relieving"]
ROS_REQUIRED_COUNT = 3

def format_transcript(messages: list[dict]) -> str:
    out = []
    # Only send the last couple of turns to not overwhelm if it's long, but ideally all
    for m in messages:
        role = "AI" if m["role"] == "assistant" else "Patient"
        out.append(f"{role}: {m['content']}")
    return "\n".join(out)

def evaluate_missing(state: ClinicalStateExtraction) -> (list[str], str):
    """
    Returns list of missing fields and the 'frontend_stage' mapped mapping.
    """
    missing = []
    stage = "intake"
    
    if not state.chief_complaint:
        missing.append("chief complaint (reason for visit)")
        return missing, stage
        
    stage = "hpi"
    for field in HPI_REQUIRED:
        val = getattr(state.hpi, field)
        if not val or val.lower() == "not specified":
            missing.append(f"HPI: {field}")
            
    if missing:
        return missing, stage
        
    stage = "ros"
    # Need at least a few systems covered if possible
    if len(state.ros.keys()) < ROS_REQUIRED_COUNT:
        missing.append(f"Review of Systems (ask about {ROS_REQUIRED_COUNT - len(state.ros.keys())} more bodily systems)")
        return missing, stage
        
    return [], "done"


# -------------------- NODES --------------------

def triage_node(state: IntakeState) -> dict:
    msgs = state.get("messages", [])
    if not msgs:
        return {"current_node": "triage"}
    
    last_msg = msgs[-1]
    if last_msg["role"] == "user":
        content = last_msg["content"].lower()
        emergencies = ["suicide", "kill myself", "crushing chest pain", "can't breathe", "heart attack"]
        if any(e in content for e in emergencies):
            return {
                "messages": [{"role": "assistant", "content": "🚨 EMERGENCY OVERRIDE: Your symptoms sound like a medical emergency. Please call 911 or visit the nearest emergency room immediately."}],
                "current_node": "done",
                "frontend_stage": "done"
            }
    
    return {"current_node": "extractor"}


def extractor_node(state: IntakeState) -> dict:
    msgs = state.get("messages", [])
    if not msgs:
        # Initial state setup
        return {
            "clinical_state": ClinicalStateExtraction().model_dump_json(),
            "current_node": "evaluator"
        }
    
    # Only run extractor if the last message was from the user
    if msgs[-1]["role"] != "user":
        return {"current_node": "evaluator"}
        
    llm = get_llm()
    transcript = format_transcript(msgs)
    
    current_state_json = state.get("clinical_state")
    if not current_state_json:
        current_state_json = ClinicalStateExtraction().model_dump_json()
        
    # Extractor Agent updates the state passively
    new_state = llm.ask_json(transcript, current_state_json, ClinicalStateExtraction)
    
    # Check if the extractor detected a latent emergency
    if new_state.emergency_detected:
         return {
            "messages": [{"role": "assistant", "content": "🚨 EMERGENCY OVERRIDE: Based on your details, you require immediate medical attention. Call 911."}],
            "current_node": "done",
            "frontend_stage": "done",
            "clinical_state": new_state.model_dump_json()
        }
    
    return {
        "clinical_state": new_state.model_dump_json(),
        "current_node": "evaluator"
    }


def evaluator_node(state: IntakeState) -> dict:
    state_json = state.get("clinical_state")
    if not state_json:
        clinical_state = ClinicalStateExtraction()
    else:
        clinical_state = ClinicalStateExtraction.model_validate_json(state_json)
        
    missing, stage = evaluate_missing(clinical_state)
    
    if not missing:
        return {
            "missing_fields": missing,
            "frontend_stage": "done",
            "current_node": "scribe"
        }
        
    return {
        "missing_fields": missing,
        "frontend_stage": stage,
        "current_node": "conversationalist"
    }


def conversationalist_node(state: IntakeState) -> dict:
    msgs = state.get("messages", [])
    clinical_json = state.get("clinical_state", "{}")
    missing = state.get("missing_fields", [])
    
    if not msgs:
        return {
            "messages": [{"role": "assistant", "content": "Hello, I'm conducting your pre-visit clinical intake. What brings you in today?"}],
            "current_node": "conversationalist"
        }
        
    # Check if the agent just spoke (prevent double-speaking if no user input)
    if msgs[-1]["role"] == "assistant":
        return {"current_node": "conversationalist"}

    # Dynamic target targeting the top missing field
    target = missing[0] if missing else "general details"
    
    system_prompt = (
        "You are an empathetic clinical intake assistant. "
        "Your sole job is to ask the next logical medical question in a conversational way. "
        f"We currently know this info about the patient:\n{clinical_json}\n\n"
        f"YOUR GOAL: You MUST naturally uncover the following missing information: {target}. "
        "Keep your response to exactly ONE question. Be concise and friendly."
    )
    
    transcript = format_transcript(msgs[-6:]) # Context window
    llm = get_llm()
    reply = llm.ask(f"Transcript:\n{transcript}\n\nAsk the next question about: {target}.", system=system_prompt)
    
    return {
        "messages": [{"role": "assistant", "content": reply}],
        "current_node": "conversationalist"
    }


def scribe_node(state: IntakeState) -> dict:
    state_json = state.get("clinical_state")
    data = ClinicalStateExtraction.model_validate_json(state_json)
    
    from datetime import datetime, timezone
    
    brief = ClinicalBrief(
        chief_complaint=data.chief_complaint or "Not specified",
        hpi=data.hpi,
        ros=data.ros,
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    return {
        "messages": [{"role": "assistant", "content": "Thank you — I have everything I need. Your clinical summary is ready."}],
        "current_node": "done",
        "clinical_brief": brief.model_dump(),
    }


def build_graph():
    workflow = StateGraph(IntakeState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("conversationalist", conversationalist_node)
    workflow.add_node("scribe", scribe_node)

    def route_triage(state: IntakeState) -> str:
        # If triage marked it 'done' (emergency), skip everything
        return state.get("current_node", "extractor")
        
    def route_extractor(state: IntakeState) -> str:
        # Extractor marks it 'done' if latent emergency, else 'evaluator'
        return state.get("current_node", "evaluator")
        
    def route_evaluator(state: IntakeState) -> str:
        return state.get("current_node", "conversationalist")

    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges("triage", route_triage, {"done": END, "extractor": "extractor"})
    workflow.add_conditional_edges("extractor", route_extractor, {"done": END, "evaluator": "evaluator"})
    workflow.add_conditional_edges("evaluator", route_evaluator, {"conversationalist": "conversationalist", "scribe": "scribe"})
    
    workflow.add_edge("conversationalist", END)
    workflow.add_edge("scribe", END)

    checkpointer = MemorySaver()
    # Interrupt after conversationalist so it waits for user input
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["conversationalist"]
    )

    return graph, checkpointer