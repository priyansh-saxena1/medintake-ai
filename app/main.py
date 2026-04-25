import argparse
import json
import os

from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import build_graph
from app.schemas import ClinicalBrief
from langgraph.types import Command


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    state: str
    brief: ClinicalBrief | None = None


app = FastAPI(title="Clinical Intake Agent")

graph, checkpointer = build_graph()


def get_current_node(session_id: str) -> str:
    """Get current node from checkpoint."""
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            return snapshot.values.get("current_node", "intake")
    except Exception:
        pass
    return "intake"


def get_last_reply(session_id: str) -> str:
    """Get last assistant reply from checkpoint."""
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            messages = snapshot.values.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
    except Exception:
        pass
    return ""


def get_brief(session_id: str) -> dict | None:
    """Get clinical brief from checkpoint."""
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            return snapshot.values.get("clinical_brief")
    except Exception:
        pass
    return None


@app.get("/health")
async def health():
    mock_mode = os.environ.get("MOCK_LLM", "false").lower() == "true"
    return {"status": "ok", "mock_mode": mock_mode}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.session_id}}
    
    # Get current checkpoint state
    snapshot = graph.get_state(config)
    
    # Check if graph is interrupted and waiting for input
    if snapshot.next:
        # First update state with the user message
        graph.update_state(config, {"messages": [{"role": "user", "content": request.message}]})
        # Then resume execution
        result = graph.invoke(None, config=config)
    else:
        # New conversation - start fresh
        input_state = {"messages": [{"role": "user", "content": request.message}]}
        result = graph.invoke(input_state, config=config)
    
    current_node = get_current_node(request.session_id)
    reply = get_last_reply(request.session_id)
    brief_dict = get_brief(request.session_id)
    
    return ChatResponse(reply=reply, state=current_node, brief=brief_dict)


def run_cli():
    print("=" * 60)
    print("Clinical Intake Agent - CLI Mode")
    print("=" * 60)
    print("Type your responses. The intake will end when complete.\n")

    session_id = "cli_session"
    
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        config = {"configurable": {"thread_id": session_id}}
        
        # Build input state from checkpoint or start fresh
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values and snapshot.values.get("messages"):
            # Continue existing conversation - only pass the new user message
            # The Annotated reducer will append it to existing messages
            input_state = {"messages": [{"role": "user", "content": user_input}]}
        else:
            input_state = {"messages": [{"role": "user", "content": user_input}]}
        
        result = graph.invoke(input_state, config=config)
        
        current_node = get_current_node(session_id)
        reply = get_last_reply(session_id)
        brief = get_brief(session_id)

        print(f"\nAgent: {reply}\n")

        if current_node == "done" and brief:
            print("=" * 60)
            print("CLINICAL INTAKE COMPLETE")
            print("=" * 60)
            print(json.dumps(brief, indent=2))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Intake Agent")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)
