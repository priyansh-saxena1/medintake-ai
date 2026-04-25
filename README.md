---
title: Clinical Intake Agent
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Clinical Intake Agent

A LangGraph-based conversational agent for conducting pre-visit clinical intakes with simulated patients. The agent generates a structured ClinicalBrief (Chief Complaint, HPI, ROS) at the end of the conversation.

## Features

- **Multi-turn conversation** with stateful memory using LangGraph checkpointing
- **Structured clinical data collection**: Chief Complaint, HPI (OPQRST), and ROS
- **Conditional ROS scoping**: Adapts review of systems based on chief complaint
- **Vague answer handling**: Gracefully re-prompts when patient responses are unclear
- **Dual mode**: Runs as FastAPI web app OR CLI tool
- **Mock/Real LLM**: Switch between mock responses and real local LLM via environment variable

## Architecture

```
intake → hpi → ros → brief_generation → done
```

### State Graph (LangGraph TypedDict)

```python
class IntakeState(TypedDict):
    messages: list[dict]           # conversation history
    chief_complaint: str
    hpi: dict                      # onset, location, duration, character, severity, aggravating, relieving
    ros: dict[str, list[str]]      # system -> [positive findings, negative findings]
    current_node: str
    clinical_brief: Optional[ClinicalBrief]
    ros_systems: list[str]
    ros_current_index: int
    ros_pending_system: Optional[str]
    last_processed_message_index: int
    vague_retry_field: Optional[str]
```

### Nodes

1. **intake_node**: Greets patient, extracts chief complaint. Moves to hpi when CC is clear.
2. **hpi_node**: Asks OPQRST questions one at a time. Re-prompts gracefully on vague answers.
3. **ros_node**: CONDITIONAL - scopes ROS systems based on CC (e.g., chest pain → cardiac, respiratory, GI).
4. **brief_generator_node**: Generates Pydantic ClinicalBrief from state (no LLM call).

## Installation

### Local Development

```bash
# Clone repository
git clone <repo-url>
cd clinical-intake-agent

# Install dependencies
pip install -r requirements.txt

# Run with Mock LLM (default)
export MOCK_LLM=true
uvicorn app.main:app --reload

# Run with Real LLM (requires model download)
export MOCK_LLM=false
uvicorn app.main:app --reload
```

### Docker (HuggingFace Spaces)

```bash
# Build and run locally
docker build -t clinical-intake-agent .
docker run -p 7860:7860 -e MOCK_LLM=true clinical-intake-agent
```

## Usage

### FastAPI Web App

#### Health Check
```bash
curl http://localhost:7860/health
# Response: {"status": "ok", "mock_mode": true}
```

#### Chat Endpoint
```bash
# Start conversation
curl -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "patient123", "message": "hello"}'

# Continue conversation
curl -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "patient123", "message": "I have chest pain"}'

# Final response includes clinical_brief when state == "done"
```

### CLI Mode

```bash
# Run interactive CLI
python app/main.py --cli

# Example session:
# Agent: Hello! I'm here to help you with your pre-visit intake. What brings you in today?
# You: I have chest pain since this morning
# Agent: I understand you're experiencing chest pain. When did it first start?
# ... (continues through HPI and ROS) ...
# Agent: Your clinical intake is complete. Here is your summary:
# {
#   "chief_complaint": "chest pain",
#   "hpi": {...},
#   "ros": {...},
#   "generated_at": "2024-01-15T10:30:00Z"
# }
```

## API Reference

### POST /chat

**Request:**
```json
{
  "session_id": "string",
  "message": "string"
}
```

**Response:**
```json
{
  "reply": "string",
  "state": "intake|hpi|ros|brief_generation|done",
  "brief": {
    "chief_complaint": "string",
    "hpi": {
      "onset": "string",
      "location": "string",
      "duration": "string",
      "character": "string",
      "severity": "string",
      "aggravating": "string",
      "relieving": "string"
    },
    "ros": {
      "system_name": ["finding1", "finding2"]
    },
    "generated_at": "ISO8601 timestamp"
  }
}
```

### GET /health

**Response:**
```json
{
  "status": "ok",
  "mock_mode": true
}
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `MOCK_LLM` | Use mock LLM responses (`true`) or real local LLM (`false`) | `true` |
| `MODEL_PATH` | Path to GGUF model file (used when `MOCK_LLM=false`) | `/models/qwen2.5-0.5b-instruct-q4_k_m.gguf` |

## Testing

```bash
# Run all tests (uses MockLLM automatically)
pytest tests/

# Run specific test
pytest tests/test_e2e.py::test_full_intake_flow -v

# Run with coverage
pytest --cov=app tests/
```

### Test Coverage

- ✅ `test_health_endpoint`: Verifies health check returns mock_mode status
- ✅ `test_full_intake_flow`: Complete conversation flow from greeting to ClinicalBrief
- ✅ `test_hpi_reprompt`: Validates vague answer re-prompting behavior
- ✅ `test_ros_scoping`: Confirms ROS systems are scoped based on chief complaint
- ✅ `test_brief_structure`: Validates ClinicalBrief Pydantic schema compliance

## Project Structure

```
clinical-intake-agent/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app + CLI entry point
│   ├── graph.py         # LangGraph state graph and nodes
│   ├── state.py         # TypedDict state definitions
│   ├── schemas.py       # Pydantic models (HPI, ClinicalBrief)
│   └── llm.py           # LLM provider (MockLLM, RealLLM)
├── tests/
│   ├── __init__.py
│   └── test_e2e.py      # End-to-end tests
├── requirements.txt
├── Dockerfile
├── README.md
```

## Dependencies

Minimal dependencies (no heavy ML libraries unless `MOCK_LLM=false`):

- `langgraph` - State graph orchestration
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `pytest` + `pytest-asyncio` - Testing
- `httpx` - Async HTTP client for tests
- `llama-cpp-python` - Only in Docker prod layer for real LLM mode

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### Model Download Fails

If running with `MOCK_LLM=false` and the model fails to download:

```bash
# Manually download the model
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/Qwen2.5-0.5B-Instruct-GGUF', 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf', local_dir='/models')"
```

### Session State Not Persisting

Ensure you're using the same `session_id` across multiple `/chat` calls. Sessions are stored in-memory per process.

### Docker Build Fails

The Dockerfile skips model download if `MOCK_LLM=true`. To force model download in Docker:

```bash
docker build --build-arg MOCK_LLM=false -t clinical-intake-agent .
```
