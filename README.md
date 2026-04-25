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
Patient → triage_node → agent_node → (done or loop back for next question)
```

### Inference Engine

- **Local dev (mock)**: `MOCK_LLM=true` — regex-based MockLLM, 0ms latency
- **Production**: `MOCK_LLM=false` — **Ollama** local server (`qwen2.5:0.5b`, C++ optimized)
  - ~2s per turn on CPU vs 25s with raw PyTorch

### State Graph Nodes

1. **triage_node**: Detects acute emergency phrases → immediate 🚨 alert
2. **agent_node**: Single LLM call — extracts all HPI/ROS fields AND generates next question  
   When all fields complete, builds ClinicalBrief inline (no extra LLM call)

## Deployment on Hugging Face Spaces

This repo is configured as a **Docker SDK Space**. On every push:

1. Docker image builds — Ollama gets installed via official install script
2. `startup.sh` starts on container boot: launches Ollama, pulls `qwen2.5:0.5b`, starts FastAPI
3. App is live on port 7860

```bash
# Test the Docker build locally before pushing
docker build -t clinical-intake .
docker run -p 7860:7860 clinical-intake
```

## Local Development

```bash
# Fast mock mode (no model needed, instant responses)
MOCK_LLM=true uvicorn app.main:app --reload

# Real Ollama mode — requires Ollama installed at localhost:11434
ollama serve &
ollama pull qwen2.5:0.5b
MOCK_LLM=false uvicorn app.main:app --reload
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
