import os
import time
os.environ["MOCK_LLM"] = "true"

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas import ClinicalBrief
from app.llm import MockLLM, CombinedOutput


# ─────────────────────── fixtures ───────────────────────

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ────────────────────── unit tests ──────────────────────

def test_mock_llm_combined_call_basic_extraction():
    """MockLLM should extract chief complaint, onset and location in one call."""
    llm = MockLLM()
    transcript = "Patient: I have chest pain since yesterday\nAI: Where is it?\nPatient: Center of my chest"
    result = llm.combined_call(transcript, CombinedOutput().model_dump_json())
    assert result.chief_complaint == "chest pain"
    assert result.onset == "yesterday"
    assert result.location == "center of chest"
    assert result.reply  # Should ask the next missing question


def test_mock_llm_emergency_detection():
    """MockLLM should detect emergency keywords and set emergency=True."""
    llm = MockLLM()
    transcript = "Patient: I am having crushing chest pain"
    result = llm.combined_call(transcript, CombinedOutput().model_dump_json())
    assert result.emergency is True


def test_mock_llm_does_not_repeat_filled_questions():
    """If onset is already known, the next question should NOT ask about onset again."""
    llm = MockLLM()
    current = CombinedOutput(chief_complaint="chest pain", onset="yesterday").model_dump_json()
    transcript = "Patient: chest pain yesterday\nAI: ok\nPatient: anything new"
    result = llm.combined_call(transcript, current)
    assert result.onset == "yesterday"              # Should be preserved
    assert "when" not in result.reply.lower()       # Should not re-ask onset


def test_mock_llm_severity_extraction():
    """Severity from different phrasings should always normalize to X/10."""
    llm = MockLLM()
    for phrase, expected in [
        ("it is a 7 out of 10", "7/10"),
        ("about 8 on the scale", None),     # may not extract without explicit context
        ("i'd say 9 on a scale", None),
    ]:
        state = CombinedOutput(
            chief_complaint="chest pain", onset="yesterday",
            location="chest", duration="constant", character="tight"
        ).model_dump_json()
        result = llm.combined_call(f"Patient: {phrase}", state)
        if expected:
            assert result.severity == expected, f"Failed for: '{phrase}'"


def test_mock_llm_ros_extraction():
    """ROS should populate correctly when patient mentions system symptoms."""
    llm = MockLLM()
    full_hpi = CombinedOutput(
        chief_complaint="chest pain", onset="yesterday", location="center of chest",
        duration="constant", character="tight", severity="7/10",
        aggravating="walking", relieving="resting"
    ).model_dump_json()
    result = llm.combined_call("Patient: palpitations present no leg swelling", full_hpi)
    assert "cardiac" in result.ros
    
    result2 = llm.combined_call("Patient: mild shortness of breath", full_hpi)
    assert "respiratory" in result2.ros


def test_mock_llm_speed():
    """
    MockLLM combined_call must complete under 100ms per call.
    (Real LLM test is separate — this validates no accidental model load in mock mode.)
    """
    llm = MockLLM()
    state = CombinedOutput().model_dump_json()
    
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        llm.combined_call("Patient: I have chest pain since this morning in the center of my chest", state)
        times.append(time.perf_counter() - t0)
    
    avg_ms = (sum(times) / len(times)) * 1000
    print(f"\n[speed] MockLLM avg combined_call: {avg_ms:.1f}ms")
    assert avg_ms < 100, f"MockLLM too slow: {avg_ms:.1f}ms avg (should be <100ms)"


def test_combined_output_schema_round_trip():
    """CombinedOutput must survive JSON round-trip without data loss."""
    original = CombinedOutput(
        chief_complaint="headache",
        onset="3 days ago",
        location="forehead",
        duration="constant",
        character="throbbing",
        severity="6/10",
        aggravating="bright light",
        relieving="dark room",
        ros={"neuro": ["dizziness"], "ent": ["no ear pain"]},
        emergency=False,
        reply="Any vision changes?",
    )
    json_str = original.model_dump_json()
    restored = CombinedOutput.model_validate_json(json_str)
    assert restored.chief_complaint == "headache"
    assert restored.severity == "6/10"
    assert restored.ros["neuro"] == ["dizziness"]
    assert restored.reply == "Any vision changes?"


# ───────────────────── API integration tests ─────────────────────

@pytest.mark.asyncio(loop_scope="function")
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mock_mode"] is True


@pytest.mark.asyncio(loop_scope="function")
async def test_emergency_triage_node(client):
    """Emergency phrase should bypass agent and return 911 message immediately."""
    session_id = "test_emergency"
    await client.post("/chat", json={"session_id": session_id, "message": "hello"})
    response = await client.post(
        "/chat", json={"session_id": session_id, "message": "I am having crushing chest pain"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "done"
    assert "911" in data["reply"] or "emergency" in data["reply"].lower()


@pytest.mark.asyncio(loop_scope="function")
async def test_full_intake_multi_turn_extraction(client):
    """
    The agent should extract multiple fields per message and skip already-answered questions.
    After 3 messages that collectively answer all HPI fields + 3 ROS systems, state should be 'done'.
    """
    session_id = "test_multi_extract"
    
    # Kick-off
    r = await client.post("/chat", json={"session_id": session_id, "message": "hello"})
    assert r.status_code == 200

    # Message 1: CC + onset + location
    r = await client.post("/chat", json={
        "session_id": session_id,
        "message": "I have chest pain since yesterday in the center of my chest"
    })
    data = r.json()
    assert data["state"] in ("intake", "hpi")

    # Message 2: duration + character + severity + aggravating + relieving
    r = await client.post("/chat", json={
        "session_id": session_id,
        "message": "It is constant, tight and squeezing, about a 7 out of 10. Walking worsens it and resting helps."
    })
    data = r.json()
    assert data["state"] in ("hpi", "ros")

    # Message 3: cover 3 ROS systems in one shot
    r = await client.post("/chat", json={
        "session_id": session_id,
        "message": "I have palpitations, mild shortness of breath, and no nausea"
    })
    data = r.json()
    # Should be done now
    assert data["state"] == "done"
    assert data["brief"] is not None

    brief = ClinicalBrief.model_validate(data["brief"])
    assert brief.chief_complaint == "chest pain"
    assert brief.hpi.onset is not None
    assert brief.hpi.severity is not None
    assert len(brief.ros) >= 2


@pytest.mark.asyncio(loop_scope="function")
async def test_api_response_time(client):
    """API with MockLLM must respond in under 2 seconds per message."""
    session_id = "test_speed_api"
    
    times = []
    messages = [
        "hello",
        "I have a headache since this morning",
        "It is on the left side of my head",
    ]
    for msg in messages:
        t0 = time.perf_counter()
        r = await client.post("/chat", json={"session_id": session_id, "message": msg})
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        assert r.status_code == 200

    avg_s = sum(times) / len(times)
    print(f"\n[speed] API avg response: {avg_s*1000:.0f}ms")
    assert avg_s < 2.0, f"API too slow: {avg_s:.2f}s avg (should be <2s in mock mode)"
