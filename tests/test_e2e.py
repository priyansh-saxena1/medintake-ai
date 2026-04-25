import os
os.environ["MOCK_LLM"] = "true"

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas import ClinicalBrief

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

@pytest.mark.asyncio(loop_scope="function")
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mock_mode"] is True

@pytest.mark.asyncio(loop_scope="function")
async def test_emergency_triage_guardrail(client):
    """If user types 'crushing chest pain', the triage node should immediately abort to 'done'."""
    session_id = "test_emergency"
    
    await client.post("/chat", json={"session_id": session_id, "message": "hello"})
    
    response = await client.post("/chat", json={"session_id": session_id, "message": "I am having crushing chest pain"})
    assert response.status_code == 200
    data = response.json()
    
    assert data["state"] == "done"
    assert "911" in data["reply"] or "emergency" in data["reply"].lower()

@pytest.mark.asyncio(loop_scope="function")
async def test_shadow_extractor_logic(client):
    """
    Test that the shadow extractor gracefully fills in missing information behind the scenes,
    transitioning the frontend stage from hpi to ros and finally done.
    """
    session_id = "test_extraction"
    
    await client.post("/chat", json={"session_id": session_id, "message": "hello"})
    
    # 1. Chief Complaint & some HPI
    # The mock LLM maps "chest pain" -> CC, "yesterday" -> onset
    res = await client.post("/chat", json={"session_id": session_id, "message": "I have chest pain since yesterday"})
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "hpi" # Needs more HPI info
    
    # 2. More HPI info
    res = await client.post("/chat", json={"session_id": session_id, "message": "It is constant pressure in the center. Severity is 7. Walking makes it worse, rest helps."})
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "ros" # Completes HPI, moves to ROS
    
    # 3. ROS info
    res = await client.post("/chat", json={"session_id": session_id, "message": "I have palpitations and shortness of breath. No nausea."})
    assert res.status_code == 200
    data = res.json()
    
    # Should be done
    assert data["state"] == "done"
    assert data["brief"] is not None
    
    brief = ClinicalBrief.model_validate(data["brief"])
    assert brief.chief_complaint == "chest pain"
    assert brief.hpi.onset == "yesterday"
    assert brief.hpi.location == "center of chest"
    assert brief.hpi.duration == "constant"
    assert brief.hpi.character == "tight pressure"
    assert brief.hpi.severity == "7/10"
    assert brief.hpi.aggravating == "walking"
    assert brief.hpi.relieving == "resting"
    
    assert "cardiac" in brief.ros
    assert "respiratory" in brief.ros
    assert "gi" in brief.ros
