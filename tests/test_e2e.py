import os

os.environ["MOCK_LLM"] = "true"

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


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
async def test_full_intake_flow(client):
    session_id = "test1"

    response = await client.post("/chat", json={"session_id": session_id, "message": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["reply"]
    assert data["state"] in ["intake", "hpi"]

    responses = [
        "I have chest pain since this morning",
        "It started about 3 hours ago",
        "In the center of my chest",
        "It has been constant",
        "It feels like pressure",
        "About a 7 out of 10",
        "It gets worse when I walk",
        "Resting helps a little",
        "palpitations present, no syncope",
        "mild shortness of breath, no cough",
        "done",
    ]

    final_data = None
    for resp_text in responses:
        response = await client.post("/chat", json={"session_id": session_id, "message": resp_text})
        assert response.status_code == 200
        final_data = response.json()

    assert final_data is not None
    assert final_data["state"] == "done"
    assert "brief" in final_data
    assert final_data["brief"] is not None

    brief = final_data["brief"]
    assert "chief_complaint" in brief
    assert "hpi" in brief
    assert "ros" in brief


@pytest.mark.asyncio(loop_scope="function")
async def test_hpi_reprompt(client):
    session_id = "test_vague"

    await client.post("/chat", json={"session_id": session_id, "message": "hello"})
    await client.post("/chat", json={"session_id": session_id, "message": "I have chest pain"})

    response = await client.post("/chat", json={"session_id": session_id, "message": "When did it start?"})

    vague_response = await client.post("/chat", json={"session_id": session_id, "message": "I don't know"})
    assert vague_response.status_code == 200
    data = vague_response.json()
    assert "specific" in data["reply"].lower() or "when" in data["reply"].lower()


@pytest.mark.asyncio(loop_scope="function")
async def test_ros_scoping(client):
    session_id = "test_chest_pain"

    await client.post("/chat", json={"session_id": session_id, "message": "hello"})
    await client.post("/chat", json={"session_id": session_id, "message": "I have chest pain"})

    hpi_responses = [
        "It started 3 hours ago",
        "In the center of my chest",
        "It has been constant",
        "It feels like pressure",
        "7 out of 10",
        "Walking makes it worse",
        "Resting helps",
    ]

    for resp in hpi_responses:
        await client.post("/chat", json={"session_id": session_id, "message": resp})

    ros_response = await client.post("/chat", json={"session_id": session_id, "message": "ready for ROS"})
    ros_data = ros_response.json()

    await client.post("/chat", json={"session_id": session_id, "message": "cardiac:palpitations,no syncope|respiratory:shortness of breath,no cough"})

    final_response = await client.post("/chat", json={"session_id": session_id, "message": "done"})
    final_data = final_response.json()

    if final_data.get("brief"):
        ros_keys = list(final_data["brief"]["ros"].keys())
        assert "cardiac" in ros_keys or "respiratory" in ros_keys


@pytest.mark.asyncio(loop_scope="function")
async def test_brief_structure(client):
    session_id = "test_brief"

    messages = [
        "hello",
        "I have chest pain",
        "It started 3 hours ago",
        "In the center of my chest",
        "Constant",
        "Pressure-like",
        "7 out of 10",
        "Walking worsens it",
        "Resting helps",
        "cardiac:palpitations,no syncope|respiratory:shortness of breath,no cough",
    ]

    for msg in messages:
        response = await client.post("/chat", json={"session_id": session_id, "message": msg})
        assert response.status_code == 200

    final_data = response.json()

    if final_data.get("brief"):
        brief = final_data["brief"]
        from app.schemas import ClinicalBrief
        validated = ClinicalBrief.model_validate(brief)

        assert validated.chief_complaint
        assert validated.hpi.onset
        assert validated.hpi.location
        assert validated.hpi.duration
        assert validated.hpi.character
        assert validated.hpi.severity
        assert validated.hpi.aggravating
        assert validated.hpi.relieving
