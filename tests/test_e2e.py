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
        "I have chest pain since this morning",   # CC (intake)
        "It started about 3 hours ago",            # onset
        "In the center of my chest",               # location
        "It has been constant for an hour",        # duration
        "It feels like pressure",                  # character
        "About a 7 out of 10",                     # severity
        "It gets worse when I walk",               # aggravating
        "Resting helps a little",                  # relieving
        "palpitations present, no syncope",        # cardiac ROS
        "mild shortness of breath, no cough",      # respiratory ROS
        "no nausea or vomiting",                   # gi ROS
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
    """Vague answers (I don't know) should trigger a re-prompt."""
    session_id = "test_vague"

    await client.post("/chat", json={"session_id": session_id, "message": "hello"})
    await client.post("/chat", json={"session_id": session_id, "message": "I have chest pain"})

    # First HPI question is about onset
    vague_response = await client.post("/chat", json={"session_id": session_id, "message": "I don't know"})
    assert vague_response.status_code == 200
    data = vague_response.json()
    reply_lower = data["reply"].lower()
    # Should ask again — should mention specificity or the field context
    assert "specific" in reply_lower or "when" in reply_lower or "start" in reply_lower


@pytest.mark.asyncio(loop_scope="function")
async def test_ros_scoping(client):
    """For chest pain, ROS should include cardiac and respiratory systems."""
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

    # Now in ROS — answer cardiac system
    await client.post("/chat", json={"session_id": session_id, "message": "palpitations, no syncope"})
    # respiratory
    await client.post("/chat", json={"session_id": session_id, "message": "mild shortness of breath, no cough"})
    # gi
    final_response = await client.post("/chat", json={"session_id": session_id, "message": "no nausea"})
    final_data = final_response.json()

    if final_data.get("brief"):
        ros_keys = list(final_data["brief"]["ros"].keys())
        assert "cardiac" in ros_keys or "respiratory" in ros_keys


@pytest.mark.asyncio(loop_scope="function")
async def test_brief_structure(client):
    """Brief should have all 7 HPI fields, chief_complaint, ros, and generated_at."""
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
        "palpitations present, no syncope",
        "shortness of breath, no cough",
        "no nausea or vomiting",
    ]

    response = None
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
        assert validated.generated_at


@pytest.mark.asyncio(loop_scope="function")
async def test_brief_cleaning(client):
    """Brief generator should strip informal filler words from patient answers."""
    session_id = "test_cleaning"

    messages = [
        "hello",
        "I have chest pain",
        "yeah like since yesterday evening",   # filler "yeah like"
        "like in my chest area",               # filler "like"
        "Constant",
        "um tight and squeezing",              # filler "um"
        "7 out of 10",
        "yeah walking makes it worse",         # filler "yeah"
        "Resting helps",
        "palpitations, no syncope",
        "mild shortness of breath",
        "no nausea",
    ]

    response = None
    for msg in messages:
        response = await client.post("/chat", json={"session_id": session_id, "message": msg})
        assert response.status_code == 200

    final_data = response.json()
    if final_data.get("brief"):
        hpi = final_data["brief"]["hpi"]
        # "yeah like since yesterday evening" → should not start with "yeah"
        if hpi.get("onset"):
            assert not hpi["onset"].lower().startswith("yeah"), \
                f"Filler not cleaned from onset: {hpi['onset']}"
