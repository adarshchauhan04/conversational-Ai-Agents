import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "tools_available" in data
    assert "web_search" in data["tools_available"]


def test_tools_endpoint():
    """Test /api/v1/tools endpoint."""
    response = client.get("/api/v1/tools")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 3
    tool_names = [t["name"] for t in data["tools"]]
    assert "web_search" in tool_names
    assert "calculate" in tool_names


def test_chat_endpoint_and_history():
    """Test /api/v1/chat endpoint returning structured JSON and check session history."""
    session_id = "test-api-session-999"
    payload = {
        "message": "Calculate 15 * 8",
        "session_id": session_id,
        "temperature": 0.5
    }
    
    chat_response = client.post("/api/v1/chat", json=payload)
    assert chat_response.status_code == 200
    res_data = chat_response.json()
    
    assert res_data["session_id"] == session_id
    assert "response" in res_data
    assert "structured_data" in res_data
    assert "tool_calls" in res_data
    assert "thought_process" in res_data

    # Retrieve history
    history_response = client.get(f"/api/v1/history/{session_id}")
    assert history_response.status_code == 200
    hist_data = history_response.json()
    assert hist_data["session_id"] == session_id
    assert hist_data["total_turns"] >= 2  # user + assistant

    # Delete history
    delete_response = client.delete(f"/api/v1/history/{session_id}")
    assert delete_response.status_code == 200
