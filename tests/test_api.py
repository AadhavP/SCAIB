"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from agent_evals.api.main import app

client = TestClient(app)


def test_api_health() -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_api_list_benchmarks() -> None:
    response = client.get("/v1/benchmarks")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_api_list_agents() -> None:
    response = client.get("/v1/agents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
