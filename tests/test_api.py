"""Tests for FastAPI endpoints."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from agent_evals.api import jobs, routes
from agent_evals.api.main import app

client = TestClient(app)


def test_api_health() -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "glm_test_mode" in data["features"]


def test_api_list_benchmarks() -> None:
    response = client.get("/v1/benchmarks")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_api_list_agents() -> None:
    response = client.get("/v1/agents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_api_lists_runtime_agent_metadata() -> None:
    response = client.get("/v1/agents")
    assert response.status_code == 200
    agents = response.json()
    assert any(item["id"] == "openai" for item in agents)
    assert all("capabilities" in item for item in agents)


def test_api_returns_benchmark_metadata() -> None:
    response = client.get("/v1/benchmarks/pbmc-cell-annotation")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "pbmc-cell-annotation"
    assert data["tasks"]
    assert data["actions"]
    task = data["tasks"][0]
    assert task["end_goal"]
    assert task["required_artifacts"]
    assert task["stopping_criteria"][0]["condition"]


def test_api_job_manager_tracks_successful_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRun:
        run_id = "run-123"

        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"run_id": self.run_id, "global_reward": {"value": 0.9}}

    captured: dict[str, object] = {}

    async def fake_run(*args: object, **kwargs: object) -> FakeRun:
        captured.update(kwargs)
        return FakeRun()

    monkeypatch.setattr(jobs.ScientificLoop, "run", fake_run)
    job_id = routes.job_manager.create(
        routes.RunBenchmarkRequest(
            benchmark_id="pbmc-cell-annotation",
            agent_id="rule-based",
            max_cells=10,
            max_steps=2,
            test_mode=True,
        )
    )
    asyncio.run(routes.job_manager.execute(job_id))
    assert captured["test_mode"] is True
    job = routes.job_manager.get(job_id)
    assert job.status == "COMPLETED"
    assert job.result == {"run_id": "run-123", "global_reward": {"value": 0.9}}
    response = client.get(f"/v1/evaluations/{job_id}/events")
    assert response.status_code == 200
    assert '"type":"run_queued"' in response.text
    assert '"terminal":true' in response.text
    assert "\\\\n" not in response.text
    assert "id: 1\n" in response.text


def test_api_reports_the_seed_the_run_actually_uses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job records the resolved seed so clients never assume a default."""
    captured: dict[str, object] = {}

    class FakeRun:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            return {"run_id": "run-seed"}

    async def fake_run(*args: object, **kwargs: object) -> FakeRun:
        captured.update(kwargs)
        return FakeRun()

    monkeypatch.setattr(jobs.ScientificLoop, "run", fake_run)
    job_id = routes.job_manager.create(
        routes.RunBenchmarkRequest(
            benchmark_id="pbmc-cell-annotation",
            agent_id="rule-based",
            seed=1234,
        )
    )
    # Reported before execution starts, while the job is still queued.
    assert routes.job_manager.get(job_id).seed == 1234

    asyncio.run(routes.job_manager.execute(job_id))
    assert captured["seed"] == 1234
    assert routes.job_manager.get(job_id).seed == 1234


def test_api_seed_from_config_override_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A seed supplied only via `config_override` must still be visible."""
    job_id = routes.job_manager.create(
        routes.RunBenchmarkRequest(
            benchmark_id="pbmc-cell-annotation",
            agent_id="rule-based",
            config_override={"seed": 7},
        )
    )
    assert routes.job_manager.get(job_id).seed == 7


def test_api_accepts_run_and_exposes_job(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_execute(job_id: str) -> None:
        return None

    monkeypatch.setattr(routes.job_manager, "execute", fake_execute)
    response = client.post(
        "/v1/evaluations/run",
        json={
            "benchmark_id": "pbmc-cell-annotation",
            "agent_id": "rule-based",
            "max_cells": 10,
            "max_steps": 2,
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert data["job_id"]
    assert data["status"] == "PENDING"
    status = client.get(f"/v1/evaluations/{data['job_id']}")
    assert status.status_code == 200


def test_api_requires_key_outside_local_environments(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = routes.get_settings()
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings.api, "api_key", None)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    response = client.get("/v1/benchmarks")
    assert response.status_code == 503
    health = client.get("/v1/health")
    assert health.status_code == 200


def test_api_enforces_configured_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = routes.get_settings()
    monkeypatch.setattr(settings.api, "api_key", "secret-token")
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    unauthenticated = client.get("/v1/benchmarks")
    assert unauthenticated.status_code == 401
    wrong = client.get("/v1/benchmarks", headers={"Authorization": "Bearer wrong"})
    assert wrong.status_code == 401
    authorized = client.get(
        "/v1/benchmarks", headers={"Authorization": "Bearer secret-token"}
    )
    assert authorized.status_code == 200


def test_api_rejects_unbounded_or_path_like_run_options() -> None:
    oversized = client.post(
        "/v1/evaluations/run",
        json={"benchmark_id": "pbmc-cell-annotation", "agent_id": "mock", "max_cells": 10001},
    )
    assert oversized.status_code == 422
    path_like = client.post(
        "/v1/evaluations/run",
        json={
            "benchmark_id": "pbmc-cell-annotation",
            "agent_id": "mock",
            "output_dir": "../../outside",
        },
    )
    assert path_like.status_code == 422
    overridden = client.post(
        "/v1/evaluations/run",
        json={
            "benchmark_id": "pbmc-cell-annotation",
            "agent_id": "mock",
            "config_override": {"max_steps": 33},
        },
    )
    assert overridden.status_code == 422
