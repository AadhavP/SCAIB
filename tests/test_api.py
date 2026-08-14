"""Tests for FastAPI endpoints."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from agent_evals.api import jobs, routes
from agent_evals.api.jobs import EvaluationJobManager, IdempotencyConflict
from agent_evals.api.main import app, create_app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_api_job_manager():
    """Keep API tests independent from the production SQLite control plane."""
    previous = routes.job_manager
    manager = EvaluationJobManager(db_path=":memory:")
    routes.job_manager = manager
    try:
        yield
    finally:
        manager.close()
        routes.job_manager = previous


def test_api_health() -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "glm_test_mode" in data["features"]
    assert "durable_job_store" in data["features"]


def test_api_readiness_exposes_control_plane_state() -> None:
    response = client.get("/v1/ready")
    # The module-level test client does not enter the application lifespan, so
    # readiness must fail closed until startup has actually completed.
    assert response.status_code == 503
    assert response.json()["detail"] == "scheduler is not started"


def test_api_readiness_becomes_ready_after_lifecycle_start() -> None:
    asyncio.run(routes.job_manager.start(execute_jobs=False))
    try:
        response = client.get("/v1/ready")
    finally:
        asyncio.run(routes.job_manager.shutdown())

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "job_store": "ok",
        "scheduler": "standby",
    }


def test_production_app_does_not_publish_interactive_api_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = routes.get_settings()
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr("agent_evals.api.main.get_settings", lambda: settings)

    production_app = create_app()

    assert production_app.docs_url is None
    assert production_app.redoc_url is None
    assert production_app.openapi_url is None


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

    async def read_after_terminal() -> list[dict[str, object]]:
        return [
            event
            async for event in routes.job_manager.events(job_id, after=10_000)
        ]

    assert asyncio.run(read_after_terminal()) == []
    summaries = client.get("/v1/evaluations").json()
    listed = next(item for item in summaries if item["job_id"] == job_id)
    assert listed["result"] is None
    detailed = client.get("/v1/evaluations?include_results=true").json()
    listed_detail = next(item for item in detailed if item["job_id"] == job_id)
    assert listed_detail["result"]["run_id"] == "run-123"


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


def test_api_preserves_selected_task_and_dataset_for_the_worker() -> None:
    """A multi-task benchmark must not silently fall back to its first task."""
    job_id = routes.job_manager.create(
        routes.RunBenchmarkRequest(
            benchmark_id="pbmc-cell-annotation",
            agent_id="http-step",
            task_id="cell-annotation",
            dataset_id="pbmc68k",
            agent_endpoint="https://agent.example/step",
        )
    )

    job = routes.job_manager.get(job_id)
    assert job.task_id == "cell-annotation"
    assert job.dataset_id == "pbmc68k"
    resolved = jobs.resolve_execution(
        routes.RunBenchmarkRequest(
            benchmark_id="pbmc-cell-annotation",
            agent_id="http-step",
            agent_endpoint="https://agent.example/step",
            config_override={"task_id": "cell-annotation", "dataset_id": "pbmc68k"},
        ).model_dump(exclude_none=True)
    )
    assert resolved["task_id"] == "cell-annotation"
    assert resolved["dataset_id"] == "pbmc68k"


def test_api_idempotency_key_replays_the_same_job_without_a_second_episode() -> None:
    """Transport retries must not duplicate a scientific run."""
    payload = routes.RunBenchmarkRequest(
        benchmark_id="pbmc-cell-annotation",
        agent_id="mock",
        seed=11,
        max_cells=10,
    )
    first = routes.job_manager.create(payload, idempotency_key="client-retry-1")
    replay = routes.job_manager.create(payload, idempotency_key="client-retry-1")
    assert replay == first
    assert routes.job_manager.get(first).request_sha256
    with pytest.raises(IdempotencyConflict, match="different evaluation request"):
        routes.job_manager.create(
            payload.model_copy(update={"seed": 12}),
            idempotency_key="client-retry-1",
        )


def test_api_idempotency_uses_the_resolved_experiment_not_request_spelling() -> None:
    """Equivalent explicit/default and override forms must replay one job."""
    omitted = routes.RunBenchmarkRequest(
        benchmark_id="pbmc-cell-annotation",
        agent_id="mock",
    )
    explicit = omitted.model_copy(update={"seed": 0})
    overridden = omitted.model_copy(update={"config_override": {"seed": 0}})

    first = routes.job_manager.create(omitted, idempotency_key="canonical-retry")
    assert routes.job_manager.create(explicit, idempotency_key="canonical-retry") == first
    assert routes.job_manager.create(overridden, idempotency_key="canonical-retry") == first


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


def test_api_can_enqueue_without_running_science_in_the_web_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = routes.get_settings()
    settings.api.execute_jobs_in_process = False
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    called = False

    async def fake_execute(job_id: str) -> None:
        nonlocal called
        del job_id
        called = True

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
    assert called is False


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


def test_api_rejects_oversized_request_bodies_before_validation() -> None:
    response = client.post(
        "/v1/evaluations/run",
        content=b"{" + b"x" * (64 * 1024) + b"}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
