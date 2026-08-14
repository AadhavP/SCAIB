"""Durable API job-control-plane tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_evals.api.jobs import EvaluationJobManager, IdempotencyConflict
from agent_evals.api.routes import RunBenchmarkRequest
from agent_evals.core.types import StatusEnum


def _request(seed: int = 0) -> RunBenchmarkRequest:
    return RunBenchmarkRequest(
        benchmark_id="pbmc-cell-annotation",
        agent_id="mock",
        seed=seed,
        max_cells=10,
    )


def test_job_manager_survives_process_reconstruction(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    first = EvaluationJobManager(db_path=database)
    job_id = first.create(_request(), idempotency_key="restart-safe")
    first.close()

    second = EvaluationJobManager(db_path=database)
    job = second.get(job_id)
    assert job.status is StatusEnum.PENDING
    events = second._store.events_after(job_id)
    assert events[0]["type"] == "run_queued"
    assert events[0]["event_id"] == 1
    second.close()


def test_idempotency_is_atomic_across_manager_instances(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    first = EvaluationJobManager(db_path=database)
    second = EvaluationJobManager(db_path=database)

    job_id = first.create(_request(seed=4), idempotency_key="same-request")
    assert second.create(_request(seed=4), idempotency_key="same-request") == job_id
    with pytest.raises(IdempotencyConflict):
        second.create(_request(seed=5), idempotency_key="same-request")
    first.close()
    second.close()


def test_interrupted_running_jobs_are_not_replayed(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    first = EvaluationJobManager(db_path=database)
    job_id = first.create(_request())
    first._update(
        job_id,
        {"status": StatusEnum.RUNNING, "current_stage": "Agent request"},
        "run_started",
        "worker started",
    )
    first.close()

    second = EvaluationJobManager(db_path=database)
    # Recovery is a worker-start operation, after singleton lease acquisition;
    # constructing an API/worker manager must not mutate another live worker's
    # records.
    assert second.get(job_id).status is StatusEnum.RUNNING
    asyncio.run(second.start(execute_jobs=True))
    recovered = second.get(job_id)
    assert recovered.status is StatusEnum.FAILED
    assert recovered.finished_at is not None
    assert recovered.error is not None
    assert "not replayed" in recovered.error
    events = second._store.events_after(job_id)
    assert events[-1]["type"] == "run_recovered"
    assert events[-1]["terminal"] is True
    asyncio.run(second.shutdown())
    second.close()


def test_event_gap_is_explicit_and_terminal_replay_still_finishes(tmp_path: Path) -> None:
    """SSE clients can detect retention gaps without a server-side stream error."""
    manager = EvaluationJobManager(db_path=tmp_path / "jobs.sqlite3")
    manager._store.max_events = 2
    job_id = manager.create(_request())
    manager._publish(job_id, "progress", "first progress update")
    manager._publish(job_id, "run_completed", "finished", terminal=True)

    async def collect() -> list[dict[str, object]]:
        return [event async for event in manager.events(job_id)]

    events = asyncio.run(collect())
    assert events[0]["type"] == "event_gap"
    assert "event_id" not in events[0]
    assert events[-1]["type"] == "run_completed"
    assert events[-1]["event_id"] == 3
    manager.close()


def test_api_only_manager_does_not_recover_a_live_worker_job(tmp_path: Path) -> None:
    """Web restarts must observe, not terminate, work owned by a worker."""
    database = tmp_path / "jobs.sqlite3"
    worker = EvaluationJobManager(db_path=database, recover_interrupted=True)
    job_id = worker.create(_request())
    worker._update(
        job_id,
        {"status": StatusEnum.RUNNING, "current_stage": "Agent request"},
        "run_started",
        "worker started",
    )
    worker.close()

    api_only = EvaluationJobManager(db_path=database, recover_interrupted=False)
    asyncio.run(api_only.start(execute_jobs=False))
    assert api_only.get(job_id).status is StatusEnum.RUNNING
    assert api_only._store.events_after(job_id)[-1]["type"] == "run_started"
    asyncio.run(api_only.shutdown())
    api_only.close()


def test_store_health_and_worker_lease_probe_are_operational(tmp_path: Path) -> None:
    """Readiness and the container health check share the durable contract."""
    from agent_evals.api.job_store import SQLiteJobStore

    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    store.ping()
    assert store.worker_lease_active() is False
    assert store.acquire_worker_lease("worker-1", lease_seconds=30) is True
    assert store.worker_lease_active() is True
    store.release_worker_lease("worker-1")
    assert store.worker_lease_active() is False
    store.close()


def test_only_one_execution_worker_can_start(tmp_path: Path) -> None:
    """Scaling the worker service cannot duplicate a non-idempotent turn."""
    database = tmp_path / "jobs.sqlite3"
    first = EvaluationJobManager(db_path=database, recover_interrupted=True)
    second = EvaluationJobManager(db_path=database, recover_interrupted=True)

    async def scenario() -> None:
        await first.start(execute_jobs=True)
        with pytest.raises(RuntimeError, match="execution lease"):
            await second.start(execute_jobs=True)
        await first.shutdown()
        await second.start(execute_jobs=True)
        await second.shutdown()

    asyncio.run(scenario())
    first.close()
    second.close()
