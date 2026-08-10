"""FastAPI router endpoints."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_evals.agents.registry import agent_adapter_registry
from agent_evals.agents.runtime import agent_runtime_registry
from agent_evals.api.jobs import EvaluationJob, EvaluationJobManager
from agent_evals.benchmarks.registry import benchmark_registry, benchmark_spec_registry
from agent_evals.core.config import get_settings
from agent_evals.core.types import StatusEnum


def _require_api_key(authorization: str | None = Header(default=None)) -> None:
    """Require a bearer token when the deployment config enables one."""
    configured = get_settings().api.api_key
    if configured is None:
        return
    expected = f"Bearer {configured}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="authentication required")


router = APIRouter(prefix="/v1", tags=["evaluations"], dependencies=[Depends(_require_api_key)])
job_manager = EvaluationJobManager()


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    features: list[str] = Field(default_factory=list)


class RunBenchmarkRequest(BaseModel):
    benchmark_id: str
    agent_id: str
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    provider: str | None = None
    test_mode: bool = False
    seed: int = 0
    max_cells: int | None = Field(default=None, ge=1, le=10000)
    max_steps: int | None = Field(default=None, ge=1, le=32)
    config_override: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_overrides(self) -> "RunBenchmarkRequest":
        for key, maximum in (("max_cells", 10000), ("max_steps", 32)):
            value = self.config_override.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > maximum
            ):
                raise ValueError(f"config_override.{key} must be between 1 and {maximum}")
        return self


class RunBenchmarkResponse(BaseModel):
    job_id: str
    benchmark_id: str
    agent_id: str
    status: StatusEnum


def _ensure_benchmark_specs() -> None:
    if benchmark_spec_registry.list_ids():
        return
    root = Path("examples/benchmarks")
    if root.exists():
        benchmark_spec_registry.discover(root, replace=True)


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Return API service health status."""
    return HealthCheckResponse(
        status="ok",
        version="0.1.0",
        features=["glm_test_mode", "sse_events"],
    )


@router.get("/benchmarks", response_model=list[str])
async def list_benchmarks() -> list[str]:
    """List all registered benchmark IDs."""
    _ensure_benchmark_specs()
    return benchmark_spec_registry.list_ids() or benchmark_registry.list_ids()


@router.get("/agents", response_model=list[dict[str, Any]])
async def list_agents() -> list[dict[str, Any]]:
    """List legacy adapters and universal runtimes with public capabilities."""
    legacy = [
        {"id": name, "type": "adapter", "capabilities": [], "available": True}
        for name in agent_adapter_registry.list_types()
    ]
    runtimes = [
        {
            "id": name,
            "type": agent_runtime_registry.manifest(name).type,
            "capabilities": agent_runtime_registry.manifest(name).capabilities,
            "available": True,
        }
        for name in agent_runtime_registry.list()
    ]
    return legacy + runtimes


@router.get("/benchmarks/{benchmark_id}")
async def benchmark_details(benchmark_id: str) -> dict[str, Any]:
    """Return the declarative benchmark details needed by the run console."""
    _ensure_benchmark_specs()
    try:
        specification = benchmark_spec_registry.get(benchmark_id)
    except Exception as error:
        raise HTTPException(status_code=404, detail="benchmark not found") from error
    return {
        "id": specification.metadata.id,
        "title": specification.metadata.title,
        "description": specification.metadata.description,
        "version": specification.metadata.version,
        "tags": specification.metadata.tags,
        "datasets": [item.model_dump(mode="json") for item in specification.datasets],
        "tasks": [item.model_dump(mode="json") for item in specification.tasks],
        "actions": [item.model_dump(mode="json") for item in specification.actions],
        "metrics": [item.model_dump(mode="json") for item in specification.metrics],
    }


@router.post("/evaluations/run", response_model=RunBenchmarkResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_evaluation_with_background(
    payload: RunBenchmarkRequest,
    background_tasks: BackgroundTasks,
) -> RunBenchmarkResponse:
    """Submit a validated run and execute it after the response is accepted."""
    _ensure_benchmark_specs()
    try:
        benchmark_spec_registry.get(payload.benchmark_id)
    except Exception as error:
        raise HTTPException(status_code=404, detail="benchmark not found") from error
    known_agents = set(agent_adapter_registry.list_types()) | set(agent_runtime_registry.list())
    if payload.agent_id not in known_agents:
        raise HTTPException(
            status_code=404,
            detail=f"agent '{payload.agent_id}' is not registered",
        )
    try:
        job_id = job_manager.create(payload)
    except RuntimeError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    background_tasks.add_task(job_manager.execute, job_id)
    return RunBenchmarkResponse(
        job_id=job_id,
        benchmark_id=payload.benchmark_id,
        agent_id=payload.agent_id,
        status=StatusEnum.PENDING,
    )


@router.get("/evaluations", response_model=list[EvaluationJob])
async def list_evaluation_jobs() -> list[EvaluationJob]:
    """List jobs submitted to this API process."""
    return job_manager.list()


@router.get("/evaluations/{job_id}/events")
async def stream_evaluation_events(
    job_id: str,
    request: Request,
    after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None),
) -> StreamingResponse:
    """Stream replayable evaluation events as server-sent events."""
    try:
        job_manager.get(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="evaluation job not found") from error
    try:
        cursor = max(after, int(last_event_id or 0))
    except ValueError:
        cursor = after

    async def event_stream() -> Any:
        async for event in job_manager.events(job_id, after=cursor):
            if await request.is_disconnected():
                break
            if event.get("type") == "heartbeat":
                yield ": heartbeat\n\n"
                continue
            yield (
                f"id: {event['event_id']}\n"
                f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/evaluations/{job_id}", response_model=EvaluationJob)
async def get_evaluation_job(job_id: str) -> EvaluationJob:
    """Return current status and, when complete, the serialized run result."""
    try:
        return job_manager.get(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="evaluation job not found") from error
