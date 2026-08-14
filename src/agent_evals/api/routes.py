"""FastAPI router endpoints."""

import json
import secrets
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_evals import __version__
from agent_evals.agents.backends.http_step import HttpStepError, validate_endpoint_url
from agent_evals.agents.registry import agent_adapter_registry
from agent_evals.agents.runtime import agent_runtime_registry
from agent_evals.api.job_store import JobQueueFull, JobStoreError
from agent_evals.api.jobs import (
    SUPPORTED_EXECUTION_KEYS,
    EvaluationJob,
    EvaluationJobManager,
    IdempotencyConflict,
)
from agent_evals.benchmarks.registry import benchmark_registry, benchmark_spec_registry
from agent_evals.core.config import get_settings
from agent_evals.core.types import StatusEnum

LOCAL_ENVIRONMENTS = frozenset({"development", "testing", "local"})


def _require_api_key(authorization: str | None = Header(default=None)) -> None:
    """Require a bearer token; only local environments may run without one."""
    settings = get_settings()
    configured = settings.api.api_key or ""
    if not configured:
        if settings.environment.lower() in LOCAL_ENVIRONMENTS:
            return
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured; set AGENT_EVALS_API__API_KEY",
        )
    expected = f"Bearer {configured}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="authentication required")


# Health stays public so container orchestration probes work without credentials.
public_router = APIRouter(prefix="/v1", tags=["health"])
router = APIRouter(prefix="/v1", tags=["evaluations"], dependencies=[Depends(_require_api_key)])
job_manager = EvaluationJobManager()


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    features: list[str] = Field(default_factory=list)


def _validate_agent_endpoint(value: Any) -> None:
    """Validate the public endpoint without accepting a second secret channel."""
    if not isinstance(value, str):
        raise ValueError("agent_endpoint must be an absolute http(s) URL")
    settings = get_settings()
    try:
        validate_endpoint_url(
            value,
            allow_private=settings.api.allow_private_agent_endpoints,
        )
    except HttpStepError as error:
        raise ValueError(str(error)) from error


def _validate_integer_override(
    overrides: dict[str, Any], key: str, maximum: int
) -> None:
    """Validate a bounded integer override before it reaches a worker."""
    value = overrides.get(key)
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise ValueError(f"config_override.{key} must be between 1 and {maximum}")


class RunBenchmarkRequest(BaseModel):
    benchmark_id: str
    agent_id: str
    model_config = ConfigDict(extra="forbid")
    model: str | None = Field(default=None, max_length=256)
    provider: str | None = Field(default=None, max_length=128)
    task_id: str | None = None
    dataset_id: str | None = None
    #: URL of the submitted black-box agent. Authentication remains in the
    #: backend environment (``SCAIB_AGENT_TOKEN``), not in the API payload.
    agent_endpoint: str | None = Field(default=None, max_length=2048)
    test_mode: bool = False
    seed: int = Field(default=0, ge=0)
    max_cells: int | None = Field(default=None, ge=1, le=10000)
    max_steps: int | None = Field(default=None, ge=1, le=32)
    config_override: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_overrides(self) -> "RunBenchmarkRequest":  # noqa: C901
        if len(self.config_override) > 32:
            raise ValueError("config_override may contain at most 32 keys")
        if len(json.dumps(self.config_override, separators=(",", ":"), default=str)) > 16_384:
            raise ValueError("config_override must be at most 16 KiB when serialized")
        unknown = sorted(set(self.config_override) - SUPPORTED_EXECUTION_KEYS)
        if unknown:
            raise ValueError(
                "config_override contains unsupported execution key(s): "
                + ", ".join(unknown)
            )
        for key in ("test_mode",):
            value = self.config_override.get(key)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"config_override.{key} must be a boolean")
        override_seed = self.config_override.get("seed")
        if override_seed is not None and (
            isinstance(override_seed, bool)
            or not isinstance(override_seed, int)
            or override_seed < 0
        ):
            raise ValueError("config_override.seed must be a non-negative integer")
        endpoint = self.agent_endpoint or self.config_override.get("agent_endpoint")
        requested_test_mode = self.test_mode or bool(self.config_override.get("test_mode", False))
        if self.agent_id == "http-step" and not endpoint:
            raise ValueError(
                "agent_id 'http-step' requires agent_endpoint; the endpoint is the "
                "agent boundary and cannot be inferred from a provider setting"
            )
        if self.agent_id != "http-step" and endpoint is not None:
            raise ValueError("agent_endpoint is only valid when agent_id is 'http-step'")
        if requested_test_mode and endpoint is not None:
            raise ValueError(
                "test_mode and agent_endpoint are mutually exclusive runtime choices"
            )
        for key in ("task_id", "dataset_id"):
            value = self.config_override.get(key)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"config_override.{key} must be a non-empty string")
        if endpoint is not None:
            _validate_agent_endpoint(endpoint)
        for key, maximum in (("max_cells", 10000), ("max_steps", 32)):
            _validate_integer_override(self.config_override, key, maximum)
        return self


class RunBenchmarkResponse(BaseModel):
    job_id: str
    benchmark_id: str
    agent_id: str
    task_id: str | None = None
    dataset_id: str | None = None
    seed: int
    status: StatusEnum
    request_sha256: str
    resolved_execution: dict[str, Any] = Field(default_factory=dict)


def _ensure_benchmark_specs() -> None:
    if benchmark_spec_registry.list_ids():
        return
    root = Path("examples/benchmarks")
    if root.exists():
        benchmark_spec_registry.discover(root, replace=True)


@public_router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Return API service health status."""
    return HealthCheckResponse(
        status="ok",
        version=__version__,
        features=[
            "glm_test_mode",
            "sse_events",
            "idempotent_runs",
            "durable_job_store",
            "restart_recovery",
            "dedicated_worker_mode",
        ],
    )


@public_router.get("/ready")
async def readiness_check() -> dict[str, Any]:
    """Report whether the API control plane can accept durable work."""
    try:
        job_manager.ready()
    except JobStoreError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if not job_manager.started:
        raise HTTPException(status_code=503, detail="scheduler is not started")
    return {
        "status": "ready" if job_manager.started else "starting",
        "job_store": "ok",
        "scheduler": (
            "running"
            if job_manager.started and job_manager.executes_jobs
            else "standby"
            if job_manager.started
            else "not_started"
        ),
    }


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
        "tasks": [
            {
                **item.model_dump(mode="json"),
                "end_goal": item.end_goal
                or (
                    f"Complete '{item.name}' and produce the strongest defensible "
                    "result supported by the available observations and required artifacts."
                ),
                "required_artifacts": sorted(
                    specification.required_task_artifacts(item)
                ),
                "stopping_criteria": [
                    condition.model_dump(mode="json")
                    for condition in item.termination
                ],
            }
            for item in specification.tasks
        ],
        "actions": [item.model_dump(mode="json") for item in specification.actions],
        "metrics": [item.model_dump(mode="json") for item in specification.metrics],
    }


@router.post("/evaluations/run", response_model=RunBenchmarkResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_evaluation_with_background(  # noqa: C901
    payload: RunBenchmarkRequest,
    background_tasks: BackgroundTasks,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> RunBenchmarkResponse:
    """Submit a validated run and execute it after the response is accepted."""
    _ensure_benchmark_specs()
    try:
        specification = benchmark_spec_registry.get(payload.benchmark_id)
    except Exception as error:
        raise HTTPException(status_code=404, detail="benchmark not found") from error
    tasks_by_id = {task.id: task for task in specification.tasks}
    selected_task = tasks_by_id.get(payload.task_id) if payload.task_id else None
    if payload.task_id is not None and selected_task is None:
        raise HTTPException(
            status_code=422,
            detail=f"task '{payload.task_id}' is not declared by benchmark '{payload.benchmark_id}'",
        )
    if payload.dataset_id is not None:
        allowed_datasets = (
            set(selected_task.datasets)
            if selected_task is not None
            else {dataset_id for task in specification.tasks for dataset_id in task.datasets}
        )
        if payload.dataset_id not in allowed_datasets:
            scope = f"task '{payload.task_id}'" if payload.task_id else "this benchmark"
            raise HTTPException(
                status_code=422,
                detail=f"dataset '{payload.dataset_id}' is not declared for {scope}",
            )
    known_agents = set(agent_adapter_registry.list_types()) | set(agent_runtime_registry.list())
    if payload.agent_id not in known_agents:
        raise HTTPException(
            status_code=404,
            detail=f"agent '{payload.agent_id}' is not registered",
        )
    try:
        job_id = job_manager.create(payload, idempotency_key=idempotency_key)
    except IdempotencyConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except JobQueueFull as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    except JobStoreError as error:
        raise HTTPException(status_code=503, detail="durable job store unavailable") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    job = job_manager.get(job_id)
    # In development the API can own a small BackgroundTasks worker. Production
    # sets execute_jobs_in_process=false and a dedicated ``agent-evals worker``
    # process polls the same durable store, so API restarts do not interrupt
    # scientific work and multiple web workers remain responsive.
    if get_settings().api.execute_jobs_in_process:
        background_tasks.add_task(job_manager.execute, job_id)
    return RunBenchmarkResponse(
        job_id=job_id,
        benchmark_id=job.benchmark_id,
        agent_id=job.agent_id,
        task_id=job.task_id,
        dataset_id=job.dataset_id,
        seed=job.seed,
        status=job.status,
        request_sha256=job.request_sha256,
        resolved_execution=job.resolved_execution,
    )


@router.get("/evaluations", response_model=list[EvaluationJob])
async def list_evaluation_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    include_results: bool = Query(default=False),
) -> list[EvaluationJob]:
    """List bounded job summaries; fetch one job for its full result payload."""
    jobs = job_manager.list()[:limit]
    if include_results:
        return jobs
    return [
        job.model_copy(update={"result": None, "logs": job.logs[-20:]})
        for job in jobs
    ]


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
            serialized = json.dumps(event, separators=(",", ":"))
            event_id = event.get("event_id")
            if isinstance(event_id, int):
                yield f"id: {event_id}\n"
            yield f"data: {serialized}\n\n"

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
