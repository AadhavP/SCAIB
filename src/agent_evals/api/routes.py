"""FastAPI router endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from agent_evals.agents.registry import agent_adapter_registry
from agent_evals.benchmarks.registry import benchmark_registry
from agent_evals.core.types import StatusEnum

router = APIRouter(prefix="/v1", tags=["evaluations"])


class HealthCheckResponse(BaseModel):
    status: str
    version: str


class RunBenchmarkRequest(BaseModel):
    benchmark_id: str
    agent_id: str
    config_override: dict[str, Any] = {}


class RunBenchmarkResponse(BaseModel):
    job_id: str
    benchmark_id: str
    agent_id: str
    status: StatusEnum


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Return API service health status."""
    return HealthCheckResponse(status="ok", version="0.1.0")


@router.get("/benchmarks", response_model=list[str])
async def list_benchmarks() -> list[str]:
    """List all registered benchmark IDs."""
    return benchmark_registry.list_ids()


@router.get("/agents", response_model=list[str])
async def list_agents() -> list[str]:
    """List all registered agent adapter types."""
    return agent_adapter_registry.list_types()


@router.post(
    "/evaluations/run",
    response_model=RunBenchmarkResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_evaluation(payload: RunBenchmarkRequest) -> RunBenchmarkResponse:
    """Trigger a benchmark evaluation run."""
    try:
        # Check if benchmark is registered
        _ = benchmark_registry.get(payload.benchmark_id)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(err)
        ) from err

    return RunBenchmarkResponse(
        job_id="job_placeholder_12345",
        benchmark_id=payload.benchmark_id,
        agent_id=payload.agent_id,
        status=StatusEnum.PENDING,
    )
