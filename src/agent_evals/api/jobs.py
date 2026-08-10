"""In-process asynchronous evaluation jobs for the API service."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.core.types import StatusEnum
from agent_evals.environment.scientific_loop import ScientificLoop


class EvaluationJob(BaseModel):
    """Public state for one submitted evaluation."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    benchmark_id: str
    agent_id: str
    status: StatusEnum = StatusEnum.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class EvaluationJobManager:
    """Track API jobs while the service process is running."""

    MAX_RETAINED_JOBS = 100

    def __init__(self) -> None:
        self._jobs: dict[str, EvaluationJob] = {}
        self._requests: dict[str, dict[str, Any]] = {}
        self._semaphore = asyncio.Semaphore(2)

    def create(self, request: Any) -> str:
        active = sum(
            job.status in {StatusEnum.PENDING, StatusEnum.RUNNING}
            for job in self._jobs.values()
        )
        if active >= 8:
            raise RuntimeError("evaluation queue is full")
        self._evict_old_jobs()
        job_id = str(uuid4())
        self._jobs[job_id] = EvaluationJob(
            job_id=job_id,
            benchmark_id=request.benchmark_id,
            agent_id=request.agent_id,
        )
        self._requests[job_id] = request.model_dump(
            exclude_none=True,
            exclude_defaults=True,
        )
        return job_id

    def _evict_old_jobs(self) -> None:
        if len(self._jobs) < self.MAX_RETAINED_JOBS:
            return
        candidates = [
            job
            for job in self._jobs.values()
            if job.status not in {StatusEnum.PENDING, StatusEnum.RUNNING}
        ]
        if not candidates:
            return
        oldest = min(candidates, key=lambda job: job.created_at)
        self._jobs.pop(oldest.job_id, None)
        self._requests.pop(oldest.job_id, None)

    def get(self, job_id: str) -> EvaluationJob:
        try:
            return self._jobs[job_id].model_copy(deep=True)
        except KeyError as error:
            raise KeyError(f"evaluation job '{job_id}' was not found") from error

    def list(self) -> list[EvaluationJob]:
        return [job.model_copy(deep=True) for job in self._jobs.values()]

    async def execute(self, job_id: str) -> None:
        async with self._semaphore:
            await self._execute_one(job_id)

    async def _execute_one(self, job_id: str) -> None:
        job = self._jobs[job_id]
        self._jobs[job_id] = job.model_copy(
            update={"status": StatusEnum.RUNNING, "started_at": datetime.now(UTC)}
        )
        request = self._requests[job_id]
        try:
            overrides = request.get("config_override", {})
            supported = {"seed", "max_cells", "max_steps", "model", "provider"}
            execution = {
                key: value
                for key, value in overrides.items()
                if key in supported and key not in request
            }
            execution.update(
                {
                    key: request[key]
                    for key in supported
                    if key in request
                }
            )
            result = await ScientificLoop().run(
                request["benchmark_id"],
                agent_type=request["agent_id"],
                output_dir="results",
                seed=execution.get("seed", 0),
                max_cells=execution.get("max_cells"),
                max_steps=execution.get("max_steps"),
                model=execution.get("model"),
                provider=execution.get("provider"),
            )
            serialized = result.model_dump(mode="json")
            self._jobs[job_id] = self._jobs[job_id].model_copy(
                update={
                    "status": StatusEnum.COMPLETED,
                    "finished_at": datetime.now(UTC),
                    "result": serialized,
                }
            )
        except Exception:  # jobs must retain failures for the UI without leaking internals
            self._jobs[job_id] = self._jobs[job_id].model_copy(
                update={
                    "status": StatusEnum.FAILED,
                    "finished_at": datetime.now(UTC),
                    "error": "Evaluation failed. Check the server logs for details.",
                }
            )


__all__ = ["EvaluationJob", "EvaluationJobManager"]
