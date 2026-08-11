"""In-process asynchronous evaluation jobs for the API service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.core.types import StatusEnum
from agent_evals.environment.scientific_loop import ScientificLoop

JobEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

DEFAULT_SEED = 0
SUPPORTED_EXECUTION_KEYS = frozenset(
    {"seed", "max_cells", "max_steps", "model", "provider", "test_mode"}
)


def resolve_execution(request: dict[str, Any]) -> dict[str, Any]:
    """Merge `config_override` into execution settings; explicit fields win.

    Resolved once so the seed recorded on the job is the seed the run uses.
    """
    overrides = request.get("config_override", {})
    execution = {
        key: value
        for key, value in overrides.items()
        if key in SUPPORTED_EXECUTION_KEYS and key not in request
    }
    execution.update({key: request[key] for key in SUPPORTED_EXECUTION_KEYS if key in request})
    return execution


class EvaluationJob(BaseModel):
    """Public state for one submitted evaluation."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    benchmark_id: str
    agent_id: str
    # Reported so clients show the seed that actually ran rather than assuming one.
    seed: int = DEFAULT_SEED
    status: StatusEnum = StatusEnum.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: int = Field(default=0, ge=0, le=100)
    current_stage: str | None = None
    logs: list[str] = Field(default_factory=list)


class EvaluationJobManager:
    """Track API jobs and their replayable live events while the process runs."""

    MAX_RETAINED_JOBS = 100
    MAX_RETAINED_EVENTS = 250

    def __init__(self) -> None:
        self._jobs: dict[str, EvaluationJob] = {}
        self._requests: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._event_sequences: dict[str, int] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
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
        serialized = request.model_dump(exclude_none=True, exclude_defaults=True)
        execution = resolve_execution(serialized)
        self._jobs[job_id] = EvaluationJob(
            job_id=job_id,
            benchmark_id=request.benchmark_id,
            agent_id=request.agent_id,
            seed=int(execution.get("seed", DEFAULT_SEED)),
            current_stage="Queued",
            logs=["Run accepted by the evaluation queue."],
        )
        self._requests[job_id] = serialized
        self._events[job_id] = []
        self._event_sequences[job_id] = 0
        self._subscribers[job_id] = set()
        self._publish(job_id, "run_queued", "Run accepted by the evaluation queue.")
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
        self._events.pop(oldest.job_id, None)
        self._event_sequences.pop(oldest.job_id, None)
        self._subscribers.pop(oldest.job_id, None)

    def get(self, job_id: str) -> EvaluationJob:
        try:
            return self._jobs[job_id].model_copy(deep=True)
        except KeyError as error:
            raise KeyError(f"evaluation job '{job_id}' was not found") from error

    def list(self) -> list[EvaluationJob]:
        return [job.model_copy(deep=True) for job in self._jobs.values()]

    def _update(
        self,
        job_id: str,
        update: dict[str, Any],
        event_type: str,
        message: str,
        *,
        payload: dict[str, Any] | None = None,
        terminal: bool = False,
    ) -> None:
        """Atomically update public state and publish the corresponding event."""
        self._jobs[job_id] = self._jobs[job_id].model_copy(update=update)
        self._publish(job_id, event_type, message, payload=payload, terminal=terminal)

    def _publish(
        self,
        job_id: str,
        event_type: str,
        message: str,
        *,
        payload: dict[str, Any] | None = None,
        terminal: bool = False,
    ) -> None:
        """Publish one event to history and all connected subscribers."""
        job = self._jobs[job_id]
        history = self._events.setdefault(job_id, [])
        event_id = self._event_sequences.get(job_id, 0) + 1
        self._event_sequences[job_id] = event_id
        event = {
            "event_id": event_id,
            "type": event_type,
            "job_id": job_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "status": job.status.value,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "message": message,
            "payload": payload or {},
            "terminal": terminal,
        }
        history.append(event)
        if len(history) > self.MAX_RETAINED_EVENTS:
            del history[:-self.MAX_RETAINED_EVENTS]
        for queue in self._subscribers.get(job_id, set()):
            queue.put_nowait(event)

    async def events(self, job_id: str, *, after: int = 0) -> AsyncIterator[dict[str, Any]]:
        """Replay missed events, then wait for new events until the job is terminal."""
        self.get(job_id)  # raise immediately for an unknown job
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        subscribers = self._subscribers.setdefault(job_id, set())
        subscribers.add(queue)
        replay = [event for event in self._events.get(job_id, []) if event["event_id"] > after]
        try:
            for event in replay:
                yield event
                if event.get("terminal"):
                    return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield {"type": "heartbeat", "job_id": job_id, "timestamp": datetime.now(UTC).isoformat()}
                    continue
                yield event
                if event.get("terminal"):
                    return
        finally:
            subscribers.discard(queue)

    async def execute(self, job_id: str) -> None:
        async with self._semaphore:
            await self._execute_one(job_id)

    async def _execute_one(self, job_id: str) -> None:
        job = self._jobs[job_id]
        self._update(
            job_id,
            {
                "status": StatusEnum.RUNNING,
                "started_at": datetime.now(UTC),
                "progress": 8,
                "current_stage": "Loading benchmark and dataset",
                "logs": [*job.logs, "Worker started; loading the benchmark specification and dataset."],
            },
            "run_started",
            "Worker started; loading the benchmark specification and dataset.",
        )
        request = self._requests[job_id]
        try:
            execution = resolve_execution(request)

            async def on_scientific_event(event: dict[str, Any]) -> None:
                action_id = str(event.get("action_id", "workflow"))
                step = int(event.get("step", 1))
                event_kind = str(event.get("type", "action_finished"))
                if event_kind == "agent_planning":
                    progress = 22
                    message = "Asking the agent for an overall scientific plan."
                    stage = "Building scientific plan"
                elif event_kind == "agent_plan":
                    progress = 25
                    message = "Agent produced an initial scientific plan."
                    stage = "Scientific plan ready"
                elif event_kind == "agent_prompt":
                    progress = min(82, max(21, 16 + step * 10))
                    message = f"Sent the environment observation to the agent at step {step}."
                    stage = f"Preparing agent request (step {step})"
                elif event_kind == "agent_waiting":
                    progress = min(85, max(22, 18 + step * 10))
                    message = f"Waiting for the agent response at step {step}."
                    stage = f"Waiting for model response (step {step})"
                elif event_kind == "agent_response":
                    action_type = str(event.get("action_type", "action"))
                    progress = min(88, max(24, 19 + step * 10))
                    message = f"Agent proposed '{action_type}' at step {step}."
                    stage = f"Received agent action (step {step})"
                elif event_kind == "action_started":
                    progress = min(90, max(25, 20 + step * 10))
                    message = f"Started benchmark action '{action_id}'."
                    stage = f"Running {action_id}"
                else:
                    progress = min(95, max(30, 20 + step * 10))
                    status = str(event.get("status", "completed"))
                    message = f"Action '{action_id}' {status}."
                    stage = f"Recorded {action_id}"
                self._update(
                    job_id,
                    {
                        "progress": progress,
                        "current_stage": stage,
                        "logs": [*self._jobs[job_id].logs, message],
                    },
                    event_kind,
                    message,
                    payload=event,
                )

            workflow_message = (
                "GLM test mode enabled; using the backend LLM_* credentials."
                if execution.get("test_mode", False)
                else f"Agent '{request['agent_id']}' is executing observable benchmark actions."
            )
            self._update(
                job_id,
                {
                    "progress": 20,
                    "current_stage": "Running GLM test agent" if execution.get("test_mode", False) else "Running agent workflow",
                    "logs": [*self._jobs[job_id].logs, workflow_message],
                },
                "workflow_started",
                workflow_message,
            )
            result = await ScientificLoop().run(
                request["benchmark_id"],
                agent_type=request["agent_id"],
                output_dir="results",
                seed=int(execution.get("seed", DEFAULT_SEED)),
                max_cells=execution.get("max_cells"),
                max_steps=execution.get("max_steps"),
                model=execution.get("model"),
                provider=execution.get("provider"),
                test_mode=bool(execution.get("test_mode", False)),
                event_callback=on_scientific_event,
            )
            serialized = result.model_dump(mode="json")
            agent_run = serialized.get("agent_run") if isinstance(serialized, dict) else None
            termination = agent_run.get("termination_status") if isinstance(agent_run, dict) else None
            run_failed = termination not in (None, "completed")
            run_reason = agent_run.get("termination_reason") if isinstance(agent_run, dict) else None
            final_message = (
                f"Agent run {termination}: {run_reason}"
                if run_failed
                else "Workflow complete; metrics and artifacts were persisted."
            )
            self._update(
                job_id,
                {
                    "status": StatusEnum.FAILED if run_failed else StatusEnum.COMPLETED,
                    "finished_at": datetime.now(UTC),
                    "progress": 100,
                    "current_stage": "Failed" if run_failed else "Complete",
                    "result": serialized,
                    "error": final_message if run_failed else None,
                    "logs": [*self._jobs[job_id].logs, final_message],
                },
                "run_failed" if run_failed else "run_completed",
                final_message,
                terminal=True,
            )
        except Exception as error:  # jobs must retain failures for the UI without leaking tracebacks
            safe_reason = f"{type(error).__name__}: {error}".strip()[:500]
            final_message = f"Evaluation failed: {safe_reason}"
            self._update(
                job_id,
                {
                    "status": StatusEnum.FAILED,
                    "finished_at": datetime.now(UTC),
                    "progress": 100,
                    "current_stage": "Failed",
                    "error": final_message,
                    "logs": [*self._jobs[job_id].logs, f"Worker stopped with an error: {safe_reason}"],
                },
                "run_failed",
                final_message,
                terminal=True,
            )


__all__ = ["EvaluationJob", "EvaluationJobManager"]
