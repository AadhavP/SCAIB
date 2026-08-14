"""Asynchronous evaluation jobs with durable restart-safe control-plane state."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.api.job_store import JobStoreError, SQLiteJobStore
from agent_evals.core.config import get_settings
from agent_evals.core.types import StatusEnum
from agent_evals.environment.scientific_loop import ScientificLoop

JobEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

DEFAULT_SEED = 0
SUPPORTED_EXECUTION_KEYS = frozenset(
    {
        "seed",
        "max_cells",
        "max_steps",
        "task_id",
        "dataset_id",
        "model",
        "provider",
        "agent_endpoint",
        "test_mode",
    }
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


class IdempotencyConflict(RuntimeError):
    """Raised when one idempotency key is reused for a different request."""


class EvaluationJob(BaseModel):
    """Public state for one submitted evaluation.

    The manager is intentionally in-process today, but the record still carries
    the resolved execution contract and a canonical request digest. A durable
    queue can persist these fields later without changing the external API, and
    operators can distinguish a repeated submission from a second experiment.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    benchmark_id: str
    agent_id: str
    task_id: str | None = None
    dataset_id: str | None = None
    #: Canonical digest of the public request and resolved execution settings.
    #: Secrets are rejected from the endpoint URL before this is computed.
    request_sha256: str = ""
    resolved_execution: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    termination_status: str | None = None
    termination_reason: str | None = None
    qualification_status: str | None = None
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
    """Coordinate jobs while SQLite provides durable cross-process state.

    The scientific run itself remains single-owner: a short database lease makes
    admission and execution claims atomic, while a process interrupted during a
    non-idempotent remote turn is marked failed rather than replayed silently.
    """

    MAX_RETAINED_JOBS = 100
    MAX_RETAINED_EVENTS = 250
    MAX_ACTIVE_JOBS = 8
    MAX_EVENT_PAYLOAD_BYTES = 256 * 1024
    MAX_RETAINED_LOGS = 200
    MAX_SUBSCRIBER_QUEUE = 64
    WORKER_LEASE_SECONDS = 30

    def __init__(
        self,
        *,
        db_path: Path | str | None = None,
        max_concurrency: int = 2,
        recover_interrupted: bool | None = None,
    ) -> None:
        settings = get_settings()
        resolved_path = db_path if db_path is not None else settings.storage.job_db_path
        self._store = SQLiteJobStore(
            resolved_path,
            max_events=self.MAX_RETAINED_EVENTS,
        )
        self._jobs: dict[str, EvaluationJob] = {}
        self._requests: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._event_sequences: dict[str, int] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}
        self._scheduled: set[str] = set()
        self._worker_tasks: set[asyncio.Task[None]] = set()
        self._supervisor_task: asyncio.Task[None] | None = None
        self._worker_id = str(uuid4())
        self._worker_lease_acquired = False
        self._lease_stop = threading.Event()
        self._lease_thread: threading.Thread | None = None
        self._worker_lease_lost = False
        self._started = False
        self._executes_jobs = False
        # API-only processes must not recover a job owned by the dedicated
        # worker. In production the setting is false for web workers and true
        # for the worker container; this decision is made before startup hooks
        # so a web restart cannot mark live science as failed.
        self._recover_interrupted = (
            settings.api.execute_jobs_in_process
            if recover_interrupted is None
            else recover_interrupted
        )
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        # Recovery is deliberately deferred until ``start`` has acquired the
        # singleton worker lease. A second worker may construct its manager while
        # the first is still alive; recovering here would falsely fail that
        # worker's in-flight jobs before the second process is rejected.
        self._load_persisted_state()

    def _load_persisted_state(self) -> None:
        """Hydrate memory caches without mutating another worker's jobs."""
        for record in self._store.list_jobs():
            job = EvaluationJob.model_validate(record)
            self._jobs[job.job_id] = job
            self._requests[job.job_id] = {
                "benchmark_id": job.benchmark_id,
                "agent_id": job.agent_id,
                **job.resolved_execution,
            }
            self._subscribers[job.job_id] = set()
        for job_id, events in self._load_events().items():
            self._events[job_id] = events
            self._event_sequences[job_id] = max(
                (int(event.get("event_id", 0)) for event in events),
                default=0,
            )
    def _recover_running_jobs(self) -> None:
        """Recover only jobs this process is authorized to execute."""
        for record in self._store.recover_running():
            job = EvaluationJob.model_validate(record)
            self._jobs[job.job_id] = job
            # Recovery state and its terminal audit event are committed together
            # by the store. Refresh the local history rather than appending a
            # second, non-atomic event here.
            events = self._store.events_after(job.job_id, after=0)
            self._events[job.job_id] = events
            self._event_sequences[job.job_id] = max(
                (int(event.get("event_id", 0)) for event in events),
                default=0,
            )

    def _load_events(self) -> dict[str, list[dict[str, Any]]]:
        """Load retained event history for known jobs."""
        return {
            job_id: self._store.events_after(job_id, after=0)
            for job_id in self._jobs
        }

    async def start(self, *, execute_jobs: bool = True) -> None:
        """Start the manager, optionally as an API-only enqueue process."""
        if self._started:
            return
        if execute_jobs:
            if not self._store.acquire_worker_lease(
                self._worker_id,
                lease_seconds=self.WORKER_LEASE_SECONDS,
            ):
                raise JobStoreError(
                    "another evaluation worker currently owns the execution lease"
                )
            self._worker_lease_acquired = True
            self._start_lease_heartbeat()
        self._started = True
        self._executes_jobs = execute_jobs
        if not execute_jobs:
            return
        try:
            # Only a lease owner may turn an interrupted RUNNING record into a
            # terminal failure. This must happen after admission, not in __init__.
            self._recover_running_jobs()
            self._recover_interrupted = True
            for job in self.list():
                if job.status is StatusEnum.PENDING:
                    self._schedule(job.job_id)
            self._supervisor_task = asyncio.create_task(self._resume_pending_jobs())
        except Exception:
            self._started = False
            self._executes_jobs = False
            self._stop_lease_heartbeat()
            self._store.release_worker_lease(self._worker_id)
            self._worker_lease_acquired = False
            raise

    def _start_lease_heartbeat(self) -> None:
        """Renew the worker lease outside the async loop as a safety net.

        Scientific libraries may occupy the event loop in synchronous sections
        for longer than the lease interval. A thread keeps the singleton lease
        alive during those sections so another worker cannot start a duplicate
        non-idempotent run.
        """
        self._lease_stop.clear()
        self._worker_lease_lost = False
        self._lease_thread = threading.Thread(
            target=self._lease_heartbeat,
            name=f"agent-evals-lease-{self._worker_id[:8]}",
            daemon=True,
        )
        self._lease_thread.start()

    def _lease_heartbeat(self) -> None:
        """Renew the SQLite lease until shutdown or a durable failure."""
        interval = max(1.0, self.WORKER_LEASE_SECONDS / 3)
        while not self._lease_stop.wait(interval):
            if not self._store.renew_worker_lease(
                self._worker_id,
                lease_seconds=self.WORKER_LEASE_SECONDS,
            ):
                self._worker_lease_lost = True
                return

    def _stop_lease_heartbeat(self) -> None:
        """Stop and join the lease-renewal thread before releasing the lease."""
        self._lease_stop.set()
        thread = self._lease_thread
        self._lease_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    async def _resume_pending_jobs(self) -> None:
        """Keep queued work recoverable after a crash or an expired lease."""
        while self._started:
            if self._worker_lease_lost or not self._store.renew_worker_lease(
                self._worker_id,
                lease_seconds=self.WORKER_LEASE_SECONDS,
            ):
                # Stop admitting new work if the database says another worker
                # owns the lease. Existing tasks are allowed to finish rather
                # than being duplicated by a second process.
                self._started = False
                self._executes_jobs = False
                return
            for job in self.list():
                if job.status is StatusEnum.PENDING and job.job_id not in self._scheduled:
                    self._schedule(job.job_id)
            await asyncio.sleep(5)

    async def shutdown(self) -> None:
        """Stop manager-owned workers without replaying in-flight science."""
        self._started = False
        self._executes_jobs = False
        if self._supervisor_task is not None:
            self._supervisor_task.cancel()
            await asyncio.gather(self._supervisor_task, return_exceptions=True)
            self._supervisor_task = None
        for job_id in list(self._scheduled):
            self._store.release_pending_lease(job_id)
        tasks = list(self._worker_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._worker_tasks.clear()
        if self._worker_lease_acquired:
            self._stop_lease_heartbeat()
            self._store.release_worker_lease(self._worker_id)
            self._worker_lease_acquired = False

    @property
    def started(self) -> bool:
        """Whether the manager lifecycle has started."""
        return self._started

    @property
    def executes_jobs(self) -> bool:
        """Whether this process owns scientific worker execution."""
        return self._executes_jobs

    def ready(self) -> None:
        """Raise when the durable control plane is not writable."""
        self._store.ping()

    def close(self) -> None:
        """Close the durable store for tests and explicit process teardown."""
        if self._worker_lease_acquired:
            self._stop_lease_heartbeat()
            self._store.release_worker_lease(self._worker_id)
            self._worker_lease_acquired = False
        self._store.close()

    def _schedule(self, job_id: str) -> None:
        """Create a tracked worker task for startup recovery."""
        task = asyncio.create_task(self.execute(job_id))
        self._worker_tasks.add(task)
        task.add_done_callback(self._worker_tasks.discard)

    def create(self, request: Any, *, idempotency_key: str | None = None) -> str:  # noqa: C901
        serialized = request.model_dump(exclude_none=True, exclude_defaults=True)
        execution = resolve_execution(serialized)
        # Materialize defaults before hashing so an omitted default and an
        # explicitly supplied equivalent value are the same experiment.
        execution.setdefault("seed", DEFAULT_SEED)
        execution.setdefault("test_mode", False)
        request_digest = _request_digest(request, execution)
        normalized_key: str | None = None
        if idempotency_key is not None:
            normalized_key = idempotency_key.strip()
            if not normalized_key or len(normalized_key) > 128:
                raise ValueError(
                    "idempotency key must contain 1-128 non-whitespace characters"
                )
            previous = self._idempotency.get(normalized_key)
            if previous is None:
                previous = self._store.get_idempotency(normalized_key)
            if previous is not None:
                previous_job, previous_digest = previous
                if previous_digest != request_digest:
                    raise IdempotencyConflict(
                        "idempotency key was already used for a different evaluation request"
                    )
                return previous_job
        active = sum(
            job.status in {StatusEnum.PENDING, StatusEnum.RUNNING}
            for job in self.list()
        )
        if active >= self.MAX_ACTIVE_JOBS:
            raise RuntimeError("evaluation queue is full")
        self._evict_old_jobs()
        job = EvaluationJob(
            job_id=str(uuid4()),
            benchmark_id=request.benchmark_id,
            agent_id=request.agent_id,
            task_id=execution.get("task_id"),
            dataset_id=execution.get("dataset_id"),
            request_sha256=request_digest,
            resolved_execution=execution,
            seed=int(execution.get("seed", DEFAULT_SEED)),
            current_stage="Queued",
            logs=["Run accepted by the evaluation queue."],
        )
        try:
            job_id, created = self._store.create_job(
                job.model_dump(mode="json"),
                idempotency_key=normalized_key,
                request_sha256=request_digest,
                max_active_jobs=self.MAX_ACTIVE_JOBS,
                initial_event={
                    "type": "run_queued",
                    "job_id": job.job_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "status": job.status.value,
                    "progress": job.progress,
                    "current_stage": job.current_stage,
                    "message": "Run accepted by the evaluation queue.",
                    "payload": {},
                    "terminal": False,
                },
            )
        except JobStoreError as error:
            if "idempotency key" in str(error):
                raise IdempotencyConflict(str(error)) from error
            raise
        if not created:
            existing = self._store.get_job(job_id)
            if existing is None:
                raise JobStoreError(
                    "idempotency record pointed at a missing evaluation job"
                )
            existing_job = EvaluationJob.model_validate(existing)
            self._cache_job(existing_job)
            self._requests[job_id] = {
                "benchmark_id": existing_job.benchmark_id,
                "agent_id": existing_job.agent_id,
                **existing_job.resolved_execution,
            }
            if normalized_key is not None:
                self._idempotency[normalized_key] = (
                    existing_job.job_id,
                    existing_job.request_sha256,
                )
            return job_id
        self._cache_job(job)
        self._requests[job.job_id] = serialized
        if normalized_key is not None:
            self._idempotency[normalized_key] = (job.job_id, request_digest)
        return job.job_id

    def _cache_job(self, job: EvaluationJob) -> None:
        """Install a durable record in the process-local fast path."""
        self._jobs[job.job_id] = job
        self._requests.setdefault(
            job.job_id,
            {
                "benchmark_id": job.benchmark_id,
                "agent_id": job.agent_id,
                **job.resolved_execution,
            },
        )
        self._events.setdefault(job.job_id, self._store.events_after(job.job_id))
        self._event_sequences[job.job_id] = max(
            (int(event.get("event_id", 0)) for event in self._events[job.job_id]),
            default=0,
        )
        self._subscribers.setdefault(job.job_id, set())

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
        self._store.delete_job(oldest.job_id)
        self._jobs.pop(oldest.job_id, None)
        self._requests.pop(oldest.job_id, None)
        self._events.pop(oldest.job_id, None)
        self._event_sequences.pop(oldest.job_id, None)
        self._subscribers.pop(oldest.job_id, None)
        self._scheduled.discard(oldest.job_id)
        for key, (job_id, _digest) in list(self._idempotency.items()):
            if job_id == oldest.job_id:
                self._idempotency.pop(key, None)

    def get(self, job_id: str) -> EvaluationJob:
        record = self._store.get_job(job_id)
        if record is None:
            raise KeyError(f"evaluation job '{job_id}' was not found")
        job = EvaluationJob.model_validate(record)
        self._cache_job(job)
        return job.model_copy(deep=True)

    def list(self) -> list[EvaluationJob]:
        jobs = [EvaluationJob.model_validate(record) for record in self._store.list_jobs()]
        for job in jobs:
            self._cache_job(job)
        return [job.model_copy(deep=True) for job in jobs]

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
        current = self.get(job_id)
        updated = current.model_copy(update=update)
        event = self._event_payload(
            job_id,
            updated,
            event_type,
            message,
            payload=payload,
            terminal=terminal,
        )
        persisted = self._store.save_job_event(
            updated.model_dump(mode="json"), event
        )
        self._jobs[job_id] = updated
        self._record_event(persisted)

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
        job = self.get(job_id)
        event = self._store.append_event(
            job_id,
            self._event_payload(
                job_id,
                job,
                event_type,
                message,
                payload=payload,
                terminal=terminal,
            ),
        )
        self._record_event(event)

    def _event_payload(
        self,
        job_id: str,
        job: EvaluationJob,
        event_type: str,
        message: str,
        *,
        payload: dict[str, Any] | None,
        terminal: bool,
    ) -> dict[str, Any]:
        """Build a bounded, public event from evaluator-owned job state."""
        return {
            "type": event_type,
            "job_id": job_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "status": job.status.value,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "message": message,
            "payload": _bounded_payload(
                payload or {}, self.MAX_EVENT_PAYLOAD_BYTES
            ),
            "terminal": terminal,
        }

    def _record_event(self, event: dict[str, Any]) -> None:
        """Update local subscribers after the durable event commit succeeds."""
        job_id = str(event["job_id"])
        history = self._events.setdefault(job_id, [])
        history.append(event)
        if len(history) > self.MAX_RETAINED_EVENTS:
            del history[:-self.MAX_RETAINED_EVENTS]
        self._event_sequences[job_id] = int(event["event_id"])
        for queue in self._subscribers.get(job_id, set()):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Durable polling/replay remains authoritative; replace a
                # stalled subscriber's queue with an explicit gap marker instead
                # of retaining an unbounded in-memory backlog.
                while True:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                queue.put_nowait(
                    {
                        "type": "event_gap",
                        "job_id": job_id,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "message": "subscriber notification backlog was compacted",
                        "payload": {"next_event_id": event["event_id"]},
                        "terminal": False,
                    }
                )

    async def events(self, job_id: str, *, after: int = 0) -> AsyncIterator[dict[str, Any]]:  # noqa: C901
        """Replay durable events, then poll and subscribe until terminal."""
        self.get(job_id)  # raise immediately for an unknown job
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self.MAX_SUBSCRIBER_QUEUE
        )
        subscribers = self._subscribers.setdefault(job_id, set())
        subscribers.add(queue)
        cursor = after
        idle_seconds = 0
        first_available = self._store.first_event_id(job_id)
        if first_available is not None and cursor < first_available - 1:
            yield {
                "type": "event_gap",
                "job_id": job_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "message": "older events were evicted from the retained event window",
                "payload": {
                    "requested_after": cursor,
                    "first_available": first_available,
                },
                "terminal": False,
            }
            cursor = first_available - 1
        try:
            while True:
                available = self._store.events_after(job_id, after=cursor)
                if available:
                    for event in available:
                        cursor = max(cursor, int(event["event_id"]))
                        yield event
                        if event.get("terminal"):
                            return
                    continue
                current = self.get(job_id)
                if current.status in {
                    StatusEnum.COMPLETED,
                    StatusEnum.FAILED,
                    StatusEnum.CANCELLED,
                }:
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1)
                except TimeoutError:
                    idle_seconds += 1
                    if idle_seconds >= 15:
                        idle_seconds = 0
                        yield {
                            "type": "heartbeat",
                            "job_id": job_id,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    continue
                idle_seconds = 0
                event_id = event.get("event_id")
                if isinstance(event_id, int) and event_id <= cursor:
                    continue
                if isinstance(event_id, int):
                    cursor = event_id
                yield event
                if event.get("terminal"):
                    return
        finally:
            subscribers.discard(queue)

    def claim_for_execution(self, job_id: str) -> bool:
        """Claim a queued job exactly once across API worker processes."""
        job = self.get(job_id)
        if job.status is not StatusEnum.PENDING or job_id in self._scheduled:
            return False
        if not self._store.claim_pending(job_id):
            return False
        self._scheduled.add(job_id)
        return True

    async def execute(self, job_id: str) -> None:
        if not self.claim_for_execution(job_id):
            return
        try:
            async with self._semaphore:
                await self._execute_one(job_id)
        finally:
            # A cancelled or unexpectedly terminated task must not be pinned in
            # this process's in-memory schedule forever. If it remains pending,
            # the durable lease/supervisor can make a deliberate retry decision.
            self._scheduled.discard(job_id)

    async def _execute_one(self, job_id: str) -> None:
        job = self._jobs[job_id]
        self._update(
            job_id,
            {
                "status": StatusEnum.RUNNING,
                "started_at": datetime.now(UTC),
                "progress": 8,
                "current_stage": "Loading benchmark and dataset",
                "logs": _append_log(
                    job.logs,
                    "Worker started; loading the benchmark specification and dataset.",
                    self.MAX_RETAINED_LOGS,
                ),
            },
            "run_started",
            "Worker started; loading the benchmark specification and dataset.",
        )
        request = self._requests[job_id]
        try:
            # Use the immutable settings captured at admission. Re-resolving the
            # mutable request here made the worker's actual configuration a second
            # interpretation of the public request, rather than the exact contract
            # whose digest was returned to the client.
            execution = dict(self._jobs[job_id].resolved_execution)

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
                        "logs": _append_log(
                            self._jobs[job_id].logs,
                            message,
                            self.MAX_RETAINED_LOGS,
                        ),
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
                    "logs": _append_log(
                        self._jobs[job_id].logs,
                        workflow_message,
                        self.MAX_RETAINED_LOGS,
                    ),
                },
                "workflow_started",
                workflow_message,
            )
            settings = get_settings()
            result = await ScientificLoop(
                cache_dir=Path(settings.storage.cache_dir) / "datasets"
            ).run(
                request["benchmark_id"],
                agent_type=request["agent_id"],
                output_dir=Path(settings.storage.reports_dir),
                seed=int(execution.get("seed", DEFAULT_SEED)),
                max_cells=execution.get("max_cells"),
                max_steps=execution.get("max_steps"),
                task_id=execution.get("task_id"),
                dataset_id=execution.get("dataset_id"),
                model=execution.get("model"),
                provider=execution.get("provider"),
                agent_endpoint=execution.get("agent_endpoint"),
                test_mode=bool(execution.get("test_mode", False)),
                event_callback=on_scientific_event,
            )
            serialized = result.model_dump(mode="json")
            agent_run = serialized.get("agent_run") if isinstance(serialized, dict) else None
            termination = agent_run.get("termination_status") if isinstance(agent_run, dict) else None
            qualification = serialized.get("qualification") if isinstance(serialized, dict) else None
            qualification_status = (
                qualification.get("status")
                if isinstance(qualification, dict)
                else None
            )
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
                    "run_id": result.run_id,
                    "termination_status": termination,
                    "termination_reason": run_reason,
                    "qualification_status": qualification_status,
                    "error": final_message if run_failed else None,
                    "logs": _append_log(
                        self._jobs[job_id].logs,
                        final_message,
                        self.MAX_RETAINED_LOGS,
                    ),
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
                    "logs": _append_log(
                        self._jobs[job_id].logs,
                        f"Worker stopped with an error: {safe_reason}",
                        self.MAX_RETAINED_LOGS,
                    ),
                },
                "run_failed",
                final_message,
                terminal=True,
            )


def _append_log(logs: list[str], message: str, maximum: int) -> list[str]:
    """Retain a bounded operational log while keeping the newest message."""
    return [*logs[-max(0, maximum - 1) :], message]


def _bounded_payload(payload: dict[str, Any], maximum_bytes: int) -> dict[str, Any]:
    """Keep event history bounded without losing an integrity signal."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    size = len(encoded.encode("utf-8"))
    if size <= maximum_bytes:
        return payload
    return {
        "truncated": True,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "bytes": size,
        "limit_bytes": maximum_bytes,
    }


def _request_digest(request: Any, execution: dict[str, Any]) -> str:
    """Hash the canonical experiment contract, not request spelling.

    The worker is driven by ``resolved_execution``. Hashing the raw Pydantic
    payload as well would make semantically identical submissions differ when a
    caller moved a value between an explicit field and ``config_override`` or
    omitted a default such as ``seed=0``. Idempotency is about the experiment that
    will run, so the digest intentionally excludes those transport spellings.
    """
    payload = {
        "benchmark_id": request.benchmark_id,
        "agent_id": request.agent_id,
        "resolved_execution": execution,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["EvaluationJob", "EvaluationJobManager", "IdempotencyConflict"]
