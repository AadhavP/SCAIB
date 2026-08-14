"""Durable SQLite storage for API evaluation jobs and event history.

The API deliberately keeps its public job model in :mod:`agent_evals.api.jobs`,
while this module owns the persistence boundary. SQLite is part of the Python
runtime, so the service gets restart-safe semantics without making PostgreSQL a
required development dependency. Deployments can put the database on durable
storage and migrate this narrow interface to a managed database later.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class JobStoreError(RuntimeError):
    """Raised when the durable job store cannot complete an operation."""


class JobQueueFull(JobStoreError):
    """Raised when durable admission reaches the configured active-job cap."""


class SQLiteJobStore:
    """Small transactional store for jobs, idempotency keys, and SSE events."""

    def __init__(
        self,
        path: Path | str,
        *,
        max_events: int = 250,
        claim_lease_seconds: int = 120,
    ) -> None:
        self.path = path
        self.max_events = max_events
        self.claim_lease_seconds = claim_lease_seconds
        self._lock = threading.RLock()
        database = ":memory:" if str(path) == ":memory:" else str(Path(path))
        if database != ":memory":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(
                database,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._configure()
        except sqlite3.Error as error:
            raise JobStoreError(f"unable to open job store {path}: {error}") from error

    def _configure(self) -> None:
        """Set durability and concurrency pragmas, then create the schema."""
        with self._lock:
            try:
                self._connection.executescript(
                    """
                    PRAGMA foreign_keys = ON;
                    PRAGMA busy_timeout = 30000;
                    PRAGMA journal_mode = WAL;
                    PRAGMA synchronous = FULL;

                    CREATE TABLE IF NOT EXISTS evaluation_jobs (
                        job_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        record_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        lease_until TEXT
                    );

                    CREATE TABLE IF NOT EXISTS evaluation_idempotency (
                        idempotency_key TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES evaluation_jobs(job_id)
                            ON DELETE CASCADE,
                        request_sha256 TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS evaluation_events (
                        job_id TEXT NOT NULL REFERENCES evaluation_jobs(job_id)
                            ON DELETE CASCADE,
                        event_id INTEGER NOT NULL,
                        event_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (job_id, event_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_evaluation_jobs_status
                        ON evaluation_jobs(status, updated_at);
                    CREATE INDEX IF NOT EXISTS idx_evaluation_events_job
                        ON evaluation_events(job_id, event_id);

                    CREATE TABLE IF NOT EXISTS evaluation_worker_leases (
                        lease_name TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        lease_until TEXT NOT NULL
                    );
                    """
                )
            except sqlite3.Error as error:
                raise JobStoreError(f"unable to initialize job store: {error}") from error

    def ping(self) -> None:
        """Verify that the store is writable, consistent, and fully initialized."""
        try:
            with self._transaction():
                row = self._connection.execute("PRAGMA quick_check").fetchone()
                if row is None or str(row[0]).lower() != "ok":
                    detail = "database integrity check did not return ok"
                    raise JobStoreError(detail)
        except JobStoreError:
            raise
        except sqlite3.Error as error:
            raise JobStoreError(f"job store health check failed: {error}") from error

    def close(self) -> None:
        """Close the process-local connection after workers have stopped."""
        with self._lock:
            self._connection.close()

    def create_job(
        self,
        record: Mapping[str, Any],
        *,
        idempotency_key: str | None,
        request_sha256: str,
        max_active_jobs: int | None = None,
        initial_event: Mapping[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """Persist a job, returning ``(job_id, created)`` atomically.

        The unique idempotency constraint is checked inside the same immediate
        transaction as job insertion. Two API workers cannot both admit the same
        key as separate scientific experiments.
        """
        job_id = str(record["job_id"])
        encoded = _encode(record)
        now = _now()
        with self._transaction():
            if idempotency_key is not None:
                existing = self._connection.execute(
                    "SELECT job_id, request_sha256 FROM evaluation_idempotency "
                    "WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    if str(existing["request_sha256"]) != request_sha256:
                        raise JobStoreError(
                            "idempotency key was already used for a different "
                            "evaluation request"
                        )
                    return str(existing["job_id"]), False
            if max_active_jobs is not None:
                active = self._connection.execute(
                    "SELECT COUNT(*) AS active FROM evaluation_jobs "
                    "WHERE status IN ('PENDING', 'RUNNING')"
                ).fetchone()
                if active is not None and int(active["active"]) >= max_active_jobs:
                    raise JobQueueFull("evaluation queue is full")
            self._connection.execute(
                "INSERT INTO evaluation_jobs "
                "(job_id, status, record_json, updated_at, lease_until) "
                "VALUES (?, ?, ?, ?, NULL)",
                (job_id, str(record["status"]), encoded, now),
            )
            if idempotency_key is not None:
                self._connection.execute(
                    "INSERT INTO evaluation_idempotency "
                    "(idempotency_key, job_id, request_sha256) VALUES (?, ?, ?)",
                    (idempotency_key, job_id, request_sha256),
                )
            if initial_event is not None:
                self._append_event_locked(job_id, initial_event)
        return job_id, True

    def get_idempotency(self, idempotency_key: str) -> tuple[str, str] | None:
        """Read a durable idempotency binding before queue admission."""
        with self._lock:
            row = self._connection.execute(
                "SELECT job_id, request_sha256 FROM evaluation_idempotency "
                "WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return str(row["job_id"]), str(row["request_sha256"])

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Read one durable job record."""
        with self._lock:
            row = self._connection.execute(
                "SELECT record_json FROM evaluation_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return None if row is None else _decode(str(row["record_json"]))

    def list_jobs(self) -> list[dict[str, Any]]:
        """Read all jobs in creation order."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT record_json FROM evaluation_jobs "
                "ORDER BY json_extract(record_json, '$.created_at') DESC"
            ).fetchall()
        return [_decode(str(row["record_json"])) for row in rows]

    def save_job(self, record: Mapping[str, Any]) -> None:
        """Replace one job record and clear any pre-start lease."""
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "UPDATE evaluation_jobs SET status = ?, record_json = ?, "
                    "updated_at = ?, lease_until = NULL WHERE job_id = ?",
                    (
                        str(record["status"]),
                        _encode(record),
                        _now(),
                        str(record["job_id"]),
                    ),
                )
                if cursor.rowcount != 1:
                    raise JobStoreError(
                        f"job '{record['job_id']}' disappeared while being persisted"
                    )
            except sqlite3.Error as error:
                raise JobStoreError(f"unable to persist job {record['job_id']}: {error}") from error

    def delete_job(self, job_id: str) -> None:
        """Delete a terminal job and its dependent evidence."""
        with self._lock:
            self._connection.execute(
                "DELETE FROM evaluation_jobs WHERE job_id = ?", (job_id,)
            )

    def claim_pending(self, job_id: str) -> bool:
        """Acquire a short lease for one queued job across worker processes."""
        now = datetime.now(UTC)
        lease_until = now + timedelta(seconds=self.claim_lease_seconds)
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE evaluation_jobs SET lease_until = ? "
                "WHERE job_id = ? AND status = 'PENDING' "
                "AND (lease_until IS NULL OR lease_until <= ?)",
                (lease_until.isoformat(), job_id, now.isoformat()),
            )
        return cursor.rowcount == 1

    def release_pending_lease(self, job_id: str) -> None:
        """Release a queued lease when this process is shutting down cleanly."""
        with self._lock:
            self._connection.execute(
                "UPDATE evaluation_jobs SET lease_until = NULL "
                "WHERE job_id = ? AND status = 'PENDING'",
                (job_id,),
            )

    def acquire_worker_lease(
        self,
        owner_id: str,
        *,
        lease_seconds: int,
        lease_name: str = "scientific-worker",
    ) -> bool:
        """Acquire the singleton execution lease for a worker process."""
        now = datetime.now(UTC)
        lease_until = now + timedelta(seconds=lease_seconds)
        with self._transaction():
            row = self._connection.execute(
                "SELECT owner_id, lease_until FROM evaluation_worker_leases "
                "WHERE lease_name = ?",
                (lease_name,),
            ).fetchone()
            if row is not None:
                current_owner = str(row["owner_id"])
                try:
                    current_expiry = datetime.fromisoformat(str(row["lease_until"]))
                except ValueError:
                    current_expiry = now
                if current_owner != owner_id and current_expiry > now:
                    return False
            self._connection.execute(
                "INSERT INTO evaluation_worker_leases "
                "(lease_name, owner_id, lease_until) VALUES (?, ?, ?) "
                "ON CONFLICT(lease_name) DO UPDATE SET owner_id = excluded.owner_id, "
                "lease_until = excluded.lease_until",
                (lease_name, owner_id, lease_until.isoformat()),
            )
        return True

    def renew_worker_lease(
        self,
        owner_id: str,
        *,
        lease_seconds: int,
        lease_name: str = "scientific-worker",
    ) -> bool:
        """Extend the singleton worker lease if this process still owns it."""
        lease_until = datetime.now(UTC) + timedelta(seconds=lease_seconds)
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE evaluation_worker_leases SET lease_until = ? "
                "WHERE lease_name = ? AND owner_id = ?",
                (lease_until.isoformat(), lease_name, owner_id),
            )
        return cursor.rowcount == 1

    def release_worker_lease(
        self,
        owner_id: str,
        *,
        lease_name: str = "scientific-worker",
    ) -> None:
        """Release a worker lease without disturbing a replacement owner."""
        with self._lock:
            self._connection.execute(
                "DELETE FROM evaluation_worker_leases "
                "WHERE lease_name = ? AND owner_id = ?",
                (lease_name, owner_id),
            )

    def worker_lease_active(self, *, lease_name: str = "scientific-worker") -> bool:
        """Return whether a worker lease is currently alive in the store."""
        with self._lock:
            row = self._connection.execute(
                "SELECT lease_until FROM evaluation_worker_leases "
                "WHERE lease_name = ?",
                (lease_name,),
            ).fetchone()
        if row is None:
            return False
        try:
            return datetime.fromisoformat(str(row["lease_until"])) > datetime.now(UTC)
        except (TypeError, ValueError):
            return False

    def recover_running(self) -> list[dict[str, Any]]:
        """Mark jobs interrupted by a process stop as failed, never replay them.

        Recovery runs in an immediate transaction so two API/worker processes
        starting at the same time cannot both publish a recovery for one job.
        """
        recovered: list[dict[str, Any]] = []
        with self._transaction():
            rows = self._connection.execute(
                "SELECT job_id, record_json FROM evaluation_jobs "
                "WHERE status = 'RUNNING'"
            ).fetchall()
            for row in rows:
                record = _decode(str(row["record_json"]))
                record.update(
                    {
                        "status": "FAILED",
                        "finished_at": _now(),
                        "current_stage": "Interrupted",
                        "error": (
                            "Evaluation worker stopped while this job was running; "
                            "the run was not replayed because the scientific endpoint "
                            "may have performed non-idempotent work. Submit a new run."
                        ),
                        "logs": [
                            *list(record.get("logs") or []),
                            "Worker recovery marked the in-flight job failed; no automatic replay was attempted.",
                        ],
                    }
                )
                cursor = self._connection.execute(
                    "UPDATE evaluation_jobs SET status = ?, record_json = ?, "
                    "updated_at = ?, lease_until = NULL "
                    "WHERE job_id = ? AND status = 'RUNNING'",
                    (
                        str(record["status"]),
                        _encode(record),
                        _now(),
                        str(row["job_id"]),
                    ),
                )
                if cursor.rowcount == 1:
                    self._append_event_locked(
                        str(row["job_id"]),
                        {
                            "type": "run_recovered",
                            "timestamp": _now(),
                            "status": "FAILED",
                            "progress": int(record.get("progress", 0)),
                            "current_stage": "Interrupted",
                            "message": (
                                "The previous worker stopped during execution; "
                                "the job was not replayed."
                            ),
                            "payload": {},
                            "terminal": True,
                        },
                    )
                    recovered.append(record)
        return recovered

    def save_job_event(
        self,
        record: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist one job state transition and its audit event atomically."""
        job_id = str(record["job_id"])
        with self._transaction():
            cursor = self._connection.execute(
                "UPDATE evaluation_jobs SET status = ?, record_json = ?, "
                "updated_at = ?, lease_until = NULL WHERE job_id = ?",
                (
                    str(record["status"]),
                    _encode(record),
                    _now(),
                    job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise JobStoreError(
                    f"job '{job_id}' disappeared while being persisted"
                )
            return self._append_event_locked(job_id, event)

    def append_event(self, job_id: str, event: Mapping[str, Any]) -> dict[str, Any]:
        """Assign a monotonic per-job event ID and persist the event atomically."""
        with self._transaction():
            return self._append_event_locked(job_id, event)

    def _append_event_locked(
        self,
        job_id: str,
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append an event while the caller owns a SQLite transaction."""
        row = self._connection.execute(
            "SELECT COALESCE(MAX(event_id), 0) + 1 AS next_id "
            "FROM evaluation_events WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        event_id = int(row["next_id"])
        persisted = {"event_id": event_id, **dict(event), "job_id": job_id}
        self._connection.execute(
            "INSERT INTO evaluation_events "
            "(job_id, event_id, event_json, created_at) VALUES (?, ?, ?, ?)",
            (job_id, event_id, _encode(persisted), _now()),
        )
        cutoff = event_id - self.max_events
        if cutoff > 0:
            self._connection.execute(
                "DELETE FROM evaluation_events WHERE job_id = ? AND event_id <= ?",
                (job_id, cutoff),
            )
        return persisted

    def events_after(self, job_id: str, after: int = 0) -> list[dict[str, Any]]:
        """Read retained events after a cursor for SSE replay or polling."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT event_json FROM evaluation_events "
                "WHERE job_id = ? AND event_id > ? ORDER BY event_id ASC",
                (job_id, after),
            ).fetchall()
        return [_decode(str(row["event_json"])) for row in rows]

    def first_event_id(self, job_id: str) -> int | None:
        """Return the oldest retained cursor, if this job has events."""
        with self._lock:
            row = self._connection.execute(
                "SELECT MIN(event_id) AS first_id FROM evaluation_events WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return None if row is None or row["first_id"] is None else int(row["first_id"])

    def _transaction(self) -> Any:
        """Begin an immediate transaction while serializing this connection."""
        return _SQLiteTransaction(self._connection, self._lock)


class _SQLiteTransaction:
    """Context manager for a short SQLite transaction."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._connection = connection
        self._lock = lock

    def __enter__(self) -> sqlite3.Connection:
        self._lock.acquire()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
        except Exception:
            self._lock.release()
            raise
        return self._connection

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if exc_type is None:
                self._connection.execute("COMMIT")
            else:
                self._connection.execute("ROLLBACK")
        finally:
            self._lock.release()


def _encode(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _decode(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise JobStoreError("persisted job/event record was not a JSON object")
    return decoded


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["JobQueueFull", "JobStoreError", "SQLiteJobStore"]
