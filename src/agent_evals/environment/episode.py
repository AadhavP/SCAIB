"""Episode lifecycle and append-only trace management."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agent_evals.benchmarks.schema import BenchmarkSpecification
from agent_evals.core.exceptions import EnvironmentStateError
from agent_evals.environment.models import (
    ActionExecutionResult,
    ActionIntent,
    ActionRecord,
    ArtifactRecord,
    EpisodeEvent,
    EpisodeSnapshot,
    EpisodeState,
    EpisodeStatus,
    EventType,
    Observation,
    ResourceUsage,
    RewardRecord,
    utc_now,
)

TERMINAL_STATUSES = {
    EpisodeStatus.COMPLETED,
    EpisodeStatus.FAILED,
    EpisodeStatus.CANCELLED,
}


class Episode:
    """Own the mutable runtime state and immutable event history of one run.

    An episode captures the exact benchmark ID/version, specification digest,
    dataset selection, seed, action history, observations, artifacts, rewards,
    and timestamps needed for debugging and replay.  Callers receive deep
    snapshots rather than references to internal state.
    """

    def __init__(self, state: EpisodeState) -> None:
        self._state = state
        self._events: list[EpisodeEvent] = []

    @classmethod
    def from_specification(
        cls,
        specification: BenchmarkSpecification,
        *,
        task_id: str,
        seed: int,
        dataset_id: str | None = None,
        episode_id: str | None = None,
    ) -> Episode:
        """Create an episode after validating task and dataset selection."""
        task = next((task for task in specification.tasks if task.id == task_id), None)
        if task is None:
            raise EnvironmentStateError(f"unknown task '{task_id}'")
        if dataset_id is not None and dataset_id not in task.datasets:
            raise EnvironmentStateError(
                f"dataset '{dataset_id}' is not supported by task '{task_id}'"
            )
        digest_payload = json.dumps(
            specification.model_dump_serializable(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(digest_payload).hexdigest()
        state = EpisodeState(
            episode_id=episode_id or cls._new_episode_id(),
            benchmark_id=specification.metadata.id,
            benchmark_version=specification.metadata.version,
            task_id=task.id,
            seed=seed,
            dataset_id=dataset_id,
            specification_digest=digest,
        )
        episode = cls(state)
        episode._append_event(EventType.CREATED, {"dataset_id": dataset_id})
        return episode

    @staticmethod
    def _new_episode_id() -> str:
        """Generate a local episode identifier without external coordination."""
        from uuid import uuid4

        return str(uuid4())

    @property
    def episode_id(self) -> str:
        """Return the stable episode identifier."""
        return self._state.episode_id

    @property
    def status(self) -> EpisodeStatus:
        """Return the current lifecycle status."""
        return self._state.status

    def snapshot(self) -> EpisodeSnapshot:
        """Return a deep copy suitable for agents, ports, or persistence."""
        return EpisodeSnapshot(
            state=self._state.model_copy(deep=True),
            events=[event.model_copy(deep=True) for event in self._events],
        )

    def start(self) -> None:
        """Transition a newly created episode to running."""
        if self._state.status != EpisodeStatus.CREATED:
            raise EnvironmentStateError(
                f"episode '{self.episode_id}' cannot start from {self.status.value}"
            )
        self._state.status = EpisodeStatus.RUNNING
        self._state.started_at = utc_now()
        self._append_event(EventType.STARTED, {})

    def record_submission(self, intent: ActionIntent) -> None:
        """Record that a valid intent entered the executor pipeline."""
        self._ensure_running()
        self._append_event(
            EventType.ACTION_SUBMITTED,
            {"action_id": intent.action_id, "intent_id": intent.intent_id},
        )

    def record_rejection(self, intent: ActionIntent, errors: list[str]) -> None:
        """Record an invalid intent without advancing the episode step."""
        self._ensure_running()
        self._append_event(
            EventType.ACTION_REJECTED,
            {
                "action_id": intent.action_id,
                "intent_id": intent.intent_id,
                "errors": errors,
            },
        )

    def record_observations(self, observations: list[Observation]) -> None:
        """Replace visible values by ID and append an observation event."""
        self._ensure_running()
        for observation in observations:
            observation.step = self._state.current_step
            self._state.observations[observation.observation_id] = observation
        self._append_event(
            EventType.OBSERVATIONS_UPDATED,
            {"observation_ids": [item.observation_id for item in observations]},
        )

    def record_action(
        self,
        intent: ActionIntent,
        result: ActionExecutionResult,
    ) -> ActionRecord:
        """Commit one accepted action and its result as the next step."""
        self._ensure_running()
        self._state.current_step += 1
        record = ActionRecord(step=self._state.current_step, intent=intent, result=result)
        self._state.actions.append(record)
        previous = self._state.resource_usage
        current = result.resource_usage
        self._state.resource_usage = ResourceUsage(
            wall_time_seconds=previous.wall_time_seconds + current.wall_time_seconds,
            cpu_seconds=(
                (previous.cpu_seconds or 0.0) + (current.cpu_seconds or 0.0)
                if previous.cpu_seconds is not None or current.cpu_seconds is not None
                else None
            ),
            peak_memory_mb=max(
                previous.peak_memory_mb or 0.0,
                current.peak_memory_mb or 0.0,
            )
            or None,
            gpu_used=previous.gpu_used or current.gpu_used,
        )
        self._append_event(
            EventType.ACTION_COMPLETED,
            {
                "action_id": intent.action_id,
                "intent_id": intent.intent_id,
                "status": result.status.value,
            },
        )
        return record

    def record_outputs(
        self,
        observations: list[Observation],
        artifacts: list[ArtifactRecord],
    ) -> None:
        """Apply successful declared outputs to derived episode state."""
        self._ensure_running()
        for observation in observations:
            observation.step = self._state.current_step
            self._state.observations[observation.observation_id] = observation
        for artifact in artifacts:
            self._state.artifacts[artifact.artifact_id] = artifact
        if observations:
            self._append_event(
                EventType.OBSERVATIONS_UPDATED,
                {"observation_ids": [item.observation_id for item in observations]},
            )
        if artifacts:
            self._append_event(
                EventType.ARTIFACTS_UPDATED,
                {"artifact_ids": [item.artifact_id for item in artifacts]},
            )

    def record_reward(self, reward: RewardRecord) -> None:
        """Append a reward emitted by an external evaluator."""
        self._ensure_running()
        self._state.rewards.append(reward)
        self._append_event(EventType.REWARD_RECORDED, reward.model_dump(mode="json"))

    def terminate(self, status: EpisodeStatus, reason: str | None = None) -> None:
        """Close the episode with an explicit terminal status."""
        if status not in TERMINAL_STATUSES:
            raise EnvironmentStateError(f"'{status.value}' is not a terminal status")
        if self._state.status in TERMINAL_STATUSES:
            raise EnvironmentStateError(
                f"episode '{self.episode_id}' is already {self.status.value}"
            )
        self._state.status = status
        self._state.finished_at = utc_now()
        self._append_event(EventType.TERMINATED, {"reason": reason, "status": status.value})

    def _ensure_running(self) -> None:
        """Guard all state mutations that require an active episode."""
        if self._state.status != EpisodeStatus.RUNNING:
            raise EnvironmentStateError(
                f"episode '{self.episode_id}' is not running: {self.status.value}"
            )

    def _append_event(self, event_type: EventType, payload: dict[str, Any]) -> None:
        """Append an event at the current derived step."""
        self._events.append(
            EpisodeEvent(
                episode_id=self.episode_id,
                event_type=event_type,
                step=self._state.current_step,
                payload=payload,
            )
        )


__all__ = ["TERMINAL_STATUSES", "Episode"]
