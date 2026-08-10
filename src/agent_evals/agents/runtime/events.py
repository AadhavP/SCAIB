"""Provider-neutral trajectory events for universal agent runs."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.agents.runtime.protocol import (
    AgentAction,
    AgentObservation,
    FinalSubmission,
)


class AgentEventType(StrEnum):
    """Observable event categories; private chain-of-thought is not required."""

    OBSERVATION = "observation"
    PLAN = "plan"
    REASONING_SUMMARY = "reasoning_summary"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ACTION = "action"
    ENVIRONMENT_RESPONSE = "environment_response"
    FAILURE = "failure"
    RECOVERY = "recovery"
    FINAL_SUBMISSION = "final_submission"


class AgentEvent(BaseModel):
    """One append-only interaction event."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    sequence: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: AgentEventType
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_event_id: str | None = None


class AgentTrajectory(BaseModel):
    """Full provider-neutral trajectory with no requirement for hidden thoughts."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    trajectory_version: str = "2.0.0"
    events: list[AgentEvent] = Field(default_factory=list)
    actions: list[AgentAction] = Field(default_factory=list)
    observations: list[AgentObservation] = Field(default_factory=list)
    final_submission: FinalSubmission | None = None

    def record(
        self,
        event_type: AgentEventType,
        payload: dict[str, Any],
        *,
        parent_event_id: str | None = None,
    ) -> AgentEvent:
        """Append an event with a deterministic sequence number."""
        event = AgentEvent(
            sequence=len(self.events),
            event_type=event_type,
            payload=payload,
            parent_event_id=parent_event_id,
        )
        self.events.append(event)
        return event


__all__ = ["AgentEvent", "AgentEventType", "AgentTrajectory"]
