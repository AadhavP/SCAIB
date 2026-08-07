"""Provider-neutral protocol exchanged between agents and benchmarks."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentModelInfo(BaseModel):
    """Optional model/provider identity for reproducible comparisons."""

    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    name: str | None = None


class AgentManifest(BaseModel):
    """Public capability metadata; it never contains private reasoning."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    model: AgentModelInfo = Field(default_factory=AgentModelInfo)
    capabilities: list[str] = Field(default_factory=list)
    temperature: float | None = None
    context_window: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentContext(BaseModel):
    """Initialization context supplied by the benchmark runner."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    benchmark_id: str
    task_id: str
    workspace: str
    tools: list[dict[str, Any]] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSession(BaseModel):
    """Opaque, serializable session envelope owned by a runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    context: AgentContext
    state: dict[str, Any] = Field(default_factory=dict)


class AgentObservation(BaseModel):
    """Only environment-owned information visible to an agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    state: dict[str, Any] = Field(default_factory=dict)
    available_actions: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    """Minimal action envelope accepted from any agent implementation."""

    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning_metadata: dict[str, Any] = Field(default_factory=dict)


class FinalSubmission(BaseModel):
    """Terminal agent output containing only observable submission metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    output_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    explanation: str | None = None


__all__ = [
    "AgentAction",
    "AgentContext",
    "AgentManifest",
    "AgentModelInfo",
    "AgentObservation",
    "AgentSession",
    "FinalSubmission",
]
