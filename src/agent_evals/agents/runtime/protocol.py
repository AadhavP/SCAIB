"""Provider-neutral protocol exchanged between agents and benchmarks."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_evals.agents.decisions.parser import (
    ExtractionMode,
    ResponseExtractionEvidence,
)

PROTOCOL_VERSION = "1.0"


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
    #: Stable correlation id for all envelopes in one benchmark episode.
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    tools: list[dict[str, Any]] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    #: The public scientific brief assembled from the validated benchmark. It is
    #: separate from ``metadata`` so provider runtimes can rely on one stable
    #: location and arbitrary operator metadata cannot accidentally masquerade as
    #: part of the task contract.
    task_package: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSession(BaseModel):
    """Opaque, serializable session envelope owned by a runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    context: AgentContext
    state: dict[str, Any] = Field(default_factory=dict)


class AgentObservation(BaseModel):
    """Only environment-owned information visible to an agent.

    ``previous_decision`` and ``state_delta`` are explicit protocol fields rather
    than values an endpoint has to reverse-engineer from the full state history.
    They are computed by SCAIB from the prior environment result; an agent cannot
    use these fields to authoritatively mutate benchmark state.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    state: dict[str, Any] = Field(default_factory=dict)
    available_actions: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    previous_decision: dict[str, Any] | None = None
    state_delta: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentUsage(BaseModel):
    """Per-request usage reported by an agent boundary.

    Usage is optional because a black-box endpoint may not expose provider
    accounting. When present it is treated as an observation for cutoff and
    reporting, not as proof supplied by the evaluator; hard wall-time and step
    limits remain independently enforced by SCAIB.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    source: str | None = None

    @model_validator(mode="after")
    def fill_total_tokens(self) -> AgentUsage:
        """Derive a total when a provider reports the two components only."""
        if self.total_tokens is None and (
            self.input_tokens is not None or self.output_tokens is not None
        ):
            object.__setattr__(
                self,
                "total_tokens",
                (self.input_tokens or 0) + (self.output_tokens or 0),
            )
        return self


class AgentPlan(BaseModel):
    """Observable high-level plan for one scientific benchmark episode."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)
    steps: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    stopping_criteria: list[str] = Field(default_factory=list)
    adaptation_policy: str | None = None
    usage: AgentUsage | None = None


class AgentAction(BaseModel):
    """Minimal action envelope accepted from any agent implementation."""

    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning_metadata: dict[str, Any] = Field(default_factory=dict)
    #: Provider-reported usage for this request, when the boundary exposes it.
    usage: AgentUsage | None = None
    #: Observable claim about what the agent believes the previous action changed.
    #: The environment never trusts it; it compares this field with its own state
    #: delta and stores both in the decision record.
    state_claim: dict[str, Any] = Field(default_factory=dict)
    #: Optional terminal/replanning signal from the universal endpoint. It is
    #: evidence about the agent's control flow, not an instruction to the
    #: controller to stop or mutate benchmark state.
    next_step: dict[str, Any] = Field(default_factory=dict)
    #: Harness-generated provenance for the response that became this action.
    #: Agents cannot author the digest or extraction mode.
    extraction_evidence: ResponseExtractionEvidence | None = None
    #: Optional observable update to the working plan. Scientific work is
    #: iterative; asking for replanning in prose while offering no protocol field
    #: made the initial plan effectively immutable.
    plan_update: AgentPlan | None = None


class FinalSubmission(BaseModel):
    """Terminal agent output containing only observable submission metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    output_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    explanation: str | None = None
    usage: AgentUsage | None = None
    #: Harness-generated evidence for a black-box terminal response. It is a
    #: digest and extraction classification, never private chain-of-thought.
    extraction_evidence: ResponseExtractionEvidence | None = None


__all__ = [
    "PROTOCOL_VERSION",
    "AgentAction",
    "AgentContext",
    "AgentManifest",
    "AgentModelInfo",
    "AgentObservation",
    "AgentPlan",
    "AgentSession",
    "AgentUsage",
    "ExtractionMode",
    "FinalSubmission",
    "ResponseExtractionEvidence",
]
