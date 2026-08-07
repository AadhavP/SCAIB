"""Hierarchical global score for scientific agent evaluation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GlobalAgentScore(BaseModel):
    """Multiplicative score that couples outcome and decision quality."""

    model_config = ConfigDict(extra="forbid")

    scientific_outcome: float = Field(ge=0, le=1)
    decision_quality: float = Field(ge=0, le=1)
    trajectory_quality: float = Field(ge=0, le=1)
    value: float = Field(ge=0, le=1)
    formula: str


def compute_global_agent_score(
    scientific_outcome: float | None,
    decision_quality: float,
    trajectory_quality: float,
) -> GlobalAgentScore | None:
    """Compute the global score or preserve unavailable scientific outcomes."""
    if scientific_outcome is None:
        return None
    outcome = max(0.0, min(1.0, scientific_outcome))
    decision = max(0.0, min(1.0, decision_quality))
    trajectory = max(0.0, min(1.0, trajectory_quality))
    return GlobalAgentScore(
        scientific_outcome=outcome,
        decision_quality=decision,
        trajectory_quality=trajectory,
        value=outcome * decision * trajectory,
        formula="scientific_outcome * decision_quality * trajectory_quality",
    )


__all__ = ["GlobalAgentScore", "compute_global_agent_score"]
