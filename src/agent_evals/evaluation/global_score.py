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
    decision_quality: float | None,
    trajectory_quality: float,
) -> GlobalAgentScore | None:
    """Compute the global score, or nothing when a dimension is unmeasured.

    A missing dimension yields no score rather than a substituted one. Filling
    ``decision_quality`` with a neutral 1.0 made an agent that recorded no
    decisions score *higher* than one whose decisions were scored and found
    merely good, which inverts what the benchmark exists to measure.
    """
    if scientific_outcome is None or decision_quality is None:
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
