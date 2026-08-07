"""Versioned metric computation results."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.metrics.models import MetricDirection, MetricRole


class MetricStatus(StrEnum):
    """Outcome state of one metric evaluation."""

    COMPUTED = "computed"
    FAILED = "failed"
    STRUCTURALLY_INELIGIBLE = "structurally_ineligible"


class MetricResult(BaseModel):
    """Raw and normalized values plus applicability provenance."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str
    version: str
    metric_name: str
    role: MetricRole
    direction: MetricDirection
    raw_value: Any | None = None
    normalized_value: float | None = Field(default=None, ge=0, le=1)
    eligible: bool
    status: MetricStatus
    eligibility_reason: str
    missing_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["MetricResult", "MetricStatus"]
