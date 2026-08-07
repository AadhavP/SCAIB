"""Strict generic scientific metric contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MetricRole(StrEnum):
    """Role used by benchmark profiles when aggregating metrics."""

    GATE = "gate"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    DIAGNOSTIC = "diagnostic"


class MetricStatus(StrEnum):
    """Lifecycle state of one generic metric result."""

    COMPUTED = "computed"
    FAILED = "failed"
    STRUCTURALLY_INELIGIBLE = "structurally_ineligible"


class ArtifactBundle(BaseModel):
    """Candidate artifacts passed to a metric backend."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    values: dict[str, Any] = Field(default_factory=dict)


class ReferenceBundle(BaseModel):
    """Evaluator-owned references hidden from the agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    values: dict[str, Any] = Field(default_factory=dict)


class EvaluationContext(BaseModel):
    """Benchmark-independent context supplied to applicability and compute."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    metadata: dict[str, Any] = Field(default_factory=dict)
    available_artifacts: set[str] = Field(default_factory=set)
    available_metadata: set[str] = Field(default_factory=set)
    payload: Any | None = None


class MetricApplicability(BaseModel):
    """Frozen decision explaining whether a metric can be evaluated."""

    model_config = ConfigDict(extra="forbid")

    applicable: bool
    structurally_ineligible: bool = False
    reason: str
    missing_artifacts: list[str] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)


class ScoreAnchors(BaseModel):
    """Native-scale anchors used by a metric's normalization policy."""

    model_config = ConfigDict(extra="forbid")

    minimum: float | None = None
    maximum: float | None = None
    bad: float | None = None
    target: float | None = None


class RawMetricResult(BaseModel):
    """Raw backend output before normalization."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    value: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScientificMetricResult(BaseModel):
    """Stable generic result returned by the scientific metric engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    metric_name: str
    category: str
    raw_value: Any | None = None
    normalized_value: float | None = Field(default=None, ge=0, le=1)
    applicable: bool
    role: MetricRole
    status: MetricStatus
    implementation_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScientificMetric(ABC):
    """Adapter contract implemented by every reusable scientific metric."""

    name: str
    category: str
    direction: Literal["maximize", "minimize"]
    role: MetricRole
    implementation_version: str = "unknown"

    @abstractmethod
    def applicability(self, context: EvaluationContext) -> MetricApplicability:
        """Return structural and candidate applicability before computation."""

    @abstractmethod
    def compute(
        self,
        prediction: ArtifactBundle,
        reference: ReferenceBundle,
        context: EvaluationContext,
    ) -> RawMetricResult:
        """Compute a raw native-scale value using an open-source backend."""

    @abstractmethod
    def normalize(self, value: float, anchors: ScoreAnchors) -> float:
        """Convert a native value to the frozen [0, 1] score scale."""


__all__ = [
    "ArtifactBundle",
    "EvaluationContext",
    "MetricApplicability",
    "MetricRole",
    "MetricStatus",
    "RawMetricResult",
    "ReferenceBundle",
    "ScientificMetric",
    "ScientificMetricResult",
    "ScoreAnchors",
]
