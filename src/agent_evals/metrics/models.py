"""Typed, versioned metric definition models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetricRole(StrEnum):
    """Role of a metric in benchmark reporting."""

    GATE = "gate"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    DIAGNOSTIC = "diagnostic"
    COST = "cost"


class MetricDirection(StrEnum):
    """Optimization direction for a metric's native value."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    TARGET_VALUE = "target_value"


class MetricCategory(StrEnum):
    """Scientific or operational category of a metric."""

    #: The two earliest pipeline stages. Added so that a per-stage score has a
    #: category to live in; no metric is registered under either one yet, which
    #: ``test_stage_progress`` asserts explicitly rather than leaving implied. A
    #: stage with no scoreable metric contributes nothing to progress instead of
    #: contributing a zero.
    QC = "qc"
    NORMALIZATION = "normalization"
    ANNOTATION = "cell_annotation"
    CLUSTERING = "clustering"
    BATCH_INTEGRATION = "batch_integration"
    BIOLOGICAL_CONSERVATION = "biological_conservation"
    EMBEDDING = "embedding_fidelity"
    DIFFERENTIAL_EXPRESSION = "differential_expression"
    TRAJECTORY = "trajectory"
    RESOURCE = "resource"


class NormalizationSpec(BaseModel):
    """Explicit native-to-reporting scale transformation."""

    model_config = ConfigDict(extra="forbid")

    policy: str = "bounded"
    bad_anchor: float | None = None
    target_anchor: float | None = None
    target_value: float | None = None
    tolerance: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_policy(self) -> NormalizationSpec:
        """Validate parameters required by the selected policy."""
        if self.policy == "anchor" and (
            self.bad_anchor is None or self.target_anchor is None
        ):
            raise ValueError("anchor normalization requires bad_anchor and target_anchor")
        if self.policy == "target" and self.target_value is None:
            raise ValueError("target normalization requires target_value")
        return self


class MetricApplicability(BaseModel):
    """Requirements divided into structural and candidate-side evidence."""

    model_config = ConfigDict(extra="forbid")

    structural_artifacts: list[str] = Field(default_factory=list)
    structural_metadata: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    required_metadata: list[str] = Field(default_factory=list)
    required_observation_columns: list[str] = Field(default_factory=list)
    required_representations: list[str] = Field(default_factory=list)
    requires_reference_labels: bool = False
    requires_predictions: bool = False


class MetricDefinition(BaseModel):
    """Complete versioned contract for one scientific metric."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = "1.0"
    description: str = Field(min_length=1)
    category: MetricCategory
    role: MetricRole
    direction: MetricDirection
    native_min: float | None = None
    native_max: float | None = None
    applicability: MetricApplicability = Field(default_factory=MetricApplicability)
    required_artifacts: list[str] = Field(default_factory=list)
    required_metadata: list[str] = Field(default_factory=list)
    computation_backend: str = Field(min_length=1)
    normalization: NormalizationSpec = Field(default_factory=NormalizationSpec)
    aggregation_policy: str = "weighted_mean"
    failure_score: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> MetricDefinition:
        """Keep native bounds coherent and mirror required evidence fields."""
        if (
            self.native_min is not None
            and self.native_max is not None
            and self.native_min > self.native_max
        ):
            raise ValueError("native_min cannot exceed native_max")
        if not self.applicability.required_artifacts and self.required_artifacts:
            self.applicability.required_artifacts = list(self.required_artifacts)
        if not self.applicability.required_metadata and self.required_metadata:
            self.applicability.required_metadata = list(self.required_metadata)
        return self


__all__ = [
    "MetricApplicability",
    "MetricCategory",
    "MetricDefinition",
    "MetricDirection",
    "MetricRole",
    "NormalizationSpec",
]
