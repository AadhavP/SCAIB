"""Structural and candidate-side metric applicability evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.metrics.models import MetricDefinition


class ApplicabilityContext(BaseModel):
    """Evidence visible to the evaluator, including hidden reference metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    structural_artifacts: set[str] = Field(default_factory=set)
    structural_metadata: set[str] = Field(default_factory=set)
    candidate_artifacts: set[str] = Field(default_factory=set)
    candidate_metadata: set[str] = Field(default_factory=set)
    observation_columns: set[str] = Field(default_factory=set)
    representations: set[str] = Field(default_factory=set)
    reference_labels_available: bool = False
    predictions_available: bool = False
    #: What to record when ``reference_labels_available`` is False. The default
    #: message blames the benchmark task for not providing a reference, which is
    #: only one of the two ways this happens -- the other is a tier that holds a
    #: reference it cannot join onto the candidate. Both exclude the metric, and
    #: a persisted exclusion that names the wrong cause is worse than a vague one.
    reference_gap_reason: str | None = None
    payload: Any | None = None


class ApplicabilityResult(BaseModel):
    """Persisted explanation of whether a metric can be evaluated."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str
    version: str
    eligible: bool
    structurally_ineligible: bool = False
    reason: str
    missing_candidate_artifacts: list[str] = Field(default_factory=list)
    missing_candidate_metadata: list[str] = Field(default_factory=list)


class MetricApplicabilityEngine:
    """Determine eligibility before candidate-specific metric computation."""

    def evaluate(
        self,
        definition: MetricDefinition,
        context: ApplicabilityContext,
    ) -> ApplicabilityResult:
        """Keep structural exclusion separate from candidate failure."""
        requirements = definition.applicability
        missing_structural_artifacts = sorted(
            set(requirements.structural_artifacts) - context.structural_artifacts
        )
        missing_structural_metadata = sorted(
            set(requirements.structural_metadata) - context.structural_metadata
        )
        if missing_structural_artifacts or missing_structural_metadata:
            missing = [
                *[f"artifact:{item}" for item in missing_structural_artifacts],
                *[f"metadata:{item}" for item in missing_structural_metadata],
            ]
            return ApplicabilityResult(
                metric_id=definition.metric_id,
                version=definition.version,
                eligible=False,
                structurally_ineligible=True,
                reason=f"structural requirements unavailable: {', '.join(missing)}",
            )
        missing_artifacts = sorted(
            set(requirements.required_artifacts) - context.candidate_artifacts
        )
        missing_metadata = sorted(
            set(requirements.required_metadata) - context.candidate_metadata
        )
        missing_columns = sorted(
            set(requirements.required_observation_columns) - context.observation_columns
        )
        missing_representations = sorted(
            set(requirements.required_representations) - context.representations
        )
        if requirements.requires_reference_labels and not context.reference_labels_available:
            return ApplicabilityResult(
                metric_id=definition.metric_id,
                version=definition.version,
                eligible=False,
                structurally_ineligible=True,
                reason=(
                    context.reference_gap_reason
                    or "reference labels are not provided by the benchmark task"
                ),
            )
        if requirements.requires_predictions and not context.predictions_available:
            missing_artifacts.append("prediction_artifact")
        missing = [
            *missing_columns,
            *missing_representations,
        ]
        if missing:
            missing_artifacts.extend(f"observation:{item}" for item in missing_columns)
        return ApplicabilityResult(
            metric_id=definition.metric_id,
            version=definition.version,
            eligible=True,
            reason=(
                "eligible; candidate evidence will receive the defined failure score"
                if missing_artifacts or missing_metadata
                else "eligible"
            ),
            missing_candidate_artifacts=sorted(set(missing_artifacts)),
            missing_candidate_metadata=missing_metadata,
        )


__all__ = ["ApplicabilityContext", "ApplicabilityResult", "MetricApplicabilityEngine"]
