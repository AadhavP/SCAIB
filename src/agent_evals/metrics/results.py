"""Versioned metric computation results."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.metrics.models import MetricDirection, MetricRole


class MetricStatus(StrEnum):
    """Why one metric produced the number it produced, or produced none.

    ``FAILED`` used to cover three situations a reader needs to tell apart: the
    agent never produced the required input, the input was there but no number
    could be read out of it, and the evaluator itself raised. All three scored
    identically and read identically, so an evaluator bug was indistinguishable
    from an agent's miss in the archived record -- and it was the agent that
    carried the number.

    Splitting them changes what is **recorded**, deliberately not what is
    **scored**. The agent controls the artifacts a metric computer runs on, so
    excluding :attr:`EVALUATOR_ERROR` from aggregation would hand it a way to
    delete a metric it was about to fail, by feeding the computer something that
    raises. This is the one place where the "not measurable, so no score" rule is
    *not* applied: here it would pay. The finer status is evidence for whoever
    maintains the harness, and it is not spent on the score.
    """

    #: A number was computed and normalized.
    SCORED = "scored"
    #: The metric structurally does not apply to this benchmark or dataset.
    INELIGIBLE = "ineligible"
    #: A required candidate artifact or metadata key was absent.
    MISSING = "missing"
    #: The inputs were present, and no number could be read out of them.
    MALFORMED = "malformed"
    #: The evaluator raised. Recorded distinctly so a harness bug is auditable
    #: rather than filed under the agent's failures.
    EVALUATOR_ERROR = "evaluator_error"
    #: No implementation of this metric exists in this deployment. Unlike every
    #: status above it, nothing about the agent's run produced this -- a perfect
    #: submission gets the same result -- so it is dropped from scoring instead of
    #: charged as a zero. It is also how a result file states which metrics the
    #: SCAIB build that produced it could not compute, rather than publishing a
    #: 0.0 that reads as a measurement.
    UNIMPLEMENTED = "unimplemented"
    #: Failed with the cause not established. Retained so results persisted
    #: before the split still load; new code should name one of the three causes.
    FAILED = "failed"

    #: Pre-split spellings, kept as aliases of their successors so existing call
    #: sites and archived enum names keep resolving.
    COMPUTED = "scored"
    STRUCTURALLY_INELIGIBLE = "ineligible"

    @property
    def excluded_from_scoring(self) -> bool:
        """Whether aggregation drops this metric instead of scoring it zero."""
        return self in _EXCLUDED_FROM_SCORING

    @classmethod
    def _missing_(cls, value: object) -> MetricStatus | None:
        """Resolve a status string written before the split.

        Aliasing the *names* above is not enough: a persisted result carries the
        old *value*, and without this hook every archived ``report.json`` would
        stop loading the moment the canonical spellings changed.
        """
        if isinstance(value, str):
            return _PRE_SPLIT_STATUS_VALUES.get(value)
        return None


#: Old on-disk spellings mapped to their successors. Only values that no longer
#: exist need an entry; ``failed`` is still a live member.
_PRE_SPLIT_STATUS_VALUES: dict[str, MetricStatus] = {
    "computed": MetricStatus.SCORED,
    "structurally_ineligible": MetricStatus.INELIGIBLE,
}

#: The statuses whose metrics leave the aggregate entirely rather than entering it
#: at their failure score. Both mean the measurement was never the agent's to
#: make. Declared once because the aggregator and the loop that feeds it must
#: agree, and disagreement fails *silently*: a status missing from this set is
#: included at 0.0 and reads in the report as a metric the agent failed.
_EXCLUDED_FROM_SCORING: frozenset[MetricStatus] = frozenset(
    {MetricStatus.INELIGIBLE, MetricStatus.UNIMPLEMENTED}
)


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
