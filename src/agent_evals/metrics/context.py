"""Inputs shared by scientific metric backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_evals.core.reference_columns import REFERENCE_LABEL_COLUMNS


@dataclass
class ScientificMetricContext:
    """Evaluator-side context; reference labels are never agent-visible."""

    adata: Any | None = None
    candidate_artifacts: dict[str, Any] = field(default_factory=dict)
    reference_artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    trajectory: Any | None = None
    #: Observation columns this run's agent actually wrote. A column the agent
    #: did not produce must never be read back as its prediction, or a dataset
    #: shipping its own ``louvain`` would be scored as though the agent had
    #: clustered. Empty means "unknown provenance", which is treated as
    #: "produced nothing" -- the safe direction.
    agent_produced_columns: frozenset[str] = frozenset()
    #: Why this run's candidate cannot be joined onto the evaluator's reference,
    #: or ``None`` when it can. A string rather than a flag because an
    #: unmeasurable outcome has to say what stopped it: this is the difference
    #: between "the agent produced nothing" and "the harness cannot see what the
    #: agent produced", and only the first belongs on the agent's score.
    reference_join_gap: str | None = None

    @property
    def has_reference_labels(self) -> bool:
        """Whether the evaluator can score this run against reference labels.

        Checks the evaluator-side reference channel *before* falling back to
        sniffing ``adata.obs``.  The fallback exists only for datasets that still
        carry their labels inline; once the reference has been partitioned out of
        the agent-visible object, the channel is the only truthful answer, and
        sniffing alone would declare every reference metric structurally
        ineligible and silently collapse the outcome score.

        ``reference_join_gap`` overrides both, because *holding* a reference is
        not the same as being able to use it. On a tier where the agent's output
        lands somewhere the evaluator cannot join back -- a workspace file rather
        than the object scored here -- the reference is present and useless, and
        presenting it anyway scores every reference metric against a candidate
        that was never available. That produces a real zero for a harness gap,
        which is the one outcome the whole "not measurable, no score" rule exists
        to prevent.
        """
        if self.reference_join_gap is not None:
            return False
        if self.reference_artifacts.get("labels") is not None:
            return True
        if self.adata is None:
            return False
        return any(key in self.adata.obs for key in REFERENCE_LABEL_COLUMNS)


__all__ = ["ScientificMetricContext"]
