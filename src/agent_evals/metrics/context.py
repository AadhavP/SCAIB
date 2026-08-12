"""Inputs shared by scientific metric backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_evals.scientific.context import REFERENCE_LABEL_COLUMNS


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

    @property
    def has_reference_labels(self) -> bool:
        """Whether the evaluator holds reference labels for this task.

        Checks the evaluator-side reference channel *before* falling back to
        sniffing ``adata.obs``.  The fallback exists only for datasets that still
        carry their labels inline; once the reference has been partitioned out of
        the agent-visible object, the channel is the only truthful answer, and
        sniffing alone would declare every reference metric structurally
        ineligible and silently collapse the outcome score.
        """
        if self.reference_artifacts.get("labels") is not None:
            return True
        if self.adata is None:
            return False
        return any(key in self.adata.obs for key in REFERENCE_LABEL_COLUMNS)


__all__ = ["ScientificMetricContext"]
