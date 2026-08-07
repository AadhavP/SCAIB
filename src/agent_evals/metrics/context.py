"""Inputs shared by scientific metric backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScientificMetricContext:
    """Evaluator-side context; reference labels are never agent-visible."""

    adata: Any | None = None
    candidate_artifacts: dict[str, Any] = field(default_factory=dict)
    reference_artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    trajectory: Any | None = None


__all__ = ["ScientificMetricContext"]
