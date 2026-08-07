"""Centralized metric normalization with explicit direction and anchors."""

from __future__ import annotations

from typing import Any

from agent_evals.metrics.models import MetricDefinition, MetricDirection


class NormalizationEngine:
    """Convert native metric values to [0, 1] without losing raw values."""

    def normalize(self, value: Any, definition: MetricDefinition) -> float | None:  # noqa: C901
        """Apply the definition's declared policy."""
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        spec = definition.normalization
        direction = definition.direction
        if spec.policy in {"bounded", "native"}:
            minimum = definition.native_min
            maximum = definition.native_max
            if minimum is None or maximum is None or maximum == minimum:
                return None
            score = (numeric - minimum) / (maximum - minimum)
            if direction == MetricDirection.LOWER_IS_BETTER:
                score = 1.0 - score
        elif spec.policy == "anchor":
            if spec.bad_anchor is None or spec.target_anchor is None:
                return None
            denominator = spec.target_anchor - spec.bad_anchor
            if denominator == 0:
                return 1.0 if numeric == spec.target_anchor else 0.0
            score = (numeric - spec.bad_anchor) / denominator
        elif spec.policy == "symmetric":
            score = (numeric + 1.0) / 2.0
            if direction == MetricDirection.LOWER_IS_BETTER:
                score = 1.0 - score
        elif spec.policy == "target":
            if spec.target_value is None:
                return None
            tolerance = spec.tolerance
            if tolerance is None or tolerance == 0:
                return 1.0 if numeric == spec.target_value else 0.0
            score = max(0.0, 1.0 - abs(numeric - spec.target_value) / tolerance)
        else:
            raise ValueError(f"unsupported normalization policy '{spec.policy}'")
        return max(0.0, min(1.0, score))


__all__ = ["NormalizationEngine"]
