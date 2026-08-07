"""Shared bounded normalization helpers for scientific metrics."""

from agent_evals.evaluation.metrics.base import ScoreAnchors


def normalize_bounded(value: float, anchors: ScoreAnchors, direction: str = "maximize") -> float:
    """Map a raw value to [0, 1] using explicit benchmark anchors."""
    if anchors.minimum is None or anchors.maximum is None:
        return max(0.0, min(1.0, value))
    span = anchors.maximum - anchors.minimum
    if span <= 0:
        return 1.0
    normalized = (value - anchors.minimum) / span
    if direction == "minimize":
        normalized = 1.0 - normalized
    return max(0.0, min(1.0, normalized))


__all__ = ["ScoreAnchors", "normalize_bounded"]
