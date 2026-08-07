"""Reusable trajectory metric helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def method_exploration_score(
    methods_attempted: Sequence[str],
    alternatives: Mapping[str, Sequence[str]] | None = None,
    unnecessary_retries: int = 0,
) -> float:
    """Reward unique method coverage while penalizing repeated retries."""
    unique = set(methods_attempted)
    declared = {method for values in (alternatives or {}).values() for method in values}
    coverage = len(unique.intersection(declared)) / len(declared) if declared else (1.0 if unique else 0.0)
    retry_penalty = min(1.0, unnecessary_retries / max(1, len(methods_attempted)))
    return max(0.0, min(1.0, coverage * (1.0 - retry_penalty)))


def decision_regret(chosen_score: float, alternative_scores: Mapping[str, float] | None) -> float:
    """Report the observed gap to the best completed alternative."""
    if not alternative_scores:
        return 0.0
    return max(0.0, min(1.0, max(alternative_scores.values()) - chosen_score))


__all__ = ["decision_regret", "method_exploration_score"]
