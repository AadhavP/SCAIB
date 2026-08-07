"""Scipy adapters for ranking and correlation metrics."""

from __future__ import annotations

from typing import Any


def spearman(reference: Any, candidate: Any) -> float:
    """Compute Spearman correlation."""
    from scipy.stats import spearmanr  # type: ignore[import-untyped]

    value = spearmanr(reference, candidate).statistic
    return float(value) if value == value else 0.0


__all__ = ["spearman"]
