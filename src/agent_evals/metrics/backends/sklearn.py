"""Scikit-learn metric adapters with no benchmark-specific policy."""

from __future__ import annotations

from typing import Any


def f1_macro(reference: Any, predicted: Any) -> float:
    """Compute macro F1."""
    from sklearn.metrics import f1_score

    return float(f1_score(reference, predicted, average="macro", zero_division=0))


def matthews_correlation(reference: Any, predicted: Any) -> float:
    """Compute multiclass Matthews correlation coefficient."""
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(reference, predicted))


def balanced_accuracy(reference: Any, predicted: Any) -> float:
    """Compute class-balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(reference, predicted))


def normalized_mutual_information(reference: Any, predicted: Any) -> float:
    """Compute normalized mutual information."""
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(reference, predicted))


def adjusted_rand(reference: Any, predicted: Any) -> float:
    """Compute adjusted Rand index."""
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(reference, predicted))


def fowlkes_mallows(reference: Any, predicted: Any) -> float:
    """Compute Fowlkes-Mallows index."""
    from sklearn.metrics import fowlkes_mallows_score

    return float(fowlkes_mallows_score(reference, predicted))


def silhouette(embedding: Any, labels: Any) -> float:
    """Compute silhouette in a declared representation."""
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(embedding, labels))


def trustworthiness(reference_embedding: Any, candidate_embedding: Any, k: int) -> float:
    """Compute trustworthiness using scikit-learn's maintained implementation."""
    from sklearn.manifold import trustworthiness as sklearn_trustworthiness

    return float(sklearn_trustworthiness(reference_embedding, candidate_embedding, n_neighbors=k))


__all__ = [
    "adjusted_rand",
    "balanced_accuracy",
    "f1_macro",
    "fowlkes_mallows",
    "matthews_correlation",
    "normalized_mutual_information",
    "silhouette",
    "trustworthiness",
]
