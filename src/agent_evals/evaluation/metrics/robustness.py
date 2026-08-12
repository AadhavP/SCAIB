"""Deterministic multi-seed robustness evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import adjusted_rand_score


class RobustnessReport(BaseModel):
    """Stability metrics computed across completed replicate outputs."""

    model_config = ConfigDict(extra="forbid")

    seeds: list[int]
    #: ``None`` when no two replicates could be compared. This was ``1.0``, which
    #: is a perfect stability score awarded for never having been tested: the live
    #: loop passes a single replicate, so every pairwise dimension below is empty
    #: and the old fallback handed out the robustness domain's full weight for
    #: free. Measuring it needs replicates, not a better default, so until they
    #: are run the honest report is that nothing was measured.
    seed_stability: float | None = Field(default=None, ge=0, le=1)
    clustering_pairwise_ari: float | None = Field(default=None, ge=0, le=1)
    embedding_neighbor_overlap: float | None = Field(default=None, ge=0, le=1)
    annotation_prediction_agreement: float | None = Field(default=None, ge=0, le=1)
    artifact_similarity: float | None = Field(default=None, ge=0, le=1)
    formula: str


class RobustnessEvaluator:
    """Compare only persisted replicate outputs; no rerun or judge is hidden."""

    def evaluate(self, replicates: Sequence[Mapping[str, Any]]) -> RobustnessReport:
        """Compute pairwise stability for cluster, neighbor, label, and artifacts."""
        seeds = [int(item.get("seed", index)) for index, item in enumerate(replicates)]
        aris = self._pairwise_cluster_ari(replicates)
        neighbors = self._pairwise_neighbor_overlap(replicates)
        labels = self._pairwise_label_agreement(replicates)
        artifacts = self._pairwise_artifact_similarity(replicates)
        values = [value for value in (aris, neighbors, labels, artifacts) if value is not None]
        stability = sum(values) / len(values) if values else None
        return RobustnessReport(
            seeds=seeds,
            seed_stability=stability,
            clustering_pairwise_ari=aris,
            embedding_neighbor_overlap=neighbors,
            annotation_prediction_agreement=labels,
            artifact_similarity=artifacts,
            formula=(
                "mean(available_pairwise_stability_metrics)"
                if values
                else "unmeasured: fewer than two comparable replicates"
            ),
        )

    @staticmethod
    def _pairwise_cluster_ari(replicates: Sequence[Mapping[str, Any]]) -> float | None:
        values = [item.get("cluster_labels") for item in replicates]
        pairs = [adjusted_rand_score(left, right) for left, right in combinations(values, 2) if left is not None and right is not None]
        return _bounded_mean(pairs)

    @staticmethod
    def _pairwise_neighbor_overlap(replicates: Sequence[Mapping[str, Any]]) -> float | None:
        values = [item.get("neighbors") for item in replicates]
        scores: list[float] = []
        for left, right in combinations(values, 2):
            if left is None or right is None or len(left) != len(right):
                continue
            scores.extend(
                len(set(a).intersection(b)) / max(1, len(set(a).union(b)))
                for a, b in zip(left, right, strict=True)
            )
        return _bounded_mean(scores)

    @staticmethod
    def _pairwise_label_agreement(replicates: Sequence[Mapping[str, Any]]) -> float | None:
        values = [item.get("predicted_labels") for item in replicates]
        scores = [
            float(np.mean(np.asarray(left) == np.asarray(right)))
            for left, right in combinations(values, 2)
            if left is not None and right is not None and len(left) == len(right)
        ]
        return _bounded_mean(scores)

    @staticmethod
    def _pairwise_artifact_similarity(replicates: Sequence[Mapping[str, Any]]) -> float | None:
        values = [item.get("artifact_checksums") for item in replicates]
        scores = [
            float(set(left) == set(right))
            for left, right in combinations(values, 2)
            if left is not None and right is not None
        ]
        return _bounded_mean(scores)


def _bounded_mean(values: list[float]) -> float | None:
    """Return a bounded mean while preserving unavailable dimensions."""
    if not values:
        return None
    return max(0.0, min(1.0, sum(values) / len(values)))


__all__ = ["RobustnessEvaluator", "RobustnessReport"]
