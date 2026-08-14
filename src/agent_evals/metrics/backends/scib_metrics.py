"""Lazy adapters for the optional ``scib-metrics`` implementation.

SCAIB keeps this dependency optional so the core protocol and typed benchmark
can run without the science stack. When installed, integration profiles call the
upstream Python implementation rather than a locally reimplemented approximation.
Every wrapper returns the upstream package/version and parameters as evidence so
an archived score identifies its measurement instrument.
"""

from __future__ import annotations

from importlib import import_module, metadata
from importlib.util import find_spec
from typing import Any


def available() -> bool:
    """Return whether the optional package and its public module are installed."""
    return find_spec("scib_metrics") is not None


def _module() -> Any:
    """Import the optional backend only after availability was established."""
    return import_module("scib_metrics")


def version() -> str | None:
    """Return the installed scib-metrics version, if it can be resolved."""
    try:
        return metadata.version("scib-metrics")
    except metadata.PackageNotFoundError:
        return None


def neighbors(
    embedding: Any,
    *,
    n_neighbors: int,
    random_state: int = 0,
    n_jobs: int = 1,
) -> Any:
    """Build the upstream ``NeighborsResults`` required by scib-metrics metrics."""
    if not available():
        raise RuntimeError("scib-metrics is not installed")
    import numpy as np
    scib_metrics = _module()

    factory = getattr(scib_metrics.nearest_neighbors, "pynndescent", None)
    if not callable(factory):
        raise RuntimeError("installed scib-metrics has no pynndescent neighbor backend")
    return factory(
        np.asarray(embedding),
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def ilisi_knn(
    embedding: Any,
    batches: Any,
    *,
    n_neighbors: int,
    perplexity: float | None = None,
    random_state: int = 0,
    n_jobs: int = 1,
) -> tuple[float, dict[str, Any]]:
    """Compute scaled iLISI using the upstream ``scib_metrics.ilisi_knn``."""
    scib_metrics = _module()

    graph = neighbors(
        embedding,
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    value = scib_metrics.ilisi_knn(
        X=graph,
        batches=batches,
        perplexity=perplexity,
        scale=True,
    )
    return float(value), {
        "backend": "scib-metrics",
        "package_version": version(),
        "function": "scib_metrics.ilisi_knn",
        "n_neighbors": n_neighbors,
        "perplexity": perplexity,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "scale": True,
    }


def kbet_per_label(
    embedding: Any,
    batches: Any,
    labels: Any,
    *,
    n_neighbors: int,
    alpha: float = 0.05,
    diffusion_n_comps: int = 100,
    random_state: int = 0,
    n_jobs: int = 1,
) -> tuple[float, dict[str, Any]]:
    """Compute kBET acceptance using the upstream per-label implementation."""
    scib_metrics = _module()

    graph = neighbors(
        embedding,
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    value = scib_metrics.kbet_per_label(
        X=graph,
        batches=batches,
        labels=labels,
        alpha=alpha,
        diffusion_n_comps=diffusion_n_comps,
        return_df=False,
    )
    return float(value), {
        "backend": "scib-metrics",
        "package_version": version(),
        "function": "scib_metrics.kbet_per_label",
        "n_neighbors": n_neighbors,
        "alpha": alpha,
        "diffusion_n_comps": diffusion_n_comps,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "return_df": False,
    }


def bras(
    embedding: Any,
    labels: Any,
    batches: Any,
    *,
    chunk_size: int = 256,
    metric: str = "cosine",
    between_cluster_distances: str = "mean_other",
) -> tuple[float, dict[str, Any]]:
    """Compute BRAS with the upstream scib-metrics implementation."""
    scib_metrics = _module()

    value = scib_metrics.bras(
        X=embedding,
        labels=labels,
        batch=batches,
        chunk_size=chunk_size,
        metric=metric,
        between_cluster_distances=between_cluster_distances,
    )
    return float(value), {
        "backend": "scib-metrics",
        "package_version": version(),
        "function": "scib_metrics.bras",
        "chunk_size": chunk_size,
        "metric": metric,
        "between_cluster_distances": between_cluster_distances,
    }


def graph_connectivity(
    embedding: Any,
    labels: Any,
    *,
    n_neighbors: int,
    random_state: int = 0,
    n_jobs: int = 1,
) -> tuple[float, dict[str, Any]]:
    """Compute graph connectivity using the upstream graph implementation."""
    scib_metrics = _module()

    graph = neighbors(
        embedding,
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    value = scib_metrics.graph_connectivity(X=graph, labels=labels)
    return float(value), {
        "backend": "scib-metrics",
        "package_version": version(),
        "function": "scib_metrics.graph_connectivity",
        "n_neighbors": n_neighbors,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }


def run(metric_name: str, embedding: Any, labels: Any, batches: Any) -> float:
    """Compatibility dispatcher for callers using the original adapter seam."""
    if metric_name == "ilisi_knn":
        return ilisi_knn(embedding, batches, n_neighbors=30)[0]
    if metric_name == "kbet_per_label":
        return kbet_per_label(embedding, batches, labels, n_neighbors=30)[0]
    if metric_name == "bras":
        return bras(embedding, labels, batches)[0]
    if metric_name == "graph_connectivity":
        return graph_connectivity(embedding, labels, n_neighbors=30)[0]
    raise ValueError(f"unsupported scib-metrics function '{metric_name}'")


__all__ = [
    "available",
    "bras",
    "graph_connectivity",
    "ilisi_knn",
    "kbet_per_label",
    "neighbors",
    "run",
    "version",
]
