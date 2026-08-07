"""Optional scib-metrics adapter boundary.

The dependency is intentionally optional.  If installed, future benchmark
configurations can bind these IDs to the pinned upstream implementation
without changing the metric engine.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any


def available() -> bool:
    """Return whether the optional scib-metrics package is installed."""
    return find_spec("scib_metrics") is not None


def run(metric_name: str, embedding: Any, labels: Any, batches: Any) -> float:
    """Dispatch to scib-metrics when available."""
    if not available():
        raise RuntimeError("scib-metrics is not installed")
    import scib_metrics  # type: ignore[import-not-found]

    function = getattr(scib_metrics, metric_name)
    return float(function(X=embedding, labels=labels, batch_labels=batches))


__all__ = ["available", "run"]
