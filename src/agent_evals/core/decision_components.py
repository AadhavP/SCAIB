"""Names of the local-reward components, and where each one's value comes from.

Three places declare this vocabulary -- ``LocalRewardEvaluator._WEIGHTS``, the
``evaluator_metrics`` of ``evaluation/taxonomy.py``, and the per-category
``decision_evaluation.*.metrics`` lists in benchmark YAML -- and before this module
existed they had already drifted apart (``rare_cell_retention`` against
``rare_population_preservation``) without anything noticing. That is the failure
mode this module is here to make loud: a scored component whose value nothing can
supply does not raise, it silently contributes zero and drags the reward down.

The names are also the reason the components went unreceived for so long. They are
unprefixed (``ari``), while the metric registry keys everything by dotted id
(``clustering.ari``), so a lookup by component name against metric evidence never
matched and the fallback branch always fired.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Components computed from the observed before/after state rather than looked up
#: in metric evidence. They need no held-out reference, which is why they remain
#: answerable on a redacted dataset where every reference-derived component does
#: not.
OBSERVED_COMPONENTS = frozenset({"artifact_removal"})

#: Component name to the registry metrics that can answer it, in preference
#: order; the first one present in the evidence wins. Several candidates are
#: listed because which metrics are computable depends on what the agent
#: produced -- a run with an embedding can answer batch removal with ``batch_asw``
#: that a run without one cannot.
COMPONENT_METRIC_SOURCES: Mapping[str, tuple[str, ...]] = {
    "biological_retention": (
        "biological_conservation.cell_type_asw",
        "biological_conservation.ari",
        "clustering.ari",
    ),
    "rare_population_preservation": (
        "clustering.rare_recall",
        "cell_annotation.rare_recall",
    ),
    "batch_removal": (
        "batch_integration.iLISI",
        "biological_conservation.graph_connectivity",
        "batch_integration.batch_asw",
    ),
    "biology_preservation": (
        "biological_conservation.cell_type_asw",
        "biological_conservation.ari",
    ),
    "ari": ("clustering.ari", "biological_conservation.ari"),
    "rare_cell_recovery": ("clustering.rare_recall", "cell_annotation.rare_recall"),
    "stability": ("clustering.stability_ari",),
}

#: Keys the observed-state mapping must use. Kept minimal on purpose: these two
#: counts are the only quantities both the typed tier (which holds an ``AnnData``)
#: and the free tier (which holds a file it did not write) can report identically.
OBSERVED_CELL_COUNT = "n_obs"
OBSERVED_GENE_COUNT = "n_vars"


def resolve_metric_component(
    name: str,
    metrics: Mapping[str, float],
) -> tuple[float, str] | None:
    """Return the component's value and the metric id it came from, if answerable.

    An exact match on the component name is honoured first, so a caller that
    already speaks component names keeps working; otherwise the dotted candidates
    are tried in order. ``None`` means no source was available, which callers must
    treat as unmeasured rather than as zero.
    """
    if name in metrics:
        return float(metrics[name]), name
    for metric_id in COMPONENT_METRIC_SOURCES.get(name, ()):
        if metric_id in metrics:
            return float(metrics[metric_id]), metric_id
    return None


def removed_fraction(
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
    key: str,
) -> float | None:
    """Fraction of ``key`` that disappeared between the two observations.

    ``None`` unless both sides were observed and the earlier count is positive,
    because a step whose before-state nobody recorded has an unknown removal
    fraction rather than a zero one.
    """
    if before is None or after is None:
        return None
    start, end = before.get(key), after.get(key)
    if not isinstance(start, int | float) or not isinstance(end, int | float):
        return None
    if start <= 0:
        return None
    return max(0.0, min(1.0, (float(start) - float(end)) / float(start)))


__all__ = [
    "COMPONENT_METRIC_SOURCES",
    "OBSERVED_CELL_COUNT",
    "OBSERVED_COMPONENTS",
    "OBSERVED_GENE_COUNT",
    "removed_fraction",
    "resolve_metric_component",
]
