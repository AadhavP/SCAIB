"""Which scoring profile a benchmark's scientific outcome is computed with.

The mapping lives in one table because getting it wrong produces a *plausible
number*, not an error. Before this module, resolution was an inline
``if specification.metadata.id == "pbmc-cell-annotation"`` with both branches
returning the annotation profile, so a batch-correction run was scored on
``clustering.ari`` and ``cell_annotation.rare_recall`` and a
differential-expression run on the same -- metrics that say nothing whatever
about either task, aggregated into an outcome a reader would take at face value.
Nothing raised, because scoring the wrong thing well is indistinguishable from
scoring the right thing well unless someone checks which profile was used.

Two rules follow from that, and both are enforced here rather than left to the
caller:

- **An unknown benchmark is an error, not a default.** Falling back to any
  profile is what the defect above *was*. A benchmark whose scoring rule nobody
  declared cannot have a scientific outcome, and saying so at resolve time makes
  it the benchmark author's error instead of a wrong number in a paper.
- **A profile may only name registered metrics.** ``pbmc_de`` required
  ``differential_expression.pseudobulk_recall``, which is registered nowhere in
  the repo, so the profile was unusable by construction -- and because the
  annotation fallback meant it was never resolved, it sat unreachable and
  therefore uncorrected. Checking here converts that from a latent crash inside
  the metric engine into a named configuration failure.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable

from agent_evals.core.exceptions import ConfigurationError, RegistryError
from agent_evals.evaluation.profiles.base import BenchmarkMetricProfile
from agent_evals.evaluation.profiles.pbmc_annotation import pbmc_annotation_profile
from agent_evals.evaluation.profiles.pbmc_de import pbmc_de_profile
from agent_evals.evaluation.profiles.pbmc_integration import pbmc_integration_profile
from agent_evals.metrics import MetricRegistry, metric_registry

#: Benchmark id -> the profile its scientific outcome is scored with.
#:
#: Keyed on the benchmark id from ``metadata.id``, which is what the YAML
#: declares and what the CLI resolves, rather than on the profile's own
#: ``benchmark`` field -- those deliberately differ in spelling
#: (``pbmc-batch-correction`` against ``pbmc_integration``) and unifying them
#: would mean renaming either a published benchmark id or a profile every test
#: imports by name.
#:
#: The free-execution annotation benchmark shares the annotation profile on
#: purpose: it is the same scientific task measured through a different agent
#: boundary, and scoring it differently would make the two integration tiers
#: incomparable, which is the one comparison that benchmark exists to support.
BUILTIN_PROFILES: dict[str, Callable[[], BenchmarkMetricProfile]] = {
    "pbmc-cell-annotation": pbmc_annotation_profile,
    "pbmc-cell-annotation-free": pbmc_annotation_profile,
    "pbmc-batch-correction": pbmc_integration_profile,
    "pbmc-differential-expression": pbmc_de_profile,
}


def profile_metric_ids(profile: BenchmarkMetricProfile) -> list[str]:
    """List every registry metric a profile scores, in declaration order.

    Excludes ``external_score`` entries, which name scores computed outside the
    metric registry -- see :func:`profile_external_scores`.
    """
    ids: list[str] = []
    for group in profile.metric_groups.values():
        ids.extend(name for name in group.metrics if name not in ids)
    return ids


def profile_digest(profile: BenchmarkMetricProfile) -> str:
    """Return the stable SHA-256 identity of a resolved scoring profile.

    The profile is part of the measurement instrument, not merely configuration.
    Hashing its canonical JSON representation lets a report prove which metric
    weights, optionality, and external scores produced its outcome even when the
    built-in profile is later revised.
    """
    payload = json.dumps(
        profile.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def profile_external_scores(profile: BenchmarkMetricProfile) -> set[str]:
    """Name the scores this profile expects from outside the metric registry.

    A set built over every group rather than an index into one. The caller used
    to reach straight for ``metric_groups["robustness"].external_score``, and
    neither the integration nor the differential-expression profile has a
    ``robustness`` group -- so that line raised ``KeyError`` for exactly the two
    benchmarks the resolution fix above makes reachable. The two changes are one
    unit of work for that reason.
    """
    return {
        group.external_score
        for group in profile.metric_groups.values()
        if group.external_score is not None
    }


def unregistered_profile_metrics(
    profile: BenchmarkMetricProfile,
    *,
    registry: MetricRegistry = metric_registry,
) -> list[str]:
    """Name the profile's metrics that no registered definition answers.

    ``external_score`` entries are exempt: they are scored by an evaluator that
    is not the metric registry, so their absence from it is by design.
    """
    missing: list[str] = []
    for metric_id in profile_metric_ids(profile):
        try:
            registry.get(metric_id)
        except RegistryError:
            missing.append(metric_id)
    return missing


def resolve_metric_profile(
    benchmark_id: str,
    *,
    registry: MetricRegistry = metric_registry,
) -> BenchmarkMetricProfile:
    """Return the scoring profile declared for ``benchmark_id``.

    Raises :class:`ConfigurationError` for an unregistered benchmark or a
    profile naming a metric that does not exist, because both are cheaper to
    discover here than after a paid run has produced a number nobody can trust.
    """
    builder = BUILTIN_PROFILES.get(benchmark_id)
    if builder is None:
        raise ConfigurationError(
            f"benchmark '{benchmark_id}' declares no metric profile; its "
            "scientific outcome cannot be scored. Register one in "
            "agent_evals.evaluation.profiles.resolution.BUILTIN_PROFILES. "
            f"Known benchmarks: {', '.join(sorted(BUILTIN_PROFILES))}."
        )
    profile = builder()
    missing = unregistered_profile_metrics(profile, registry=registry)
    if missing:
        raise ConfigurationError(
            f"metric profile '{profile.benchmark}' for benchmark "
            f"'{benchmark_id}' names unregistered metrics: "
            f"{', '.join(missing)}. A profile entry with no registered metric "
            "cannot be computed, so the domain it belongs to would report a "
            "score for something nobody measured."
        )
    return profile


def profiled_benchmark_ids() -> Iterable[str]:
    """Benchmark ids with a declared scoring profile, for tripwire tests."""
    return sorted(BUILTIN_PROFILES)


__all__ = [
    "BUILTIN_PROFILES",
    "profile_digest",
    "profile_external_scores",
    "profile_metric_ids",
    "profiled_benchmark_ids",
    "resolve_metric_profile",
    "unregistered_profile_metrics",
]
