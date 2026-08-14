"""Per-benchmark scoring-profile resolution and the DE profile it unblocks.

Every assertion here guards a defect whose symptom is a *plausible number*. A
benchmark scored with the wrong profile does not raise, does not warn, and does
not look unusual in a report -- so the only way to see it is to compare the
resolved profile against the benchmark that asked for it, which is what this
module does.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agent_evals.benchmarks.io import load_benchmark
from agent_evals.core.exceptions import ConfigurationError, RegistryError
from agent_evals.environment.scientific_loop import ScientificLoop
from agent_evals.evaluation.profiles import (
    BUILTIN_PROFILES,
    BenchmarkMetricProfile,
    MetricGroupProfile,
    MetricProfileEntry,
    load_metric_profile,
    pbmc_annotation_profile,
    pbmc_de_profile,
    pbmc_integration_profile,
    profile_digest,
    profile_external_scores,
    profile_metric_ids,
    profiled_benchmark_ids,
    resolve_metric_profile,
    unregistered_profile_metrics,
)
from agent_evals.evaluation.scientific import ScientificMetricEngine
from agent_evals.evaluation.scoring import (
    UNRECORDED_METRIC_REASON,
    MetricScoreInput,
    WeightedGeometricAggregator,
    describe_unmeasured_domains,
)
from agent_evals.metrics import NormalizationEngine, metric_registry
from agent_evals.metrics.builtin.differential_expression import (
    REFERENCE_EFFECT_SIZES,
    REFERENCE_MARKERS,
)
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.results import MetricStatus

#: Benchmark id -> the ``benchmark`` field of the profile it must resolve to.
#: Written out longhand rather than derived from ``BUILTIN_PROFILES``, because a
#: test that reads the same table as the code under test agrees with any future
#: mis-wiring of it.
EXPECTED_PROFILE = {
    "pbmc-cell-annotation": "pbmc_annotation",
    "pbmc-cell-annotation-free": "pbmc_annotation",
    "pbmc-batch-correction": "pbmc_integration",
    "pbmc-differential-expression": "pbmc_de",
}

RANKED_DE_METRICS = (
    "differential_expression.precision_at_k",
    "differential_expression.recall_at_k",
    "differential_expression.f1_at_k",
    "differential_expression.auprc",
    "differential_expression.auroc",
    "differential_expression.rbo",
)
EFFECT_SIZE_DE_METRICS = (
    "differential_expression.effect_size_correlation",
    "differential_expression.direction_agreement",
)


def de_context(*, markers: bool = True, effect_sizes: bool = False) -> ScientificMetricContext:
    """Build a DE metric context with the evaluator evidence named."""
    genes = [f"GENE{index}" for index in range(10)]
    table = pd.DataFrame(
        {
            "gene": genes,
            "effect_size": [2.0, 1.8, 1.5, 1.2, 0.9, -0.4, -0.7, -1.0, -1.4, -1.9],
        }
    )
    metadata: dict[str, object] = {"k": 5}
    if markers:
        metadata[REFERENCE_MARKERS] = genes[:4]
    if effect_sizes:
        metadata[REFERENCE_EFFECT_SIZES] = dict(
            zip(genes, table["effect_size"], strict=True)
        )
    return ScientificMetricContext(
        candidate_artifacts={"de_table": table},
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("benchmark_id", sorted(EXPECTED_PROFILE))
def test_each_benchmark_resolves_to_the_profile_declared_for_it(benchmark_id: str) -> None:
    profile = resolve_metric_profile(benchmark_id)

    assert profile.benchmark == EXPECTED_PROFILE[benchmark_id]


@pytest.mark.parametrize(
    "benchmark_id", ["pbmc-batch-correction", "pbmc-differential-expression"]
)
def test_non_annotation_benchmarks_are_not_scored_with_annotation_metrics(
    benchmark_id: str,
) -> None:
    """The direct regression test for the silent annotation fallback.

    Asserted on the metric ids rather than only the profile name, because the
    defect's whole signature was a batch-correction run whose biology domain was
    ``clustering.ari`` and ``cell_annotation.rare_recall``.
    """
    metric_ids = profile_metric_ids(resolve_metric_profile(benchmark_id))

    assert metric_ids
    assert not [name for name in metric_ids if name.startswith("cell_annotation.")]
    assert metric_ids != profile_metric_ids(pbmc_annotation_profile())


def test_a_profile_digest_is_stable_and_changes_with_measurement_configuration() -> None:
    """A published score must identify the exact instrument that produced it."""
    profile = pbmc_annotation_profile()
    same = BenchmarkMetricProfile.model_validate(profile.model_dump(mode="json"))
    changed = profile.model_copy(
        update={
            "metric_groups": {
                **profile.metric_groups,
                "biology": profile.metric_groups["biology"].model_copy(
                    update={"weight": 0.5}
                ),
            }
        }
    )

    assert profile_digest(profile) == profile_digest(same)
    assert profile_digest(profile) != profile_digest(changed)
    assert len(profile_digest(profile)) == 64


def test_the_two_annotation_tiers_share_one_profile() -> None:
    """The typed and free tiers must stay comparable, which means same scoring."""
    typed = resolve_metric_profile("pbmc-cell-annotation")
    free = resolve_metric_profile("pbmc-cell-annotation-free")

    assert profile_metric_ids(typed) == profile_metric_ids(free)
    assert typed.metric_groups == free.metric_groups


def test_an_unknown_benchmark_is_an_error_rather_than_a_default() -> None:
    with pytest.raises(ConfigurationError) as error:
        resolve_metric_profile("pbmc-spatial-deconvolution")

    message = str(error.value)
    assert "pbmc-spatial-deconvolution" in message
    # The message has to name where to fix it and what already exists, or the
    # cheapest response to it is to re-add a fallback.
    assert "BUILTIN_PROFILES" in message
    for known in EXPECTED_PROFILE:
        assert known in message


def test_every_shipped_example_benchmark_has_a_declared_profile() -> None:
    """Tripwire: adding an example without a profile must fail here, loudly.

    Before per-benchmark resolution an unprofiled benchmark scored fine, with
    annotation metrics. Now it raises, so this test is what turns that raise into
    a build failure at authoring time instead of a crash during a paid run.
    """
    paths = sorted(Path("examples/benchmarks").glob("*.yaml"))
    assert paths, "no example benchmarks found; the tripwire would be vacuous"

    for path in paths:
        specification = load_benchmark(path)
        profile = resolve_metric_profile(specification.metadata.id)
        assert profile.benchmark == EXPECTED_PROFILE[specification.metadata.id]


def test_profiled_benchmark_ids_matches_the_resolution_table() -> None:
    assert list(profiled_benchmark_ids()) == sorted(BUILTIN_PROFILES)


# --------------------------------------------------------------------------- #
# The registered-metric guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "builder",
    [pbmc_annotation_profile, pbmc_de_profile, pbmc_integration_profile],
    ids=lambda builder: builder.__name__,
)
def test_builtin_profiles_name_only_registered_metrics(builder) -> None:
    """``pbmc_de`` shipped for months requiring a metric registered nowhere."""
    assert unregistered_profile_metrics(builder()) == []


def test_a_profile_naming_an_unregistered_metric_is_rejected(monkeypatch) -> None:
    def phantom_profile() -> BenchmarkMetricProfile:
        return BenchmarkMetricProfile(
            benchmark="phantom",
            metric_groups={
                "biology": MetricGroupProfile(
                    weight=1.0,
                    metrics={
                        "differential_expression.pseudobulk_recall": MetricProfileEntry(
                            weight=1.0
                        )
                    },
                )
            },
        )

    monkeypatch.setitem(BUILTIN_PROFILES, "phantom-benchmark", phantom_profile)

    assert unregistered_profile_metrics(phantom_profile()) == [
        "differential_expression.pseudobulk_recall"
    ]
    with pytest.raises(ConfigurationError, match="pseudobulk_recall"):
        resolve_metric_profile("phantom-benchmark")


def test_an_external_score_is_exempt_from_the_registered_metric_guard() -> None:
    """An external score is computed outside the registry by design.

    The annotation profile's ``robustness.seed_stability`` is not a registered
    metric and must not be reported as a missing one, or the guard would reject
    the profile the benchmark actually uses.
    """
    profile = pbmc_annotation_profile()

    assert profile_external_scores(profile) == {"robustness.seed_stability"}
    assert unregistered_profile_metrics(profile) == []
    with pytest.raises(RegistryError):
        metric_registry.get("robustness.seed_stability")


# --------------------------------------------------------------------------- #
# External scores over every group, not an index into one
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "builder",
    [pbmc_de_profile, pbmc_integration_profile],
    ids=lambda builder: builder.__name__,
)
def test_external_scores_do_not_require_a_robustness_group(builder) -> None:
    """The direct regression test for ``metric_groups["robustness"]``.

    That index sat one line away from the resolution fallback and raised
    ``KeyError`` for exactly the two benchmarks fixing the fallback made
    reachable, which is why both landed in one commit.
    """
    profile = builder()

    assert "robustness" not in profile.metric_groups
    assert profile_external_scores(profile) == set()


def test_external_scores_are_collected_from_every_group() -> None:
    profile = BenchmarkMetricProfile(
        benchmark="multi",
        metric_groups={
            "biology": MetricGroupProfile(
                weight=0.5,
                metrics={"clustering.ari": MetricProfileEntry(weight=1.0)},
                external_score="first.external",
            ),
            "robustness": MetricGroupProfile(
                weight=0.5,
                metrics={"clustering.ami": MetricProfileEntry(weight=1.0)},
                external_score="second.external",
            ),
        },
    )

    assert profile_external_scores(profile) == {"first.external", "second.external"}
    # An external score is not a registry metric, so it must not be offered to
    # the metric engine as one.
    assert profile_metric_ids(profile) == ["clustering.ari", "clustering.ami"]


# --------------------------------------------------------------------------- #
# The loop reads the same resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("benchmark_path", sorted(Path("examples/benchmarks").glob("*.yaml")))
def test_the_loop_resolves_each_benchmarks_own_profile(benchmark_path: Path) -> None:
    specification = load_benchmark(benchmark_path)

    profile = ScientificLoop._load_metric_profile(specification)

    assert profile.benchmark == EXPECTED_PROFILE[specification.metadata.id]


def test_progress_metrics_come_from_the_benchmarks_own_profile() -> None:
    """``S_t`` must be tracked on the metrics the final score is made of.

    If per-step metrics came from a different profile than the terminal ``O``,
    ``ΔS`` would measure a quantity the score never uses -- which is the same
    class of error as scoring the wrong profile, one layer in.
    """
    de_ids = ScientificLoop._progress_metric_ids(
        load_benchmark("examples/benchmarks/pbmc-differential-expression.yaml")
    )
    integration_ids = ScientificLoop._progress_metric_ids(
        load_benchmark("examples/benchmarks/pbmc-batch-correction.yaml")
    )
    annotation_ids = ScientificLoop._progress_metric_ids(
        load_benchmark("examples/benchmarks/pbmc-cell-annotation.yaml")
    )

    assert de_ids and all(name.startswith("differential_expression.") for name in de_ids)
    assert integration_ids
    assert any(name.startswith("batch_integration.") for name in integration_ids)
    assert any(
        name.startswith("biological_conservation.") for name in integration_ids
    )
    assert de_ids != annotation_ids
    # An external score has no registry computer, so asking for it every step
    # would raise inside the engine.
    assert "robustness.seed_stability" not in annotation_ids


def test_the_example_metric_yaml_still_mirrors_the_builtin_profile() -> None:
    """Anti-drift pin for the deleted preferential YAML load.

    ``_load_metric_profile`` used to prefer ``configs/metrics/<name>.yaml`` when
    the file existed, so the file and the built-in were two declarations of one
    scoring rule -- and a ``path.exists()`` guard meant deleting the file changed
    the rule silently. The file is kept as documentation, and this pins it to the
    built-in it documents.
    """
    path = Path("configs/metrics/pbmc_annotation.yaml")
    assert path.exists(), "the documented example profile is gone"

    assert load_metric_profile(path) == pbmc_annotation_profile()


# --------------------------------------------------------------------------- #
# The DE metric definitions the profile depends on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric_id", RANKED_DE_METRICS)
def test_ranked_de_metrics_declare_a_marker_requirement(metric_id: str) -> None:
    definition = metric_registry.get(metric_id)

    assert definition.applicability.structural_metadata == [REFERENCE_MARKERS]


@pytest.mark.parametrize("metric_id", EFFECT_SIZE_DE_METRICS)
def test_effect_size_metrics_declare_an_effect_size_requirement(metric_id: str) -> None:
    """These two read effect sizes and nothing else, so that is what they require.

    Declaring ``reference_markers`` here let them pass the structural gate on a
    marker-only benchmark, find no effect sizes, and land at ``failure_score`` --
    a manufactured zero charged to the agent for evidence the *evaluator* never
    supplied, which annihilates the DE domain's geometric mean.
    """
    definition = metric_registry.get(metric_id)

    assert definition.applicability.structural_metadata == [REFERENCE_EFFECT_SIZES]


@pytest.mark.parametrize("metric_id", RANKED_DE_METRICS + EFFECT_SIZE_DE_METRICS)
def test_the_de_table_stays_a_candidate_requirement(metric_id: str) -> None:
    """The DE table is the agent's own output, so its absence *should* score zero.

    The structural/candidate split is what decides between excluding a metric and
    charging a failure score, so both halves need pinning: moving ``de_table`` to
    the structural side would silently excuse an agent that produced nothing.
    """
    definition = metric_registry.get(metric_id)

    assert definition.applicability.required_artifacts == ["de_table"]
    assert definition.applicability.structural_artifacts == []


@pytest.mark.parametrize(
    ("correlation", "expected"), [(-1.0, 0.0), (-0.5, 0.25), (0.0, 0.5), (1.0, 1.0)]
)
def test_a_negative_effect_size_correlation_declines_rather_than_annihilating(
    correlation: float, expected: float
) -> None:
    """Pearson's r is native to [-1, 1] and must normalize through zero.

    Declared ``native_min=0`` with the ``bounded`` policy, r = 0 clamped to
    exactly 0.0 while r = 0.01 normalized to 0.01 -- and in a geometric mean 0.0
    annihilates the domain while 0.01 does not, so the score fell off a cliff at
    r = 0 instead of declining through it.
    """
    definition = metric_registry.get("differential_expression.effect_size_correlation")

    normalized = NormalizationEngine().normalize(correlation, definition)

    assert normalized == pytest.approx(expected)
    assert definition.native_min == -1


def test_direction_agreement_stays_a_bounded_fraction() -> None:
    definition = metric_registry.get("differential_expression.direction_agreement")

    assert definition.native_min == 0
    assert NormalizationEngine().normalize(0.0, definition) == pytest.approx(0.0)
    assert NormalizationEngine().normalize(1.0, definition) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# What the DE profile does with real evidence
# --------------------------------------------------------------------------- #


def de_domain(context: ScientificMetricContext):
    """Score the DE profile's only group through the real engine and aggregator."""
    profile = pbmc_de_profile()
    metric_ids = profile_metric_ids(profile)
    results, _, _, _ = ScientificMetricEngine().evaluate(metric_ids, context)
    inputs = [MetricScoreInput.from_metric_result(result) for result in results]
    domain = WeightedGeometricAggregator().aggregate(
        "biology", profile.metric_groups["biology"], inputs
    )
    return {result.metric_id: result for result in results}, domain


def test_markers_alone_score_the_ranked_metrics_and_exclude_the_effect_sizes() -> None:
    """The behavioural consequence of the requirement split, end to end."""
    results, domain = de_domain(de_context(markers=True, effect_sizes=False))

    assert results["differential_expression.precision_at_k"].status is MetricStatus.SCORED
    assert results["differential_expression.auroc"].status is MetricStatus.SCORED
    effect_size = results["differential_expression.effect_size_correlation"]
    assert effect_size.status is MetricStatus.INELIGIBLE
    assert REFERENCE_EFFECT_SIZES in effect_size.eligibility_reason

    # Excluded, not charged: a manufactured zero here would drag the whole domain
    # to zero through the geometric mean.
    assert domain.excluded_metrics == [
        "differential_expression.effect_size_correlation"
    ]
    assert domain.failed_metrics == []
    assert domain.value is not None and domain.value > 0


def test_effect_sizes_are_scored_when_the_evaluator_supplies_them() -> None:
    results, domain = de_domain(de_context(markers=True, effect_sizes=True))

    assert results["differential_expression.effect_size_correlation"].status is (
        MetricStatus.SCORED
    )
    assert domain.excluded_metrics == []
    assert set(domain.included_metrics) == {
        "differential_expression.precision_at_k",
        "differential_expression.auroc",
        "differential_expression.effect_size_correlation",
    }


def test_a_de_run_with_no_reference_reports_o_as_unmeasured_not_as_zero() -> None:
    """Records the honest state of the DE family until a reference is supplied.

    Nothing writes ``reference_markers`` evaluator-side yet, so this is what a
    real DE run currently produces. ``None`` is the correct answer -- the domain
    is dropped from the outcome with a reason attached, rather than reported as a
    zero the agent did not earn. A number here would mean the family had silently
    become scoreable against evidence nobody supplies.
    """
    results, domain = de_domain(de_context(markers=False, effect_sizes=False))

    assert {result.status for result in results.values()} == {MetricStatus.INELIGIBLE}
    assert domain.value is None
    assert domain.included_metrics == []
    assert sorted(domain.excluded_metrics) == sorted(results)


def test_the_unmeasured_de_domain_names_the_evidence_the_evaluator_never_supplied() -> None:
    """``None`` is only honest if something says *why*, and this is that something.

    The companion to the formula string persisted beside it: the formula names
    which domains were dropped, this names what was missing from each. Without it
    a DE run whose reference was never supplied publishes the same bare ``None``
    as one whose metrics legitimately did not apply -- the absent-versus-unobserved
    ambiguity this project removes everywhere else.

    Only the two *required* metrics are named. ``effect_size_correlation`` was
    excluded here too, but it is optional and did not void anything, so listing it
    would present three equal causes for one real one.
    """
    results, domain = de_domain(de_context(markers=False, effect_sizes=False))

    described = describe_unmeasured_domains(
        [domain],
        {result.metric_id: result.eligibility_reason for result in results.values()},
    )

    assert len(described) == 1
    explanation = described[0]
    assert explanation.startswith("domain 'biology' unmeasured:")
    assert "differential_expression.precision_at_k" in explanation
    assert "differential_expression.auroc" in explanation
    assert REFERENCE_MARKERS in explanation
    # Named because it blocked, not merely because it was excluded.
    assert "differential_expression.effect_size_correlation" not in explanation
    # And nothing here was invented: every cause came from a recorded reason.
    assert UNRECORDED_METRIC_REASON not in explanation


def test_the_de_profile_is_the_only_declaration_of_the_de_scoring_rule() -> None:
    """Why the annotation fallback was invisible on this benchmark.

    The DE YAML declares no ``metric_groups``, so the loop's benchmark-derived
    metric list is *empty* for it and the profile supplies every computed metric.
    Before this fix that combination was filled by a hardcoded five-metric
    annotation list plus the annotation profile, and neither had anything to do
    with differential expression.
    """
    specification = load_benchmark(
        "examples/benchmarks/pbmc-differential-expression.yaml"
    )

    assert specification.metric_groups == []
    assert ScientificLoop._progress_metric_ids(specification) == sorted(
        profile_metric_ids(pbmc_de_profile())
    )


def test_the_de_benchmarks_declared_metric_ids_are_prose_not_registry_ids() -> None:
    """Pins why the profile cannot be checked against the YAML mechanically.

    A reader comparing the two by hand would expect ``precision_at_k`` in the
    benchmark's ``metrics:`` block to be the registered
    ``differential_expression.precision_at_k``. It is not -- that block declares
    directions and ranges for reward shaping and documentation, and none of its
    ids resolve. This test exists so that stops being a surprise, and so that a
    future YAML which *does* use registry ids fails here and gets a real pin.
    """
    specification = load_benchmark(
        "examples/benchmarks/pbmc-differential-expression.yaml"
    )
    declared = [item.id for item in specification.metrics]
    assert declared, "the DE benchmark declares no metrics at all"

    for metric_id in declared:
        with pytest.raises(RegistryError):
            metric_registry.get(metric_id)
