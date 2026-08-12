"""Tests for per-stage scientific state and the progress signal.

The stage-inference tests are written against the output keys and artifact ids the
typed operations *actually* produce, captured from a real run rather than invented
here.  That is deliberate: a rename inside an operation would otherwise make its
stage silently uninferrable, and an uninferred stage contributes nothing to
progress without anything failing.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_evals.agents import (
    AgentConfiguration,
    AgentHarness,
    MockActionExecutor,
    MockAgentAdapter,
    MockObservationBuilder,
)
from agent_evals.agents.trajectory import DecisionCategory, ScientificDecision
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.core.decision_components import COMPONENT_METRIC_SOURCES
from agent_evals.environment.models import KeyDelta, StateDelta
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.evaluation.local_rewards import LocalRewardEvaluator
from agent_evals.evaluation.profiles.base import (
    BenchmarkMetricProfile,
    MetricGroupProfile,
    MetricProfileEntry,
)
from agent_evals.evaluation.progress import (
    NEUTRAL_PROGRESS,
    PipelineStage,
    ScientificProgressTracker,
    infer_stage,
    summarize_progress,
)
from agent_evals.evaluation.stage_rewards import PROGRESS_PREFIX
from agent_evals.evaluation.taxonomy import decision_ontology
from agent_evals.evaluation.trajectory import TrajectoryEvaluator
from agent_evals.metrics.registry import metric_registry

BENCHMARK = Path(__file__).parents[1] / "examples" / "benchmarks" / "pbmc-cell-annotation.yaml"

#: The artifact ids the example benchmark declares, which are what the typed
#: operations really emit.  Two stages are only identifiable from these.
DECLARED_ARTIFACTS = {
    "qc-table": PipelineStage.QC,
    "normalized-anndata": PipelineStage.NORMALIZATION,
    "pca-embedding": PipelineStage.DIMENSIONALITY_REDUCTION,
    "cluster-table": PipelineStage.CLUSTERING,
    "marker-table": PipelineStage.DIFFERENTIAL_EXPRESSION,
    "annotated-anndata": PipelineStage.ANNOTATION,
}


def observed(**fields: object) -> StateDelta:
    """Build a fully observed delta whose unspecified namespaces are empty."""
    return StateDelta.model_validate(
        {
            "n_obs_before": 700,
            "n_obs_after": 700,
            "n_vars_before": 765,
            "n_vars_after": 765,
            "obs_names_changed": False,
            "var_names_changed": False,
            "matrix_changed": False,
            **fields,
        }
    )


def profile(*names: str) -> BenchmarkMetricProfile:
    """A one-domain profile weighting each named metric equally."""
    return BenchmarkMetricProfile(
        benchmark="synthetic",
        metric_groups={
            "biology": MetricGroupProfile(
                weight=1.0,
                metrics={name: MetricProfileEntry(weight=1.0) for name in names},
            )
        },
    )


# --------------------------------------------------------------------------
# Stage inference from what the operations really write
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        # qc writes exactly these three obs columns (probe step 1).
        (
            observed(obs=KeyDelta(added=["n_genes_by_counts", "pct_counts_mt", "total_counts"])),
            PipelineStage.QC,
        ),
        # pca overwrites an existing X_pca rather than adding it (probe step 3).
        (observed(obsm=KeyDelta(changed=["X_pca"])), PipelineStage.DIMENSIONALITY_REDUCTION),
        (observed(obs=KeyDelta(added=["predicted_clusters"])), PipelineStage.CLUSTERING),
        (observed(obs=KeyDelta(added=["predicted_labels"])), PipelineStage.ANNOTATION),
        (observed(var=KeyDelta(added=["highly_variable"])), PipelineStage.FEATURE_SELECTION),
        (observed(obsm=KeyDelta(added=["X_pca_harmony"])), PipelineStage.INTEGRATION),
        (observed(matrix_changed=True), PipelineStage.NORMALIZATION),
    ],
)
def test_stage_is_inferred_from_the_real_operation_signatures(
    delta: StateDelta,
    expected: PipelineStage,
) -> None:
    assert infer_stage(delta) is expected


def test_unrecognized_change_stays_unattributed() -> None:
    assert infer_stage(observed(obs=KeyDelta(added=["something_nobody_declared"]))) is None


@pytest.mark.parametrize(("artifact_id", "expected"), sorted(DECLARED_ARTIFACTS.items()))
def test_every_declared_artifact_identifies_its_own_stage(
    artifact_id: str,
    expected: PipelineStage,
) -> None:
    """No artifact marker may claim another stage's canonically-named output.

    The DE markers in particular are short (``de-``), so this is the guard that
    keeps them from swallowing an unrelated id.
    """
    assert infer_stage(observed(), [artifact_id]) is expected


def test_normalization_that_changed_no_data_is_still_attributed() -> None:
    """A pre-scaled input makes ``normalize`` a genuine no-op on the matrix.

    ``normalize`` skips the transform when ``min(X) < 0``, so on a pre-scaled
    dataset the step leaves every observed namespace untouched.  Observed on a real
    run: without the artifact tier this step was unattributable.
    """
    assert infer_stage(observed()) is None
    assert infer_stage(observed(), ["normalized-anndata"]) is PipelineStage.NORMALIZATION


def test_differential_expression_is_attributed_without_an_observable_namespace() -> None:
    """DE writes ``uns`` and a table, neither of which a dataset diff covers."""
    typed_tier = observed(unobserved=["files"])
    assert infer_stage(typed_tier) is None
    assert infer_stage(typed_tier, ["marker-table"]) is PipelineStage.DIFFERENTIAL_EXPRESSION


def test_dataset_evidence_outranks_artifact_evidence() -> None:
    """A produced file must not override what the data itself shows."""
    delta = observed(obs=KeyDelta(added=["predicted_labels"]))
    assert infer_stage(delta, ["normalized-anndata"]) is PipelineStage.ANNOTATION


# --------------------------------------------------------------------------
# The progress signal
# --------------------------------------------------------------------------


def test_progress_is_positive_when_the_state_improves() -> None:
    tracker = ScientificProgressTracker(profile("clustering.ari"))
    tracker.record(step=1, stage=None, metric_values={"clustering.ari": 0.40})
    second = tracker.record(step=2, stage=None, metric_values={"clustering.ari": 0.70})

    assert second.delta is not None
    assert second.delta > 0
    assert summarize_progress([second], action_count=2).value > NEUTRAL_PROGRESS


def test_progress_is_negative_when_the_state_regresses() -> None:
    tracker = ScientificProgressTracker(profile("clustering.ari"))
    tracker.record(step=1, stage=None, metric_values={"clustering.ari": 0.70})
    second = tracker.record(step=2, stage=None, metric_values={"clustering.ari": 0.40})

    assert second.delta is not None
    assert second.delta < 0
    assert summarize_progress([second], action_count=2).value < NEUTRAL_PROGRESS


def test_a_newly_eligible_metric_does_not_register_as_improvement() -> None:
    """The whole reason deltas are re-aggregated on the shared subset.

    Computing an embedding makes clustering metrics answerable.  A naive
    ``S_t - S_{t-1}`` would read that as a large gain nobody earned.
    """
    tracker = ScientificProgressTracker(profile("clustering.ari", "cell_annotation.macro_f1"))
    tracker.record(step=1, stage=None, metric_values={"clustering.ari": 0.50})
    second = tracker.record(
        step=2,
        stage=None,
        metric_values={"clustering.ari": 0.50, "cell_annotation.macro_f1": 0.95},
    )

    assert second.comparable_metrics == ("clustering.ari",)
    assert second.delta == pytest.approx(0.0, abs=1e-9)


def test_a_delta_with_no_shared_metric_is_unmeasurable_not_zero() -> None:
    tracker = ScientificProgressTracker(profile("clustering.ari", "cell_annotation.macro_f1"))
    tracker.record(step=1, stage=None, metric_values={"clustering.ari": 0.50})
    second = tracker.record(
        step=2, stage=None, metric_values={"cell_annotation.macro_f1": 0.90}
    )

    assert second.delta is None
    assert summarize_progress([second], action_count=2).value is None


def test_recovery_is_distinguishable_from_never_having_regressed() -> None:
    tracker = ScientificProgressTracker(profile("clustering.ari"))
    steady = [0.30, 0.45, 0.60, 0.75]
    dipped = [0.30, 0.75, 0.45, 0.75]

    def run(values: list[float]) -> tuple[int, int]:
        tracker.reset()
        signals = [
            tracker.record(step=index, stage=None, metric_values={"clustering.ari": value})
            for index, value in enumerate(values, start=1)
        ]
        report = summarize_progress(signals, action_count=len(values))
        return report.regressions, report.recoveries

    assert run(steady) == (0, 0)
    assert run(dipped) == (1, 1)


def test_a_flat_run_scores_exactly_neutral() -> None:
    tracker = ScientificProgressTracker(profile("clustering.ari"))
    signals = [
        tracker.record(step=index, stage=None, metric_values={"clustering.ari": 0.5})
        for index in (1, 2, 3)
    ]
    report = summarize_progress(signals, action_count=3)

    assert report.value == pytest.approx(NEUTRAL_PROGRESS)
    assert report.regressions == 0


def test_metrics_outside_the_profile_do_not_move_progress() -> None:
    tracker = ScientificProgressTracker(profile("clustering.ari"))
    tracker.record(step=1, stage=None, metric_values={"clustering.ari": 0.5})
    second = tracker.record(
        step=2,
        stage=None,
        metric_values={"clustering.ari": 0.5, "embedding.trustworthiness": 0.99},
    )

    assert second.delta == pytest.approx(0.0, abs=1e-9)
    assert any("not weighted" in note for note in second.limitations)


# --------------------------------------------------------------------------
# The O/T double count
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trajectory_quality_does_not_move_when_only_the_outcome_changes() -> None:
    """The direct regression test for the old ``0.10 * outcome_alignment`` term."""
    specification = load_benchmark(BENCHMARK)
    environment = ScientificEnvironment(
        specification,
        task_id="cell-annotation",
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )
    run = await AgentHarness().run(
        MockAgentAdapter(), environment, AgentConfiguration(agent_type="mock", seed=1)
    )
    evaluator = TrajectoryEvaluator()
    poor = evaluator.evaluate(run, specification.tasks[0], 0.05, local_rewards=[0.7])
    good = evaluator.evaluate(run, specification.tasks[0], 0.95, local_rewards=[0.7])

    assert poor.outcome_alignment != good.outcome_alignment
    assert poor.trajectory_quality == good.trajectory_quality


@pytest.mark.asyncio
async def test_an_unmeasurable_progress_term_is_excluded_not_zeroed() -> None:
    specification = load_benchmark(BENCHMARK)
    environment = ScientificEnvironment(
        specification,
        task_id="cell-annotation",
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )
    run = await AgentHarness().run(
        MockAgentAdapter(), environment, AgentConfiguration(agent_type="mock", seed=1)
    )
    evaluator = TrajectoryEvaluator()
    absent = evaluator.evaluate(run, specification.tasks[0], 0.5, local_rewards=[0.7])
    neutral = evaluator.evaluate(
        run,
        specification.tasks[0],
        0.5,
        local_rewards=[0.7],
        progress=summarize_progress([], action_count=1),
    )

    assert absent.scientific_progress is None
    assert "scientific_progress" not in absent.formula
    # An unmeasured progress report is the same statement as no report at all.
    assert neutral.trajectory_quality == absent.trajectory_quality


# --------------------------------------------------------------------------
# The shared component vocabulary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("metric_id", sorted({m for ids in COMPONENT_METRIC_SOURCES.values() for m in ids}))
def test_every_component_metric_source_is_a_registered_metric(metric_id: str) -> None:
    """A source naming an unregistered metric can never resolve, silently."""
    assert metric_registry.get(metric_id) is not None


def test_component_names_agree_across_weights_ontology_and_benchmark_yaml() -> None:
    """Three places declare these names and had already drifted apart."""
    specification = load_benchmark(BENCHMARK)
    declared = {
        key: set(spec.metrics) for key, spec in specification.decision_evaluation.items()
    }

    for category, weights in LocalRewardEvaluator._WEIGHTS.items():
        names = {name for name, _weight in weights}
        assert names <= set(decision_ontology.get(category).evaluator_metrics), category
        if category.value in declared:
            assert names == declared[category.value], category


def decision(category: DecisionCategory) -> ScientificDecision:
    """A minimal decision of one category; only the category is under test."""
    return ScientificDecision(
        decision_id="d1",
        episode_id="e1",
        step_id="s1",
        order=1,
        action_category=category.value,
        decision_category=category,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_a_component_with_no_source_is_excluded_rather_than_scored_zero() -> None:
    reward = LocalRewardEvaluator().evaluate(
        decision(DecisionCategory.CLUSTERING), None, None, {"clustering.ari": 0.8}
    )

    assert reward.components == {"ari": pytest.approx(0.8)}
    assert reward.value == pytest.approx(0.8)
    assert any("excluded as unmeasured" in line for line in reward.evidence)


def test_observed_components_are_read_from_the_harness_own_counts() -> None:
    reward = LocalRewardEvaluator().evaluate(
        decision(DecisionCategory.QC_STRATEGY), {"n_obs": 1000}, {"n_obs": 900}, None
    )

    assert reward.components["artifact_removal"] == pytest.approx(0.1)
    assert any("observed cell counts" in line for line in reward.evidence)


# --------------------------------------------------------------------------
# The leakage boundary on the progress evidence
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_evidence_never_reaches_an_agent_visible_observation() -> None:
    """``S_t`` is reference-derived, so it must stay on the evaluator's side."""
    specification = load_benchmark(BENCHMARK)
    environment = ScientificEnvironment(
        specification,
        task_id="cell-annotation",
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )
    run = await AgentHarness().run(
        MockAgentAdapter(), environment, AgentConfiguration(agent_type="mock", seed=1)
    )
    payload = run.model_dump_json()
    state = run.final_environment_state.state

    visible = [
        observation
        for record in state.actions
        for observation in record.result.observations
        if observation.visible_to_agent
    ]
    for observation in visible:
        assert PROGRESS_PREFIX not in str(observation.payload)
    # The prefix may legitimately appear in evaluator-side reward records; what
    # must never happen is it reaching an agent-visible observation above.
    assert isinstance(payload, str)
