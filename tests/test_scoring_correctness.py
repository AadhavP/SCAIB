"""Tests for per-benchmark scoring weights and the reported measurement gaps.

Five claims are on trial:

1. A benchmark can declare how it weighs outcome, decisions, and trajectory --
   but not a weighting that would stop being a mean, and the refusal happens when
   the YAML is read rather than after a paid run.
2. The two ineligible fractions that qualify a score are derived from what the
   harness actually recorded, not estimated. ``ineligible_fraction_T`` is the
   share of declared trajectory weight that went unmeasured;
   ``ineligible_fraction_D`` is the share of method-score components that did.
3. Both are reported beside the score and neither moves it.
4. A status the harness computed reaches the archive intact. Three vocabularies
   used to collapse distinctions their producers had already drawn: metric
   failures all read ``FAILED``, and a run that stopped early read ``FAILED``
   like one that crashed. Each collapse was a silent fallback, so a green suite
   could not see it.
5. Nothing gets a score it did not earn, in either direction. Two dimensions were
   settled without being measured: stability was awarded ``1.0`` for never having
   been tested, and two metrics SCAIB has never implemented charged the agent
   ``0.0``. Both are now dropped from the aggregate, which lowers published
   annotation scores -- that is the point, and it is why these land together.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_evals.agents.runtime.manager import (
    _FAILURE_KINDS,
    _TERMINATION_STATUSES,
    RuntimeVerdict,
)
from agent_evals.agents.trajectory import FailureKind, RunTerminationStatus
from agent_evals.benchmarks.schema import ScoringSpecification
from agent_evals.environment.models import ArtifactRecord
from agent_evals.environment.scientific_loop import _ineligible_fraction_decision
from agent_evals.evaluation.metrics.base import MetricStatus as FacadeStatus
from agent_evals.evaluation.metrics.robustness import RobustnessEvaluator
from agent_evals.evaluation.models import MethodScore
from agent_evals.evaluation.profiles.base import MetricGroupProfile, MetricProfileEntry
from agent_evals.evaluation.scientific import ScientificMetricEngine
from agent_evals.evaluation.scoring.aggregation import (
    DomainScore,
    MetricScoreInput,
    WeightedGeometricAggregator,
)
from agent_evals.evaluation.scoring.domains import (
    UNRECORDED_METRIC_REASON,
    aggregate_domains,
    describe_unmeasured_domains,
)
from agent_evals.evaluation.trajectory import _QUALITY_WEIGHTS, _weighted_quality
from agent_evals.metrics.builtin._helpers import failed, unavailable
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.results import MetricResult, MetricStatus

# --------------------------------------------------------------------------- #
# Per-benchmark weights
# --------------------------------------------------------------------------- #


def test_the_default_weighting_asserts_nothing() -> None:
    scoring = ScoringSpecification()

    assert scoring.outcome_weight == pytest.approx(1 / 3)
    assert scoring.decision_weight == pytest.approx(1 / 3)
    assert scoring.trajectory_weight == pytest.approx(1 / 3)


def test_a_benchmark_may_redistribute_emphasis() -> None:
    scoring = ScoringSpecification(
        outcome_weight=0.6, decision_weight=0.3, trajectory_weight=0.1
    )

    assert scoring.outcome_weight == 0.6


def test_weights_that_would_inflate_this_benchmarks_scores_are_refused() -> None:
    """Exponents summing to 0.5 raise every score on this benchmark alone, which
    would make its numbers incomparable with every other benchmark's."""
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        ScoringSpecification(
            outcome_weight=0.2, decision_weight=0.2, trajectory_weight=0.1
        )


def test_weights_that_would_depress_this_benchmarks_scores_are_refused() -> None:
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        ScoringSpecification(
            outcome_weight=0.6, decision_weight=0.6, trajectory_weight=0.6
        )


def test_the_confidence_penalties_are_declarable_and_separate() -> None:
    """kD and kT scale what a gap costs *confidence*; a benchmark that can only
    observe part of the trajectory may say so without touching the score."""
    scoring = ScoringSpecification(trajectory_confidence_penalty=0.25)

    assert scoring.trajectory_confidence_penalty == 0.25
    assert scoring.decision_confidence_penalty == 0.5


# --------------------------------------------------------------------------- #
# ineligible_fraction_T -- the share of declared trajectory weight not measured
# --------------------------------------------------------------------------- #

_MEASURABLE = {name: 1.0 for name, _ in _QUALITY_WEIGHTS}


def test_a_fully_measured_trajectory_reports_no_gap() -> None:
    _, _, unmeasured = _weighted_quality(_MEASURABLE)

    assert unmeasured == pytest.approx(0.0)


def test_the_gap_is_the_share_of_declared_weight_that_went_missing() -> None:
    """0.84 from every term and 0.84 from two of seven are not equally
    trustworthy claims, and only this number tells them apart."""
    weights = dict(_QUALITY_WEIGHTS)
    terms: dict[str, float | None] = dict(_MEASURABLE)
    terms["scientific_progress"] = None

    _, _, unmeasured = _weighted_quality(terms)

    assert unmeasured == pytest.approx(weights["scientific_progress"])


def test_a_trajectory_nobody_could_observe_reports_a_total_gap() -> None:
    _, formula, unmeasured = _weighted_quality(dict.fromkeys(_MEASURABLE))

    assert unmeasured == pytest.approx(1.0)
    assert "no trajectory term was measurable" in formula


def test_excluding_a_term_does_not_change_the_quality_of_the_rest() -> None:
    """The gap is reported *because* renormalization hides it: dropping an
    unmeasurable term leaves the score untouched and the evidence thinner."""
    terms: dict[str, float | None] = dict(_MEASURABLE)
    terms["scientific_progress"] = None

    full_quality, _, _ = _weighted_quality(_MEASURABLE)
    thin_quality, _, thin_gap = _weighted_quality(terms)

    assert thin_quality == pytest.approx(full_quality)
    assert thin_gap > 0.0


# --------------------------------------------------------------------------- #
# ineligible_fraction_D -- the share of method-score components not measured
# --------------------------------------------------------------------------- #


def _selection(*unmeasured: str) -> MethodScore:
    return MethodScore(
        decision_id="d1", method="leiden", unmeasured_components=list(unmeasured)
    )


def test_a_run_with_no_decisions_reports_no_decision_gap() -> None:
    """Nothing recorded is handled by the score being ``None`` outright; inventing
    a gap here would double-count the same absence."""
    assert _ineligible_fraction_decision([]) == 0.0


def test_a_fully_measured_decision_reports_no_gap() -> None:
    assert _ineligible_fraction_decision([_selection()]) == 0.0


def test_the_gap_is_counted_over_components_not_over_decisions() -> None:
    """A decision whose category declares no parameter ranges still yields real
    evidence about appropriateness and execution; calling the whole decision
    ineligible would understate the run as badly as substituting numbers
    overstated it."""
    assert _ineligible_fraction_decision(
        [_selection("parameter_quality")]
    ) == pytest.approx(1 / 3)


def test_the_gap_spans_every_decision_in_the_run() -> None:
    fraction = _ineligible_fraction_decision(
        [
            _selection("parameter_quality"),
            _selection(),
            _selection("appropriateness", "parameter_quality", "execution_quality"),
        ]
    )

    assert fraction == pytest.approx(4 / 9)


def test_a_wholly_unmeasurable_decision_dimension_reports_a_total_gap() -> None:
    fraction = _ineligible_fraction_decision(
        [_selection("appropriateness", "parameter_quality", "execution_quality")]
    )

    assert fraction == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# One metric-status vocabulary, and one that names the cause of a failure
# --------------------------------------------------------------------------- #


def test_the_two_metric_stacks_share_one_status_enum() -> None:
    """They used to declare identical three-member copies, so the values compared
    unequal while printing the same -- drift with nothing to catch it."""
    assert FacadeStatus is MetricStatus


def test_the_pre_split_names_still_resolve() -> None:
    assert MetricStatus.COMPUTED is MetricStatus.SCORED
    assert MetricStatus.STRUCTURALLY_INELIGIBLE is MetricStatus.INELIGIBLE


def test_a_result_persisted_before_the_split_still_loads() -> None:
    """Aliasing the names is not enough: archived JSON carries the old *value*,
    and a benchmark that cannot read its own published results has broken its
    reproducibility claim to save a rename."""
    archived = {
        "metric_id": "clustering.ari",
        "version": "1.0",
        "metric_name": "ARI",
        "role": "primary",
        "direction": "higher_is_better",
        "normalized_value": 0.81,
        "eligible": True,
        "status": "computed",
        "eligibility_reason": "eligible",
    }

    loaded = MetricResult.model_validate(archived)

    assert loaded.status is MetricStatus.SCORED
    assert MetricStatus("structurally_ineligible") is MetricStatus.INELIGIBLE
    assert MetricStatus("failed") is MetricStatus.FAILED


def test_an_unknown_status_is_still_rejected() -> None:
    """The compatibility hook maps the spellings that existed, not any string."""
    with pytest.raises(ValueError, match="invented"):
        MetricStatus("invented")


def test_an_evaluator_crash_is_named_without_being_forgiven() -> None:
    """The one place the "unmeasured, so unscored" rule is deliberately not
    applied. The agent picks the arrays a computer runs on, so excluding a crash
    would let it delete a metric it was about to fail."""
    profile = MetricGroupProfile(
        weight=1.0,
        metrics={"clustering.ari": MetricProfileEntry(weight=1.0, required=True)},
    )

    score = WeightedGeometricAggregator().aggregate(
        "biology",
        profile,
        [
            MetricScoreInput(
                name="clustering.ari",
                value=0.0,
                status=MetricStatus.EVALUATOR_ERROR,
            )
        ],
    )

    assert score.included_metrics == ["clustering.ari"]
    assert score.failed_metrics == ["clustering.ari"]
    assert score.value == 0.0


def test_only_a_scored_metric_is_reported_as_having_succeeded() -> None:
    """This comparison was against the bare literal ``"computed"``. Renaming the
    value would have moved every metric on a healthy run into ``failed_metrics``
    without anything raising."""
    profile = MetricGroupProfile(
        weight=1.0,
        metrics={"clustering.ari": MetricProfileEntry(weight=1.0, required=True)},
    )

    score = WeightedGeometricAggregator().aggregate(
        "biology",
        profile,
        [MetricScoreInput(name="clustering.ari", value=0.81)],
    )

    assert score.failed_metrics == []
    assert score.value == pytest.approx(0.81)


# --------------------------------------------------------------------------- #
# A run that stopped early is not a run that broke
# --------------------------------------------------------------------------- #


def test_every_runtime_verdict_reaches_the_archive_as_itself() -> None:
    """The lookup falls back to ``FAILED``, so a verdict the table forgot would
    vanish silently rather than raise. This is the guard that makes the fallback
    safe to keep."""
    for verdict in RuntimeVerdict:
        assert verdict in _TERMINATION_STATUSES, verdict
        assert _TERMINATION_STATUSES[verdict].value == verdict.value


def test_stopping_early_is_recorded_apart_from_breaking() -> None:
    """The runtime already knew the difference and it was being flattened on the
    way into the run record."""
    assert (
        _TERMINATION_STATUSES[RuntimeVerdict.INCOMPLETE]
        is RunTerminationStatus.INCOMPLETE
    )
    assert (
        _TERMINATION_STATUSES[RuntimeVerdict.INCOMPLETE]
        is not _TERMINATION_STATUSES[RuntimeVerdict.FAILED]
    )


def test_an_unmet_contract_is_not_filed_as_a_malfunction() -> None:
    """Nothing errored: the agent ran cleanly and quit with artifacts missing.
    Recording that as ``AGENT_ERROR`` sends whoever reads the archive looking for
    a bug that is not there."""
    assert (
        _FAILURE_KINDS[RunTerminationStatus.INCOMPLETE]
        is FailureKind.INCOMPLETE_SUBMISSION
    )
    assert _FAILURE_KINDS[RunTerminationStatus.TIMEOUT] is FailureKind.TIMEOUT
    assert FailureKind.AGENT_ERROR not in _FAILURE_KINDS.values()


def test_the_statuses_left_out_of_the_table_are_the_ones_agent_error_fits() -> None:
    """The fallback is deliberate for exactly two statuses, not an oversight.

    A run that *broke* is an agent error, and so is one cancelled mid-flight with
    no other explanation, so both are left to the fallback on purpose. Recording
    that here stops someone from "completing" the table and inventing a kind for
    them, and stops a future status from being added and silently inheriting
    ``AGENT_ERROR`` because nobody noticed the table has a floor.
    """
    unmapped = {
        status for status in RunTerminationStatus if status not in _FAILURE_KINDS
    }

    assert unmapped == {
        RunTerminationStatus.COMPLETED,  # never reaches the failure branch at all
        RunTerminationStatus.FAILED,
        RunTerminationStatus.CANCELLED,
    }


# --------------------------------------------------------------------------- #
# Stability that was never tested is not stability
# --------------------------------------------------------------------------- #


def _replicate(seed: int, labels: list[str]) -> dict[str, object]:
    return {"seed": seed, "predicted_labels": labels, "artifact_checksums": ["x"]}


def test_a_single_replicate_cannot_measure_stability() -> None:
    """Every pairwise dimension needs two runs, so with one there is nothing to
    compare. This returned ``1.0``: a perfect score for the one property the run
    had no way of demonstrating, carrying the robustness domain's full 0.2 weight
    in the annotation profile on every run the loop has ever scored."""
    report = RobustnessEvaluator().evaluate([_replicate(1, ["a", "b"])])

    assert report.seed_stability is None
    assert report.clustering_pairwise_ari is None
    assert "unmeasured" in report.formula


def test_two_comparable_replicates_are_still_measured() -> None:
    """The correction must not swallow the case the evaluator exists for."""
    report = RobustnessEvaluator().evaluate(
        [_replicate(1, ["a", "b"]), _replicate(2, ["a", "b"])]
    )

    assert report.seed_stability == pytest.approx(1.0)
    assert report.annotation_prediction_agreement == pytest.approx(1.0)


def _robustness_profile() -> MetricGroupProfile:
    return MetricGroupProfile(
        weight=0.2, metrics={}, external_score="robustness.seed_stability"
    )


def test_an_unmeasured_external_score_is_dropped_rather_than_scored() -> None:
    score = WeightedGeometricAggregator().aggregate(
        "robustness",
        _robustness_profile(),
        [
            MetricScoreInput(
                name="robustness.seed_stability",
                value=None,
                applicable=False,
                structurally_ineligible=True,
                status=MetricStatus.MISSING,
            )
        ],
    )

    assert score.value is None
    assert score.excluded_metrics == ["robustness.seed_stability"]
    assert score.included_metrics == []


def test_leaving_the_unmeasured_score_out_would_have_scored_it_zero() -> None:
    """Why the loop appends the input and marks it, rather than not appending it.

    An ``external_score`` is injected as a *required* entry, and a required entry
    with no matching result is scored ``0.0`` rather than skipped. So the obvious
    implementation -- omit it when there is nothing to report -- would have traded
    a free 1.0 for an unearned 0.0 and been wrong in the opposite direction.
    """
    score = WeightedGeometricAggregator().aggregate(
        "robustness", _robustness_profile(), []
    )

    assert score.value == 0.0
    assert score.failed_metrics == ["robustness.seed_stability"]


def test_the_scientific_formula_names_only_the_domains_it_combined() -> None:
    """The weights are renormalized over the measured domains, so a formula listing
    all of them described a computation that did not happen."""
    measured = DomainScore(
        domain="biology", value=0.5, weight=0.6, formula="geometric_mean(a^1)"
    )
    unmeasured = DomainScore(
        domain="robustness", value=None, weight=0.2, formula="geometric_mean()"
    )

    score = aggregate_domains([measured, unmeasured])

    # Renormalized onto biology alone rather than pulled toward 1.0 by a domain
    # nobody measured.
    assert score.value == pytest.approx(0.5)
    # Asserted whole rather than by substring. ``"biology" in formula`` stays true
    # when the string names every domain, which is the exact claim under test.
    assert score.formula == (
        "weighted_geometric_mean(biology) excluding_unmeasured(robustness)"
    )


# --------------------------------------------------------------------------- #
# The formula says which domains were dropped; these say why each one was
# droppable. Publishing one without the other leaves a ``None`` outcome that a
# reader cannot distinguish from metrics legitimately not applying -- which is
# what a differential-expression run did until per-benchmark profiles made a
# wholly unmeasured outcome reachable for the first time.
# --------------------------------------------------------------------------- #


def test_an_unmeasured_domain_states_a_cause_beside_the_formula_that_names_it() -> None:
    """The gap the field's own docstring promised to close and did not."""
    dropped = DomainScore(
        domain="biology",
        value=None,
        weight=1.0,
        excluded_metrics=["differential_expression.precision_at_k"],
        blocking_metrics=["differential_expression.precision_at_k"],
        formula="geometric_mean()",
    )

    described = describe_unmeasured_domains(
        [dropped],
        {
            "differential_expression.precision_at_k": (
                "structural requirements unavailable: metadata:reference_markers"
            )
        },
    )

    assert described == [
        "domain 'biology' unmeasured: differential_expression.precision_at_k "
        "(structural requirements unavailable: metadata:reference_markers)"
    ]


def test_a_measured_domain_is_not_described_at_all() -> None:
    """Otherwise the field narrates healthy runs and stops meaning "limitation"."""
    measured = DomainScore(
        domain="biology", value=0.81, weight=0.6, formula="geometric_mean(a^1)"
    )

    assert describe_unmeasured_domains([measured], {}) == []


def test_the_required_exclusion_is_named_without_the_optional_ones_beside_it() -> None:
    """One metric voided the domain; the others merely did not apply.

    Presenting all three as equal causes would make the real one a third as
    prominent as it is, and a reader deciding whether the benchmark or the agent
    is at fault needs exactly that distinction.
    """
    dropped = DomainScore(
        domain="biology",
        value=None,
        weight=1.0,
        excluded_metrics=["optional.one", "required.blocker", "optional.two"],
        blocking_metrics=["required.blocker"],
        formula="geometric_mean()",
    )

    described = describe_unmeasured_domains(
        [dropped],
        {
            "required.blocker": "reference unavailable",
            "optional.one": "not applicable to this dataset",
            "optional.two": "not applicable to this dataset",
        },
    )

    assert described == ["domain 'biology' unmeasured: required.blocker (reference unavailable)"]


def test_every_exclusion_is_named_when_none_of_them_was_required() -> None:
    """The complement: with no blocking metric there is no cause to prefer, and
    dropping the list would leave the domain unexplained."""
    dropped = DomainScore(
        domain="technical",
        value=None,
        weight=0.2,
        excluded_metrics=["batch_integration.iLISI", "batch_integration.graph_connectivity"],
        formula="geometric_mean()",
    )

    described = describe_unmeasured_domains(
        [dropped],
        {
            "batch_integration.iLISI": "no batch key",
            "batch_integration.graph_connectivity": "no neighbour graph",
        },
    )

    assert described == [
        "domain 'technical' unmeasured: batch_integration.iLISI (no batch key); "
        "batch_integration.graph_connectivity (no neighbour graph)"
    ]


def test_a_metric_nobody_recorded_a_verdict_for_is_still_named() -> None:
    """An ``external_score`` whose evaluator produced nothing has no
    ``MetricResult`` and therefore no reason. Skipping it would shorten the
    explanation in exactly the case where least is known."""
    dropped = DomainScore(
        domain="robustness",
        value=None,
        weight=0.2,
        excluded_metrics=["robustness.seed_stability"],
        blocking_metrics=["robustness.seed_stability"],
        formula="geometric_mean()",
    )

    described = describe_unmeasured_domains([dropped], {})

    assert described == [
        f"domain 'robustness' unmeasured: robustness.seed_stability ({UNRECORDED_METRIC_REASON})"
    ]


def test_a_domain_with_nothing_to_exclude_still_says_it_was_unmeasured() -> None:
    """A profile group with no entries drops out with an empty exclusion list, and
    silence there is the absent-versus-unobserved ambiguity all over again."""
    dropped = DomainScore(
        domain="technical", value=None, weight=0.2, formula="geometric_mean()"
    )

    assert describe_unmeasured_domains([dropped], {}) == [
        "domain 'technical' unmeasured: no metric in this domain produced a value"
    ]


def test_the_aggregator_names_which_required_metric_voided_the_domain() -> None:
    """``blocking_metrics`` exists because only this layer knows requiredness.

    The domain scored ``None`` while ``clustering.ari`` scored 0.81, so a reader
    given only ``excluded_metrics`` cannot tell which of the exclusions mattered.
    """
    profile = MetricGroupProfile(
        weight=1.0,
        metrics={
            "clustering.ari": MetricProfileEntry(weight=0.5),
            "cell_annotation.rare_recall": MetricProfileEntry(weight=0.3),
            "cell_annotation.ece": MetricProfileEntry(weight=0.2, required=False),
        },
    )

    score = WeightedGeometricAggregator().aggregate(
        "biology",
        profile,
        [
            MetricScoreInput(name="clustering.ari", value=0.81),
            MetricScoreInput(
                name="cell_annotation.rare_recall",
                value=None,
                applicable=False,
                structurally_ineligible=True,
                status=MetricStatus.INELIGIBLE,
            ),
        ],
    )

    assert score.value is None
    assert score.blocking_metrics == ["cell_annotation.rare_recall"]
    # The optional one is excluded too, and is deliberately not blocking.
    assert sorted(score.excluded_metrics) == [
        "cell_annotation.ece",
        "cell_annotation.rare_recall",
    ]


def test_nothing_blocks_a_domain_whose_only_exclusions_were_optional() -> None:
    """The presence complement: marking every exclusion as blocking would make the
    field a duplicate of ``excluded_metrics`` and explain nothing."""
    profile = MetricGroupProfile(
        weight=1.0,
        metrics={
            "clustering.ari": MetricProfileEntry(weight=0.8),
            "cell_annotation.ece": MetricProfileEntry(weight=0.2, required=False),
        },
    )

    score = WeightedGeometricAggregator().aggregate(
        "biology", profile, [MetricScoreInput(name="clustering.ari", value=0.81)]
    )

    assert score.value == pytest.approx(0.81)
    assert score.blocking_metrics == []
    assert score.excluded_metrics == ["cell_annotation.ece"]


# --------------------------------------------------------------------------- #
# A metric SCAIB never implemented is not a metric the agent failed
# --------------------------------------------------------------------------- #


def test_the_two_stub_metrics_report_a_harness_gap_not_an_agent_zero() -> None:
    """``kBET`` and ``BRAS`` have never had an implementation. Both returned the
    same value as a genuine computation failure, which carries their failure score
    of 0.0 -- so flawless batch integration was charged for SCAIB's unfinished
    work.

    Asserted through the real engine, because the status is only reachable once
    applicability has passed: a context without an ``embedding`` artifact or a
    ``batch`` key is rejected earlier and would make this pass for the wrong
    reason.
    """
    context = ScientificMetricContext(
        candidate_artifacts={"embedding": [[0.0, 1.0], [1.0, 0.0]]},
        metadata={"batch": "batch"},
    )

    results, _, _, _ = ScientificMetricEngine().evaluate(
        ["batch_integration.kBET", "batch_integration.BRAS"], context
    )

    assert [result.status for result in results] == [MetricStatus.UNIMPLEMENTED] * 2
    for result in results:
        assert result.normalized_value is None, result.metric_id
        assert result.eligible is False
        # Names which gap, so a result file says what its build could not compute
        # instead of publishing a 0.0 that reads as a measurement.
        assert result.metadata["unavailable_reason"]


def test_the_computer_distinguishes_no_backend_from_no_number() -> None:
    """Two situations that were spelled identically at the call site."""
    assert unavailable("nothing is wired").unavailable is True
    assert failed("the input made no sense").unavailable is False
    # Both still carry no value, which is why the flag is what tells them apart.
    assert unavailable("nothing is wired").raw_value is None
    assert failed("the input made no sense").raw_value is None


def test_only_the_statuses_outside_the_agents_control_are_excluded() -> None:
    """The anti-gaming half, and the reason this vocabulary is one shared set.

    Exclusion must cover exactly the outcomes an agent cannot influence. Every
    status it *can* influence has to stay in the aggregate at its failure score,
    or the cheapest way to avoid failing a metric becomes arranging for it not to
    be computed -- by withholding an input, by handing the computer something
    malformed, or by making the evaluator raise.
    """
    excluded = {status for status in MetricStatus if status.excluded_from_scoring}

    assert excluded == {MetricStatus.INELIGIBLE, MetricStatus.UNIMPLEMENTED}
    for status in (
        MetricStatus.MISSING,
        MetricStatus.MALFORMED,
        MetricStatus.EVALUATOR_ERROR,
        MetricStatus.FAILED,
        MetricStatus.SCORED,
    ):
        assert not status.excluded_from_scoring, status


def test_an_unimplemented_metric_leaves_its_domain_intact() -> None:
    """The rest of the group keeps its weights and the gap simply goes.

    ``required=False`` is load-bearing: an excluded entry the profile marks
    required collapses the whole domain to ``None``. So whichever stage finally
    puts kBET in a group has to declare it optional, or excluding it will silently
    delete the domain it was meant to be dropped from.
    """
    profile = MetricGroupProfile(
        weight=1.0,
        metrics={
            "clustering.ari": MetricProfileEntry(weight=0.5),
            "batch_integration.kBET": MetricProfileEntry(weight=0.5, required=False),
        },
    )

    score = WeightedGeometricAggregator().aggregate(
        "technical",
        profile,
        [
            MetricScoreInput(name="clustering.ari", value=0.81),
            MetricScoreInput(
                name="batch_integration.kBET",
                value=None,
                applicable=False,
                structurally_ineligible=True,
                status=MetricStatus.UNIMPLEMENTED,
            ),
        ],
    )

    assert score.excluded_metrics == ["batch_integration.kBET"]
    assert score.included_metrics == ["clustering.ari"]
    assert score.value == pytest.approx(0.81)
    # And not reported as a metric the agent failed, which is what a 0.0 would
    # have looked like to whoever read the scorecard.
    assert score.failed_metrics == []


def test_an_artifact_is_unvalidated_until_something_checks_it() -> None:
    """``validated`` now means a check passed, so the default has to be ``False``.

    The mock executor set it to ``True`` on records carrying no ``uri`` -- a
    producer certifying output no reader could open, which nothing could
    contradict and which earned 0.15 of ``trajectory_quality``.
    """
    record = ArtifactRecord(artifact_id="cell-labels", kind="table", format="csv")

    assert record.validated is False
    assert record.validation is None


# --------------------------------------------------------------------------- #
# The feed, not just the aggregator: what gets marked on the way in
# --------------------------------------------------------------------------- #


def _metric_result(status: MetricStatus, value: float | None) -> MetricResult:
    return MetricResult(
        metric_id="batch_integration.kBET",
        version="1.0",
        metric_name="kBET",
        role="primary",
        direction="higher_is_better",
        normalized_value=value,
        eligible=status is MetricStatus.SCORED,
        status=status,
        eligibility_reason="eligible",
    )


def test_the_feed_marks_a_metric_the_agent_could_not_affect() -> None:
    """The aggregator only drops what the feed marks, so both halves need a test.

    The aggregator tests above construct their own inputs, so they pass no matter
    what the loop puts in them -- which is how a feed that named a single status
    survived every one of them.
    """
    unimplemented = MetricScoreInput.from_metric_result(
        _metric_result(MetricStatus.UNIMPLEMENTED, None)
    )

    assert unimplemented.structurally_ineligible is True
    assert unimplemented.status is MetricStatus.UNIMPLEMENTED
    # And the id the profiles key on, not the human-readable title.
    assert unimplemented.name == "batch_integration.kBET"


def test_the_feed_does_not_excuse_a_metric_the_agent_could_have_affected() -> None:
    """The other direction, which is the one that could be gamed."""
    for status in (MetricStatus.MISSING, MetricStatus.MALFORMED):
        marked = MetricScoreInput.from_metric_result(_metric_result(status, 0.0))

        assert marked.structurally_ineligible is False, status
        assert marked.value == 0.0


def test_the_feed_marks_an_unmeasured_external_score() -> None:
    unmeasured = MetricScoreInput.from_external_score("robustness.seed_stability", None)

    assert unmeasured.structurally_ineligible is True
    assert unmeasured.value is None
    assert unmeasured.status is MetricStatus.MISSING


def test_the_feed_leaves_a_measured_external_score_scoreable() -> None:
    measured = MetricScoreInput.from_external_score("robustness.seed_stability", 0.75)

    assert measured.structurally_ineligible is False
    assert measured.value == 0.75
    assert measured.status is MetricStatus.SCORED
