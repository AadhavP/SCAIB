"""Tests for deterministic method selection scoring.

The second half of this module is the regression test for three substituted
numbers. Each component of a method score used to be answered even when it was
unanswerable -- appropriateness with a neutral 0.5, parameter quality with a
*perfect* 1.0, execution quality with a failing 0.0 -- so the same score could
mean "measured and good" or "never asked". Each now reports ``None`` and is
excluded from ``overall``.
"""

from datetime import UTC, datetime

import pytest

from agent_evals.agents.trajectory import DecisionCategory, ScientificDecision
from agent_evals.environment.models import ActionStatus
from agent_evals.evaluation.methods import MethodSelectionEvaluator
from agent_evals.evaluation.taxonomy import DecisionProfile


def _decision(**overrides: object) -> ScientificDecision:
    fields: dict[str, object] = {
        "decision_id": "d1",
        "episode_id": "e1",
        "step_id": "s1",
        "order": 0,
        "action_category": "qc",
        "decision_category": DecisionCategory.QC_STRATEGY,
        "chosen_method": "adaptive_filter",
        "chosen_parameters": {"min_genes": 200},
        "execution_status": ActionStatus.SUCCEEDED,
        "timestamp": datetime.now(UTC),
    }
    fields.update(overrides)
    return ScientificDecision(**fields)  # type: ignore[arg-type]


FULL_PROFILE = DecisionProfile(
    category=DecisionCategory.QC_STRATEGY,
    allowed_methods=["adaptive_filter"],
    parameter_ranges={"min_genes": {"minimum": 0, "maximum": 1000}},
)


def test_method_selection_scores_method_parameters_and_execution() -> None:
    result = MethodSelectionEvaluator().evaluate(
        _decision(), {}, FULL_PROFILE, {"quality": 0.9}
    )

    assert result.appropriateness == 1
    assert result.parameter_quality == 1
    assert result.execution_quality == 0.9
    assert result.overall > 0.9
    assert result.unmeasured_components == []


def test_an_undeclared_method_list_leaves_appropriateness_unmeasured() -> None:
    """A benchmark that restricted nothing has not judged the method neutral."""
    profile = DecisionProfile(
        category=DecisionCategory.QC_STRATEGY,
        allowed_methods=[],
        parameter_ranges={"min_genes": {"minimum": 0, "maximum": 1000}},
    )

    result = MethodSelectionEvaluator().evaluate(
        _decision(), {}, profile, {"quality": 0.9}
    )

    assert result.appropriateness is None
    assert result.unmeasured_components == ["appropriateness"]
    # 0.5 used to sit in the average and drag a perfect run down to 0.8.
    assert result.overall == pytest.approx((1.0 + 0.9) / 2)


def test_undeclared_parameter_ranges_leave_parameter_quality_unmeasured() -> None:
    """The costliest substitution: a free third of the score, paid to everyone."""
    profile = DecisionProfile(
        category=DecisionCategory.QC_STRATEGY,
        allowed_methods=["adaptive_filter"],
        parameter_ranges={},
    )

    result = MethodSelectionEvaluator().evaluate(
        _decision(), {}, profile, {"quality": 0.6}
    )

    assert result.parameter_quality is None
    assert result.unmeasured_components == ["parameter_quality"]
    assert result.overall == pytest.approx((1.0 + 0.6) / 2)


def test_a_step_no_metric_could_answer_is_unmeasured_not_a_failed_execution() -> None:
    """Scoring 0.0 recorded "not yet assessable" as "executed badly"."""
    result = MethodSelectionEvaluator().evaluate(_decision(), {}, FULL_PROFILE, {})

    assert result.execution_quality is None
    assert result.unmeasured_components == ["execution_quality"]
    assert result.overall == pytest.approx(1.0)


def test_execution_status_is_still_scored_when_no_metrics_were_requested() -> None:
    """``None`` results means "do not consult metrics", not "nothing observable"."""
    succeeded = MethodSelectionEvaluator().evaluate(
        _decision(), {}, FULL_PROFILE, None
    )
    failed = MethodSelectionEvaluator().evaluate(
        _decision(execution_status=ActionStatus.FAILED), {}, FULL_PROFILE, None
    )

    assert succeeded.execution_quality == 1.0
    assert failed.execution_quality == 0.0
    assert succeeded.unmeasured_components == []


def test_a_wholly_unmeasurable_selection_has_no_overall_score() -> None:
    """Nothing to average is ``None``, never 0.0 and never 1.0."""
    result = MethodSelectionEvaluator().evaluate(
        _decision(chosen_method=None),
        {},
        DecisionProfile(category=DecisionCategory.QC_STRATEGY),
        {},
    )

    assert result.overall is None
    assert result.unmeasured_components == [
        "appropriateness",
        "parameter_quality",
        "execution_quality",
    ]


def test_every_unmeasured_component_says_so_in_the_evidence() -> None:
    """A gap a reader cannot see recorded is a gap they will read as a score."""
    result = MethodSelectionEvaluator().evaluate(
        _decision(chosen_method=None),
        {},
        DecisionProfile(category=DecisionCategory.QC_STRATEGY),
        {},
    )

    joined = " ".join(result.evidence)
    assert joined.count("unmeasured") == 3
