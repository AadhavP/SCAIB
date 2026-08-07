"""Tests for deterministic method selection scoring."""

from datetime import UTC, datetime

from agent_evals.agents.trajectory import DecisionCategory, ScientificDecision
from agent_evals.environment.models import ActionStatus
from agent_evals.evaluation.methods import MethodSelectionEvaluator
from agent_evals.evaluation.taxonomy import DecisionProfile


def test_method_selection_scores_method_parameters_and_execution() -> None:
    decision = ScientificDecision(
        decision_id="d1",
        episode_id="e1",
        step_id="s1",
        order=0,
        action_category="qc",
        decision_category=DecisionCategory.QC_STRATEGY,
        chosen_method="adaptive_filter",
        chosen_parameters={"min_genes": 200},
        execution_status=ActionStatus.SUCCEEDED,
        timestamp=datetime.now(UTC),
    )
    profile = DecisionProfile(
        category=DecisionCategory.QC_STRATEGY,
        allowed_methods=["adaptive_filter"],
        parameter_ranges={"min_genes": {"minimum": 0, "maximum": 1000}},
    )

    result = MethodSelectionEvaluator().evaluate(decision, {}, profile, {"quality": 0.9})

    assert result.appropriateness == 1
    assert result.parameter_quality == 1
    assert result.execution_quality == 0.9
    assert result.overall > 0.9
