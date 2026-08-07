"""Tests for deterministic local decision reward formulas."""

from datetime import UTC, datetime

import pytest

from agent_evals.agents.trajectory import DecisionCategory, ScientificDecision
from agent_evals.evaluation.local_rewards import LocalRewardEvaluator


def test_qc_local_reward_uses_declared_component_weights() -> None:
    decision = ScientificDecision(
        decision_id="d1",
        episode_id="e1",
        step_id="s1",
        order=0,
        action_category="qc",
        decision_category=DecisionCategory.QC_STRATEGY,
        timestamp=datetime.now(UTC),
    )
    result = LocalRewardEvaluator().evaluate(
        decision,
        {"quality": "before"},
        {"quality": "after"},
        {"artifact_removal": 1.0, "biological_retention": 0.8, "rare_population_preservation": 0.6},
    )

    assert result.value == pytest.approx(0.82)
    assert result.formula == "0.4*artifact_removal + 0.3*biological_retention + 0.3*rare_population_preservation"
