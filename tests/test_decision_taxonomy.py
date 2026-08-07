"""Tests for the benchmark-independent scientific decision ontology."""

from datetime import UTC, datetime

from agent_evals.agents.trajectory import DecisionCategory, ScientificDecision
from agent_evals.evaluation.taxonomy import (
    DecisionOntology,
    DecisionProfile,
    default_decision_ontology,
)


def test_default_taxonomy_has_scientific_categories() -> None:
    ontology = default_decision_ontology()
    assert ontology.get(DecisionCategory.QC_STRATEGY).allowed_methods
    assert ontology.get(DecisionCategory.INTEGRATION).evaluator_metrics


def test_decision_exposes_structured_choice_without_private_reasoning() -> None:
    decision = ScientificDecision(
        decision_id="d1",
        episode_id="e1",
        step_id="s1",
        order=0,
        action_category="harmony",
        decision_category=DecisionCategory.INTEGRATION,
        intent="remove technical variation",
        hypothesis="batch effects are obscuring cell populations",
        chosen_method="harmony",
        chosen_parameters={"theta": 2},
        evidence_used=["batch-labels"],
        confidence=0.8,
        expected_effect={"batch_mixing": 0.8},
        downstream_dependency={"feeds": "clustering"},
        timestamp=datetime.now(UTC),
    )

    assert decision.method == "harmony"
    assert decision.parameters == {"theta": 2}
    assert decision.decision_category == DecisionCategory.INTEGRATION


def test_custom_ontology_profile_is_resolvable() -> None:
    ontology = DecisionOntology()
    ontology.register(DecisionProfile(category=DecisionCategory.QC_STRATEGY, allowed_methods=["adaptive_filter"]))
    assert ontology.get(DecisionCategory.QC_STRATEGY).allowed_methods == ["adaptive_filter"]
