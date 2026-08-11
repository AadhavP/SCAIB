"""Translate observable scientific decisions into validated action intents."""

from __future__ import annotations

from typing import ClassVar

from agent_evals.agents.trajectory import ScientificDecision
from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
from agent_evals.environment.models import (
    ActionIntent,
    ActionValidationResult,
    EpisodeSnapshot,
)
from agent_evals.environment.ports import DeclarativeActionValidator


class ActionMappingError(ValueError):
    """Raised when an agent decision cannot become a legal scientific action."""


class ScientificActionMapper:
    """Resolve decision methods against benchmark-declared action contracts."""

    _ALIASES: ClassVar[dict[str, str]] = {
        "qc_filter": "qc",
        "quality_control": "qc",
        "select_hvg": "select_hvg",
        "batch_correct": "harmony",
        "differential_expression": "differential-expression",
        "leiden": "cluster",
        "louvain": "cluster",
        "clustering": "cluster",
        "marker_based": "annotate",
        "annotation": "annotate",
    }

    def __init__(self, validator: DeclarativeActionValidator | None = None) -> None:
        self.validator = validator or DeclarativeActionValidator()

    def to_action_intent(
        self,
        decision: ScientificDecision,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
    ) -> ActionIntent:
        """Translate and validate one structured scientific decision."""
        declared = {action.id: action for action in specification.actions}
        method_candidates = [
            decision.method,
            self._ALIASES.get(decision.method or ""),
        ]
        method_action = next(
            (
                candidate
                for candidate in method_candidates
                if candidate is not None
                and candidate in declared
                and candidate in task.allowed_actions
            ),
            None,
        )
        if decision.method is not None and method_action is None:
            raise ActionMappingError(
                f"decision selects an unknown or disallowed method '{decision.method}'"
            )
        candidates = [
            method_action,
            decision.action_category,
            self._ALIASES.get(decision.action_category),
        ]
        action_id = next(
            (
                candidate
                for candidate in candidates
                if candidate is not None
                and candidate in declared
                and candidate in task.allowed_actions
            ),
            None,
        )
        if action_id is None:
            raise ActionMappingError(
                f"decision does not select an allowed declared action: "
                f"method={decision.method!r}, category={decision.action_category!r}"
            )
        intent = ActionIntent(
            action_id=action_id,
            parameters=decision.parameters,
            rationale=decision.rationale,
            metadata={
                **decision.metadata,
                "decision_type": decision.decision_type,
                "method": decision.method or action_id,
                "method_id": decision.method or action_id,
                "alternatives_considered": decision.alternatives_considered,
                "decision_category": decision.decision_category.value,
                "intent": decision.intent,
                "hypothesis": decision.hypothesis,
                "evidence_used": decision.evidence_used,
                "confidence": decision.confidence,
                "expected_effect": decision.expected_effect,
                "downstream_dependency": decision.downstream_dependency,
                "expected_outputs": declared[action_id].expected_outputs,
                "input_artifacts": decision.input_artifacts,
            },
        )
        validation = self.validator.validate(intent, specification, task, snapshot)
        if not validation.valid:
            raise ActionMappingError("; ".join(validation.errors))
        return intent

    def validate(
        self,
        intent: ActionIntent,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
    ) -> ActionValidationResult:
        """Expose the same validation contract for callers that already have intents."""
        return self.validator.validate(intent, specification, task, snapshot)


__all__ = ["ActionMappingError", "ScientificActionMapper"]
