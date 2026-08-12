"""Deterministic method-level evaluation for scientific action traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_evals.agents.trajectory import AgentRun, ScientificDecision
from agent_evals.benchmarks.schema import TaskSpecification
from agent_evals.environment.models import ActionStatus
from agent_evals.evaluation.models import MethodEvaluation, MethodScore
from agent_evals.evaluation.taxonomy import DecisionOntology, DecisionProfile
from agent_evals.metrics.results import MetricResult


class MethodEvaluator:
    """Evaluate declared methods as observed execution choices."""

    def evaluate(
        self,
        run: AgentRun,
        task: TaskSpecification,
        metric_ids: list[str],
        eventual_global_outcome: float | None,
    ) -> list[MethodEvaluation]:
        """Return method evidence without inferring why an outcome occurred."""
        allowed = set(task.allowed_actions)
        rewards = {reward.step: reward.value for reward in run.final_environment_state.state.rewards}
        evaluations: list[MethodEvaluation] = []
        for record in run.final_environment_state.state.actions:
            schema_valid = record.intent.action_id in allowed
            succeeded = record.result.status == ActionStatus.SUCCEEDED
            applicable = schema_valid
            score = (float(schema_valid) + float(applicable) + float(succeeded)) / 3
            evaluations.append(
                MethodEvaluation(
                    decision_id=f"decision-{record.step}",
                    method=(
                        str(record.intent.metadata["method"])
                        if record.intent.metadata.get("method") is not None
                        else record.intent.action_id
                    ),
                    schema_valid=schema_valid,
                    scientifically_applicable=applicable,
                    execution_succeeded=succeeded,
                    produced_artifacts=[artifact.artifact_id for artifact in record.result.artifacts],
                    downstream_metric_ids=list(metric_ids),
                    local_objective=rewards.get(record.step),
                    eventual_global_outcome=eventual_global_outcome,
                    score=score,
                    reason="method was declared, applicable, and executed" if score == 1 else "method evidence was incomplete",
                )
            )
        return evaluations


def method_score(evaluations: list[MethodEvaluation]) -> float:
    """Average method quality, returning a neutral score for an empty trace."""
    return sum(item.score for item in evaluations) / len(evaluations) if evaluations else 1.0


class MethodSelectionEvaluator:
    """Score method appropriateness, parameters, and observed execution."""

    def __init__(self, ontology: DecisionOntology | None = None) -> None:
        self.ontology = ontology

    def evaluate(
        self,
        decision: ScientificDecision,
        dataset_metadata: Mapping[str, Any] | None,
        task_profile: DecisionProfile | None,
        final_results: Sequence[MetricResult] | Mapping[str, float] | None,
    ) -> MethodScore:
        """Return deterministic method evidence without inferring private intent."""
        del dataset_metadata
        profile = task_profile
        if profile is None and self.ontology is not None:
            profile = self.ontology.get(decision.decision_category)
        allowed = profile.allowed_methods if profile is not None else []
        appropriateness: float | None
        if not allowed:
            appropriateness = None
            evidence = [
                "no benchmark method restriction was declared, so method "
                "appropriateness is unmeasured rather than neutral"
            ]
        elif decision.chosen_method is None:
            appropriateness = None
            evidence = [
                "the decision named no method, so appropriateness against the "
                "declared list is unmeasured"
            ]
        elif decision.chosen_method in allowed:
            appropriateness = 1.0
            evidence = [f"method '{decision.chosen_method}' is declared for the decision category"]
        else:
            appropriateness = 0.0
            evidence = [f"method '{decision.chosen_method}' is outside the declared allowed methods"]
        parameter_quality = self._parameter_quality(decision, profile, evidence)
        execution_quality = self._execution_quality(decision, final_results, evidence)
        components = {
            "appropriateness": appropriateness,
            "parameter_quality": parameter_quality,
            "execution_quality": execution_quality,
        }
        measured = [value for value in components.values() if value is not None]
        return MethodScore(
            decision_id=decision.decision_id,
            method=decision.chosen_method,
            appropriateness=appropriateness,
            parameter_quality=parameter_quality,
            execution_quality=execution_quality,
            # The mean of what was measured, not of three slots two of which may
            # hold placeholders. An unanswerable component is dropped and the rest
            # re-weighted, the same rule the trajectory and local-reward
            # evaluators follow.
            overall=sum(measured) / len(measured) if measured else None,
            unmeasured_components=[
                name for name, value in components.items() if value is None
            ],
            evidence=evidence,
        )

    @staticmethod
    def _parameter_quality(
        decision: ScientificDecision,
        profile: DecisionProfile | None,
        evidence: list[str],
    ) -> float | None:
        """Score declared parameters, or report that none were declared.

        Returning 1.0 for a category with no declared ranges was the worst of the
        substitutions: it paid a full third of the selection score for a question
        the benchmark never asked, and it paid it to every agent equally, so the
        component carried no signal while still carrying weight.
        """
        if profile is None or not profile.parameter_ranges:
            evidence.append(
                "no parameter ranges were declared for this category, so "
                "parameter quality is unmeasured"
            )
            return None
        scores: list[float] = []
        for name, bounds in profile.parameter_ranges.items():
            value = decision.chosen_parameters.get(name)
            if value is None:
                scores.append(0.0)
                evidence.append(f"parameter '{name}' was not supplied")
                continue
            choices = bounds.get("choices", [])
            if choices:
                scores.append(1.0 if value in choices else 0.0)
                continue
            minimum = bounds.get("minimum")
            maximum = bounds.get("maximum")
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and (minimum is None or value >= minimum)
                and (maximum is None or value <= maximum)
            )
            scores.append(1.0 if valid else 0.0)
        return sum(scores) / len(scores) if scores else None

    @staticmethod
    def _execution_quality(
        decision: ScientificDecision,
        final_results: Sequence[MetricResult] | Mapping[str, float] | None,
        evidence: list[str],
    ) -> float | None:
        """Score the observed consequence, or report that nothing was scoreable.

        Empty metric results used to yield ``0.0`` -- the mirror of the other two
        substitutions, and just as wrong. A step taken before any metric could be
        answered was recorded as having executed badly rather than as not yet
        having been assessed.
        """
        if final_results is None:
            return 1.0 if decision.execution_status == ActionStatus.SUCCEEDED else 0.0
        if isinstance(final_results, Mapping):
            values = [float(value) for value in final_results.values()]
        else:
            values = [
                float(result.normalized_value)
                for result in final_results
                if result.normalized_value is not None
            ]
        if not values:
            evidence.append(
                "no metric produced a value against this run, so execution "
                "quality is unmeasured"
            )
            return None
        return sum(values) / len(values)


__all__ = ["MethodEvaluator", "MethodSelectionEvaluator", "method_score"]
