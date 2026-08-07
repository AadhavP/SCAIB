"""Deterministic evaluation of observable scientific decisions."""

from __future__ import annotations

from agent_evals.agents.trajectory import AgentRun
from agent_evals.benchmarks.schema import TaskSpecification
from agent_evals.environment.models import ActionStatus, EventType
from agent_evals.evaluation.models import DecisionEvaluation


class DecisionEvaluator:
    """Score action choices from the recorded episode, without causal claims."""

    def evaluate(self, run: AgentRun, task: TaskSpecification) -> list[DecisionEvaluation]:
        """Return one evaluation for every submitted action."""
        rewards = {reward.step: reward.value for reward in run.final_environment_state.state.rewards}
        allowed = set(task.allowed_actions)
        cascade = {
            decision.step_id: decision
            for decision in run.trajectory.decisions.decisions
            if decision.parent_decision_id is None
        }
        evaluations: list[DecisionEvaluation] = []
        for record in run.final_environment_state.state.actions:
            valid = record.intent.action_id in allowed
            execution_succeeded = record.result.status == ActionStatus.SUCCEEDED
            artifacts = [artifact.artifact_id for artifact in record.result.artifacts]
            structured = cascade.get(f"step-{record.step}")
            score = (float(valid) + float(execution_succeeded)) / 2
            evaluations.append(
                DecisionEvaluation(
                    decision_id=f"decision-{record.step}",
                    step_index=record.step,
                    action_category=record.intent.action_id,
                    decision_category=(structured.decision_category.value if structured else "other"),
                    intent=structured.intent if structured else None,
                    hypothesis=structured.hypothesis if structured else None,
                    method=(
                        str(record.intent.metadata["method"])
                        if record.intent.metadata.get("method") is not None
                        else None
                    ),
                    parameters=record.intent.parameters,
                    evidence_used=structured.evidence_used if structured else [],
                    confidence=structured.confidence if structured else None,
                    expected_effect=structured.expected_effect if structured else {},
                    downstream_dependency=structured.downstream_dependency if structured else {},
                    valid=valid,
                    scientific_applicable=valid,
                    execution_succeeded=execution_succeeded,
                    produced_artifacts=artifacts,
                    consumed_artifacts=[
                        str(item) for item in record.intent.metadata.get("input_artifacts", [])
                    ],
                    local_reward=rewards.get(record.step),
                    score=score,
                    reason=(
                        "valid action executed successfully"
                        if score == 1
                        else "action was valid but execution did not succeed"
                        if valid
                        else "action was not permitted by the task"
                    ),
                )
            )
        proposed = sum(
            1
            for event in run.final_environment_state.events
            if event.event_type in {EventType.ACTION_SUBMITTED, EventType.ACTION_PROPOSED}
        )
        rejected = sum(
            1
            for event in run.final_environment_state.events
            if event.event_type == EventType.ACTION_REJECTED
        )
        if rejected and proposed == 0:
            evaluations.append(
                DecisionEvaluation(
                    decision_id="rejected-actions",
                    step_index=0,
                    action_category="rejected",
                    method=None,
                    parameters={},
                    valid=False,
                    scientific_applicable=False,
                    execution_succeeded=False,
                    score=0.0,
                    reason=f"{rejected} action proposal(s) were rejected before execution",
                )
            )
        return evaluations


__all__ = ["DecisionEvaluator"]
