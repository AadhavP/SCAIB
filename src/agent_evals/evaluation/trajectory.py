"""Deterministic trajectory-quality evaluation over normalized episode state."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from agent_evals.agents.trajectory import AgentRun
from agent_evals.benchmarks.schema import TaskSpecification
from agent_evals.environment.models import ActionRecord, ActionStatus, EventType
from agent_evals.evaluation.metrics.trajectory import (
    decision_regret,
    method_exploration_score,
)
from agent_evals.evaluation.models import TrajectoryEvaluation


class TrajectoryEvaluator:
    """Compute transparent protocol, dependency, artifact, and efficiency scores."""

    def evaluate(
        self,
        run: AgentRun,
        task: TaskSpecification,
        outcome: float | None = None,
        local_rewards: Sequence[float] | None = None,
        alternative_methods: Mapping[str, Sequence[str]] | None = None,
        alternative_scores: Mapping[str, float] | None = None,
    ) -> TrajectoryEvaluation:
        """Evaluate only persisted events, actions, artifacts, and resources."""
        actions = run.final_environment_state.state.actions
        events = run.final_environment_state.events
        action_count = len(actions)
        proposed = sum(
            event.event_type in {EventType.ACTION_SUBMITTED, EventType.ACTION_PROPOSED}
            for event in events
        )
        rejected = sum(event.event_type == EventType.ACTION_REJECTED for event in events)
        failed = sum(record.result.status == ActionStatus.FAILED for record in actions)
        action_ids = [record.intent.action_id for record in actions]
        counts = Counter(action_ids)
        duplicate_steps = sum(max(0, count - 1) for count in counts.values())
        failed_ids = [record.intent.action_id for record in actions if record.result.status == ActionStatus.FAILED]
        failed_retries = sum(max(0, counts[item] - 1) for item in set(failed_ids))
        artifacts = [artifact for record in actions for artifact in record.result.artifacts]
        artifact_validity = (
            sum(bool(artifact.validated) for artifact in artifacts) / len(artifacts)
            if artifacts
            else 1.0
        )
        dependency_score = self._dependency_score(actions, task)
        consistency_score, contradictions = self._consistency_score(actions)
        consistency_score = (consistency_score + dependency_score) / 2
        adaptation_score = self._adaptation_score(run, action_count)
        methods_attempted = [
            str(record.intent.metadata.get("method", record.intent.action_id))
            for record in actions
        ]
        exploration_score = method_exploration_score(
            methods_attempted,
            alternative_methods,
            duplicate_steps,
        )
        declared_alternatives = {
            method
            for methods in (alternative_methods or {}).values()
            for method in methods
        }
        alternative_coverage = (
            len(set(methods_attempted).intersection(declared_alternatives))
            / len(declared_alternatives)
            if declared_alternatives
            else (1.0 if methods_attempted else 0.0)
        )
        regret = decision_regret(outcome or 0.0, alternative_scores)
        invalid_rate = rejected / max(1, proposed)
        failed_rate = failed / max(1, action_count)
        protocol = max(0.0, 1.0 - invalid_rate)
        efficiency = max(0.0, 1.0 - (duplicate_steps + failed_retries) / max(1, action_count))
        alignment = max(0.0, min(1.0, outcome if outcome is not None else 0.0))
        short_term_gain, long_term_damage = self._counterproductive_signal(local_rewards, outcome)
        counterproductive_score = 1.0 - long_term_damage
        quality = (
            0.25 * protocol
            + 0.15 * consistency_score
            + 0.15 * artifact_validity
            + 0.15 * efficiency
            + 0.10 * adaptation_score
            + 0.10 * alignment
            + 0.10 * exploration_score
        )
        good_signals = []
        bad_signals = []
        recommendations = []
        if protocol == 1.0:
            good_signals.append("all recorded action submissions followed the protocol")
        if consistency_score == 1.0:
            good_signals.append("no contradictory transformations were detected")
        if adaptation_score < 0.5:
            bad_signals.append("few observable observations were followed by a new decision")
            recommendations.append("record and respond to intermediate scientific observations")
        if duplicate_steps:
            bad_signals.append(f"{duplicate_steps} redundant action repetition(s) detected")
            recommendations.append("avoid repeating completed pipeline transformations")
        if contradictions:
            bad_signals.extend(contradictions)
            recommendations.append("check downstream data assumptions before selecting the next method")
        if long_term_damage > 0:
            bad_signals.append("local reward exceeded the final outcome, indicating a counterproductive signal")
            recommendations.append("validate local improvements against downstream scientific outcomes")
        comparisons = [
            {
                "alternative": name,
                "score": score,
                "regret": max(0.0, score - (outcome or 0.0)),
            }
            for name, score in (alternative_scores or {}).items()
        ]
        step_table = [
            {
                "step": record.step,
                "action": record.intent.action_id,
                "status": record.result.status.value,
                "artifacts": [artifact.artifact_id for artifact in record.result.artifacts],
                "resource_usage": record.result.resource_usage.model_dump(mode="json"),
            }
            for record in actions
        ]
        return TrajectoryEvaluation(
            protocol_compliance=protocol,
            invalid_action_rate=invalid_rate,
            failed_action_rate=failed_rate,
            artifact_validity=artifact_validity,
            duplicate_action_rate=duplicate_steps / max(1, action_count),
            step_count=action_count,
            redundant_steps=duplicate_steps,
            runtime_seconds=run.wall_clock_seconds,
            token_usage=run.token_usage.total_tokens if run.token_usage else None,
            resource_usage=run.final_environment_state.state.resource_usage.model_dump(mode="json"),
            failed_retries=failed_retries,
            dependency_consistency=dependency_score,
            outcome_alignment=alignment,
            efficiency=efficiency,
            decision_efficiency=efficiency,
            decision_consistency=consistency_score,
            adaptation_ability=adaptation_score,
            counterproductive_action_detection=counterproductive_score,
            short_term_gain=short_term_gain,
            long_term_damage=long_term_damage,
            good_signals=good_signals,
            bad_signals=bad_signals,
            recommended_improvements=recommendations,
            method_exploration_score=exploration_score,
            alternative_coverage=alternative_coverage,
            unnecessary_retries=duplicate_steps,
            decision_regret=regret,
            alternative_comparisons=comparisons,
            trajectory_quality=quality,
            step_table=step_table,
            formula="0.25*protocol + 0.15*consistency + 0.15*artifact_validity + 0.15*efficiency + 0.10*adaptation + 0.10*outcome_alignment + 0.10*method_exploration",
        )

    @staticmethod
    def _dependency_score(actions: list[ActionRecord], task: TaskSpecification) -> float:
        known = set(task.observations)
        valid = 0
        total = 0
        for record in actions:
            required = [
                str(item)
                for item in record.intent.metadata.get("input_artifacts", [])
            ]
            total += 1
            if set(required).issubset(known):
                valid += 1
            known.update(artifact.artifact_id for artifact in record.result.artifacts)
        return valid / total if total else 1.0

    @staticmethod
    def _consistency_score(actions: list[ActionRecord]) -> tuple[float, list[str]]:
        """Detect explicit incompatible data-mode choices in the trace."""
        normalized_seen = False
        contradictions: list[str] = []
        for record in actions:
            if record.intent.action_id == "normalize":
                normalized_seen = True
            data_mode = str(record.intent.parameters.get("data_mode", "")).lower()
            method = str(record.intent.metadata.get("method", "")).lower()
            if normalized_seen and (data_mode in {"raw", "raw_counts", "counts"} or "raw-count" in method):
                contradictions.append("raw-count assumptions appeared after normalization")
        return (0.0 if contradictions else 1.0), contradictions

    @staticmethod
    def _adaptation_score(run: AgentRun, action_count: int) -> float:
        """Measure whether observable environment updates precede decisions."""
        observation_events = sum(
            event.event_type in {EventType.OBSERVATION_RECEIVED, EventType.OBSERVATIONS_UPDATED}
            for event in run.trajectory.events
        )
        return min(1.0, observation_events / max(1, action_count))

    @staticmethod
    def _counterproductive_signal(
        local_rewards: Sequence[float] | None,
        outcome: float | None,
    ) -> tuple[float, float]:
        """Report an observable local/global gap without assigning causality."""
        if not local_rewards or outcome is None:
            return 0.0, 0.0
        local_peak = max(max(0.0, min(1.0, float(value))) for value in local_rewards)
        damage = max(0.0, min(1.0, local_peak - outcome))
        return local_peak, damage


__all__ = ["TrajectoryEvaluator"]
