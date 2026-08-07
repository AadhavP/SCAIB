"""Deterministic rule-based single-cell analysis baseline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from agent_evals.agents.harness import build_agent_run
from agent_evals.agents.trajectory import (
    AgentConfiguration,
    AgentFailure,
    AgentRun,
    DecisionCategory,
    FailureKind,
    RawTraceEvent,
    RunTerminationStatus,
    ScientificDecision,
)
from agent_evals.benchmarks.schema import TaskSpecification
from agent_evals.environment.models import ActionStatus, EpisodeStatus
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.scientific.action_mapper import (
    ActionMappingError,
    ScientificActionMapper,
)
from agent_evals.scientific.observations import ScientificObservation


def _event(sequence: int, event_type: str, payload: dict[str, Any]) -> RawTraceEvent:
    return RawTraceEvent(
        event_id=str(uuid4()),
        source="rule-based",
        sequence=sequence,
        timestamp=datetime.now(UTC),
        event_type=event_type,
        payload=payload,
    )


class RuleBasedSingleCellAgent:
    """A deterministic policy that follows defensible preprocessing rules."""

    adapter_name = "rule-based"
    adapter_version = "1.0.0"

    def __init__(self) -> None:
        self.decision_trace: list[dict[str, Any]] = []

    async def run(
        self,
        task: TaskSpecification,
        environment: ScientificEnvironment,
        configuration: AgentConfiguration,
    ) -> AgentRun:
        """Observe, select a declared method, execute it, and repeat."""
        started_at = datetime.now(UTC)
        raw_events: list[RawTraceEvent] = []
        failures: list[AgentFailure] = []
        self.decision_trace = []
        snapshot = await environment.reset(
            seed=configuration.seed,
            dataset_id=configuration.metadata.get("dataset_id")
            or (task.datasets[0] if task.datasets else None),
        )
        initial_observation = snapshot.state.observations.get("scientific-observation")
        raw_events.append(
            _event(
                0,
                "observation",
                {
                    "step": 0,
                    "observation": initial_observation.value
                    if initial_observation is not None
                    else {},
                },
            )
        )
        mapper = ScientificActionMapper()
        max_steps = configuration.max_steps or 8
        for step in range(max_steps):
            scientific = self._scientific_observation(snapshot)
            decision = self.choose(scientific, task, snapshot.state.episode_id, step)
            if decision is None:
                break
            raw_events.append(
                _event(
                    len(raw_events),
                    "action_proposed",
                    {
                        "decision": decision.model_dump(mode="json"),
                        "observation": scientific.model_dump(mode="json"),
                    },
                )
            )
            try:
                intent = mapper.to_action_intent(
                    decision,
                    environment.specification,
                    task,
                    snapshot,
                )
            except ActionMappingError as error:
                failures.append(AgentFailure(kind=FailureKind.INVALID_ACTION, message=str(error)))
                self.decision_trace.append(
                    {"observation": scientific.model_dump(mode="json"), "decision": decision.model_dump(mode="json"), "error": str(error)}
                )
                break
            outcome = await environment.step(intent)
            snapshot = outcome.observation
            reward = outcome.reward
            self.decision_trace.append(
                {
                    "observation": scientific.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                    "intent": intent.model_dump(mode="json"),
                    "result": outcome.execution.model_dump(mode="json") if outcome.execution else None,
                    "reward": reward.model_dump(mode="json") if reward else None,
                }
            )
            raw_events.append(
                _event(
                    len(raw_events),
                    "tool_result",
                    {
                        "action_id": intent.action_id,
                        "status": outcome.execution.status.value if outcome.execution else "rejected",
                        "reward": reward.model_dump(mode="json") if reward else None,
                    },
                )
            )
            if (
                not outcome.accepted
                or outcome.execution is None
                or outcome.execution.status != ActionStatus.SUCCEEDED
            ):
                message = (
                    "; ".join(outcome.validation.errors)
                    if not outcome.accepted
                    else (
                        outcome.execution.error or "scientific execution failed"
                        if outcome.execution
                        else "missing execution result"
                    )
                )
                failures.append(
                    AgentFailure(
                        kind=FailureKind.INVALID_ACTION if not outcome.accepted else FailureKind.TOOL_ERROR,
                        message=message,
                    )
                )
                break
        status = RunTerminationStatus.FAILED if failures else RunTerminationStatus.COMPLETED
        final_snapshot = environment.terminate(
            status=EpisodeStatus.FAILED if failures else EpisodeStatus.COMPLETED,
            reason=failures[0].message if failures else "rule-based trajectory complete",
        )
        finished_at = datetime.now(UTC)
        return build_agent_run(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            configuration=configuration,
            task=task,
            snapshot=final_snapshot,
            raw_events=raw_events,
            run_id=configuration.metadata.get("run_id"),
            started_at=started_at,
            finished_at=finished_at,
            termination_status=status,
            termination_reason=failures[0].message if failures else "rule-based trajectory complete",
            failures=failures,
            metadata={"policy": "deterministic-rule-based"},
        )

    def choose(
        self,
        observation: ScientificObservation,
        task: TaskSpecification,
        episode_id: str,
        order: int,
    ) -> ScientificDecision | None:
        """Select the next action using only the structured observation."""
        available = observation.available_actions
        state = observation.pipeline_state
        action_id: str | None = None
        parameters: dict[str, Any] = {}
        category = "pipeline"
        method = None
        rationale = "No further supported declared action is available."
        if "qc" in available and not state.get("qc_complete", False):
            action_id, category, method = "qc", "quality_control", "qc_filter"
            parameters = {"min_genes": 200, "max_mito_fraction": 0.2}
            rationale = "Run quality control before normalization so downstream comparisons use retained cells."
        elif "normalize" in available and not state.get("normalized", False):
            action_id, method = "normalize", "normalize"
            parameters = {"target_sum": 10_000}
            rationale = "Normalize library size before representation learning or marker analysis."
        elif "pca" in available and not state.get("pca_complete", False):
            action_id, method = "pca", "pca"
            parameters = {"n_components": 20}
            rationale = "Construct a compact representation before integration or neighborhood analysis."
        elif "harmony" in available and not state.get("batch_corrected", False):
            batch_key = observation.batch_information.get("label_key") or "batch"
            action_id, method = "harmony", "harmony"
            parameters = {"batch_key": batch_key}
            rationale = "Correct the observed batch structure while preserving the biological labels."
        elif "marker-genes" in available and not state.get("differential_expression_complete", False):
            counts = observation.biological_information.get("counts", {})
            if counts and min(int(count) for count in counts.values()) >= 2:
                group_key = observation.biological_information.get("label_key") or "bulk_labels"
                action_id, method = "marker-genes", "marker-genes"
                parameters = {"group_key": group_key}
                rationale = "Generate ranked marker evidence using the observed biological grouping."
        if action_id is None:
            return None
        category_map = {
            "qc": DecisionCategory.QC_STRATEGY,
            "normalize": DecisionCategory.NORMALIZATION,
            "pca": DecisionCategory.DIMENSIONALITY_REDUCTION,
            "harmony": DecisionCategory.INTEGRATION,
            "marker-genes": DecisionCategory.DIFFERENTIAL_EXPRESSION,
        }
        intent_map = {
            "qc": "remove low-quality cells before downstream analysis",
            "normalize": "put retained cells on a comparable library-size scale",
            "pca": "construct a compact representation for neighborhood analysis",
            "harmony": "reduce technical batch variation while preserving biology",
            "marker-genes": "generate marker evidence for biological interpretation",
        }
        return ScientificDecision(
            decision_id=f"decision-{order + 1}",
            episode_id=episode_id,
            step_id=f"step-{order + 1}",
            order=order,
            decision_type="method_selection",
            action_category=category if action_id == "qc" else action_id,
            decision_category=category_map.get(action_id, DecisionCategory.OTHER),
            intent=intent_map.get(action_id),
            hypothesis="the declared upstream transformation will improve downstream scientific evidence",
            method=method,
            parameters=parameters,
            rationale=rationale,
            evidence_used=["scientific-observation"],
            confidence=0.7,
            expected_effect={"downstream_quality": 0.7},
            alternatives_considered=list(available),
            timestamp=datetime.now(UTC),
            selected_value=action_id,
            metadata={"policy": "deterministic-rule-based", "rule_action": action_id},
        )

    @staticmethod
    def _scientific_observation(snapshot: Any) -> ScientificObservation:
        value = snapshot.state.observations.get("scientific-observation")
        if value is None:
            raise RuntimeError("scientific observation is missing from environment state")
        return ScientificObservation.model_validate(value.value)


__all__ = ["RuleBasedSingleCellAgent"]
