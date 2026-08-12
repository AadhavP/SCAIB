"""Deterministic rule-based single-cell analysis baseline."""

from __future__ import annotations

from datetime import UTC, datetime
from time import monotonic
from typing import Any
from uuid import uuid4

from agent_evals.agents.harness import build_agent_run
from agent_evals.agents.runtime.manager import (
    cutoff_termination,
    decision_signature,
    progress_delta,
)
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
from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
from agent_evals.environment.cutoff import (
    CutoffBudget,
    CutoffController,
    StepObservation,
    budget_from_specification,
)
from agent_evals.environment.models import ActionStatus, EpisodeStatus
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.scientific.action_mapper import (
    ActionMappingError,
    ScientificActionMapper,
)
from agent_evals.scientific.observations import ScientificObservation
from agent_evals.scientific.operations.cluster import CLUSTER_COLUMN

#: Canonical PBMC marker genes per cell type. The baseline states its evidence
#: explicitly so its annotation score reflects this panel, not the answer key.
PBMC_MARKER_PANEL: dict[str, tuple[str, ...]] = {
    "CD4 T": ("IL7R", "CD3D", "CD3E", "CCR7"),
    "CD8 T": ("CD8A", "CD8B", "CD3D", "GZMK"),
    "B": ("MS4A1", "CD79A", "CD79B", "CD19"),
    "NK": ("GNLY", "NKG7", "KLRD1", "NCAM1"),
    "CD14 Monocyte": ("CD14", "LYZ", "S100A9", "VCAN"),
    "FCGR3A Monocyte": ("FCGR3A", "MS4A7", "CDKN1C"),
    "Dendritic": ("FCER1A", "CST3", "CLEC10A"),
    "Megakaryocyte": ("PPBP", "PF4", "ITGA2B"),
}


#: Step ceiling applied only when neither the benchmark nor the caller declares
#: one. Preserves the bound this baseline has always had, so a benchmark with no
#: ``cutoff`` block stops it exactly where it used to. It is a fallback rather
#: than a caller limit on purpose: passed as one it would be *stricter* than a
#: declared ceiling and would silently cap a benchmark asking for 40 steps at 8.
BASELINE_MAX_STEPS = 8


def baseline_budget(
    specification: BenchmarkSpecification, *, caller_max_steps: int | None = None
) -> CutoffBudget:
    """Resolve this baseline's cutoff budget, including its own step fallback.

    A module-level function rather than four lines inside ``run`` so the fallback
    rule can be asserted on its own. Wired inline, the only way to reach it would
    be an end-to-end run, and the one thing that has to hold -- that the fallback
    never overrides a declared ceiling -- is exactly what an end-to-end run against
    a benchmark that declares one cannot show.
    """
    budget = budget_from_specification(
        specification.cutoff,
        specification.constraints,
        caller_max_steps=caller_max_steps,
    )
    if budget.max_steps is None:
        return budget.model_copy(update={"max_steps": BASELINE_MAX_STEPS})
    return budget


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
        # This baseline drives its own episode loop, so it needs its own
        # controller: without one, the benchmark's declared cutoff block governs
        # every agent *except* the reference agent the paper reports against.
        controller = CutoffController(
            baseline_budget(
                environment.specification,
                caller_max_steps=configuration.max_steps,
            )
        )
        run_origin = monotonic()
        cutoff_status: RunTerminationStatus | None = None
        cutoff_reason: str | None = None
        step = -1
        while True:
            cutoff = controller.decide(elapsed_seconds=monotonic() - run_origin)
            if cutoff.stop:
                termination = cutoff_termination(cutoff, environment)
                cutoff_status = termination.status
                cutoff_reason = termination.reason
                if termination.failure_kind is not None:
                    failures.append(
                        AgentFailure(
                            kind=termination.failure_kind, message=termination.reason
                        )
                    )
                raw_events.append(
                    _event(
                        len(raw_events),
                        "run_cutoff",
                        {"cutoff": cutoff.model_dump(mode="json"), "reason": termination.reason},
                    )
                )
                break
            step += 1
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
            step_succeeded = (
                outcome.accepted
                and outcome.execution is not None
                and outcome.execution.status == ActionStatus.SUCCEEDED
            )
            controller.observe(
                StepObservation(
                    step=step + 1,
                    succeeded=step_succeeded,
                    signature=decision_signature(intent.action_id, intent.parameters),
                    progress_delta=progress_delta(outcome),
                )
            )
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
            if not step_succeeded:
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
        # A cutoff status wins over the failure default because it is the more
        # specific reading of the same stop: the controller ended the run and
        # already said whether the artifact contract was met. Collapsing that to
        # ``FAILED`` would report a run cut off for looping as a crash, and one
        # that finished its contract and then hit the step ceiling as a failure.
        if cutoff_status is not None:
            status = cutoff_status
            reason = cutoff_reason or "the run was stopped by its cutoff budget"
        elif failures:
            status, reason = RunTerminationStatus.FAILED, failures[0].message
        else:
            status, reason = (
                RunTerminationStatus.COMPLETED,
                "rule-based trajectory complete",
            )
        final_snapshot = environment.terminate(
            status=EpisodeStatus.COMPLETED
            if status is RunTerminationStatus.COMPLETED
            else EpisodeStatus.FAILED,
            reason=reason,
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
            termination_reason=reason,
            failures=failures,
            metadata={
                "policy": "deterministic-rule-based",
                # Archived for the same reason the universal loop archives it: a
                # budget computed correctly every run and read by nobody cannot be
                # audited, and this is the only conversion into the stored run.
                "cutoff": controller.report().model_dump(mode="json"),
            },
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
        elif "cluster" in available and not state.get("clustered", False):
            action_id, method = "cluster", "leiden"
            parameters = {"resolution": 1.0, "n_neighbors": 15}
            rationale = "Group cells without reference labels so annotation has agent-produced groups."
        elif "marker-genes" in available and not state.get(
            "differential_expression_complete", False
        ):
            if state.get("clustered", False):
                action_id, method = "marker-genes", "marker-genes"
                parameters = {"group_key": CLUSTER_COLUMN}
                rationale = "Rank marker genes for the agent-produced clusters."
        elif "annotate" in available and not state.get("annotated", False):
            action_id, method = "annotate", "marker_based"
            parameters = {
                "label_vocabulary": sorted(PBMC_MARKER_PANEL),
                "markers": {label: list(genes) for label, genes in PBMC_MARKER_PANEL.items()},
                "group_key": CLUSTER_COLUMN,
            }
            rationale = "Label each cluster with the canonical PBMC panel it most expresses."
        if action_id is None:
            return None
        category_map = {
            "qc": DecisionCategory.QC_STRATEGY,
            "normalize": DecisionCategory.NORMALIZATION,
            "pca": DecisionCategory.DIMENSIONALITY_REDUCTION,
            "harmony": DecisionCategory.INTEGRATION,
            "cluster": DecisionCategory.CLUSTERING,
            "marker-genes": DecisionCategory.DIFFERENTIAL_EXPRESSION,
            "annotate": DecisionCategory.ANNOTATION,
        }
        intent_map = {
            "qc": "remove low-quality cells before downstream analysis",
            "normalize": "put retained cells on a comparable library-size scale",
            "pca": "construct a compact representation for neighborhood analysis",
            "harmony": "reduce technical batch variation while preserving biology",
            "cluster": "recover candidate cell populations without using reference labels",
            "marker-genes": "generate marker evidence for biological interpretation",
            "annotate": "assign a cell-type label to every retained cell from marker evidence",
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


__all__ = ["BASELINE_MAX_STEPS", "RuleBasedSingleCellAgent", "baseline_budget"]
