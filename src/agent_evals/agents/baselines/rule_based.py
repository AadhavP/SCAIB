"""Deterministic rule-based single-cell analysis baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
from agent_evals.scientific.operations.normalize import LIBRARY_SIZE_LOG1P

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


@dataclass(frozen=True)
class _Rule:
    """One pipeline stage this policy can take, and the conditions under which.

    ``parameters`` is a callable rather than a mapping so that every rule builds a
    fresh dict and none of them shares state with the table -- and because
    ``harmony`` has to read the batch column out of the observation, which a
    literal cannot. A rule that needed a special case would make the table lie
    about being one.
    """

    action_id: str
    #: The ``pipeline_state`` flag that means this stage already ran. Set, the rule
    #: is skipped and the next one is considered.
    completed_flag: str
    method: str
    decision_category: DecisionCategory
    intent: str
    rationale: str
    parameters: Callable[[ScientificObservation], dict[str, Any]]
    #: Flags that must already be true for this stage to be meaningful. Unlike
    #: ``completed_flag``, an unmet entry *stops* the policy rather than falling
    #: through: the stage is next in the pipeline and its input does not exist yet,
    #: so taking a later stage instead would reorder the analysis.
    requires: tuple[str, ...] = ()
    #: The coarse stage an action belongs to, when that differs from the action
    #: itself. Only ``qc`` reports one; every other action is its own category.
    action_category: str | None = None


#: The pipeline, in order, as the policy's single source of what comes next.
#:
#: An if/elif chain expressed the same thing, but the order -- the one property a
#: reader of a baseline policy actually needs -- could only be recovered by tracing
#: control flow past nine branch conditions and two nested guards. Every entry has
#: the same shape, so the shape belongs in a type and the order in a sequence.
#:
#: ``marker-genes`` and ``differential-expression`` are separate entries for the
#: same pipeline stage because the two catalogs that declare them ask for different
#: parameters, and a parameter a catalog does not declare is rejected outright.
_RULES: tuple[_Rule, ...] = (
    _Rule(
        action_id="qc",
        completed_flag="qc_complete",
        method="qc_filter",
        decision_category=DecisionCategory.QC_STRATEGY,
        action_category="quality_control",
        intent="remove low-quality cells before downstream analysis",
        rationale=(
            "Run quality control before normalization so downstream comparisons "
            "use retained cells."
        ),
        parameters=lambda _: {"min_genes": 200, "max_mito_fraction": 0.2},
    ),
    _Rule(
        action_id="normalize",
        completed_flag="normalized",
        method="normalize",
        decision_category=DecisionCategory.NORMALIZATION,
        intent="put retained cells on a comparable library-size scale",
        rationale=(
            "Normalize library size before representation learning or marker analysis."
        ),
        # ``method`` is named rather than left to the default so the choice is
        # recorded as one this policy made. All three catalogs declare the same two
        # choices, so one parameter set validates against every one of them.
        parameters=lambda _: {"method": LIBRARY_SIZE_LOG1P, "target_sum": 10_000},
    ),
    _Rule(
        action_id="pca",
        completed_flag="pca_complete",
        method="pca",
        decision_category=DecisionCategory.DIMENSIONALITY_REDUCTION,
        intent="construct a compact representation for neighborhood analysis",
        rationale=(
            "Construct a compact representation before integration or neighborhood "
            "analysis."
        ),
        parameters=lambda _: {"n_components": 20},
    ),
    _Rule(
        action_id="harmony",
        completed_flag="batch_corrected",
        method="harmony",
        decision_category=DecisionCategory.INTEGRATION,
        intent="reduce technical batch variation while preserving biology",
        rationale=(
            "Correct the observed batch structure while preserving the biological "
            "labels."
        ),
        parameters=lambda observation: {
            "batch_key": observation.batch_information.get("label_key") or "batch"
        },
    ),
    _Rule(
        action_id="cluster",
        completed_flag="clustered",
        method="leiden",
        decision_category=DecisionCategory.CLUSTERING,
        intent="recover candidate cell populations without using reference labels",
        rationale=(
            "Group cells without reference labels so annotation has agent-produced "
            "groups."
        ),
        parameters=lambda _: {"resolution": 1.0, "n_neighbors": 15},
    ),
    _Rule(
        action_id="marker-genes",
        completed_flag="differential_expression_complete",
        method="marker-genes",
        decision_category=DecisionCategory.DIFFERENTIAL_EXPRESSION,
        intent="generate marker evidence for biological interpretation",
        rationale="Rank marker genes for the agent-produced clusters.",
        parameters=lambda _: {"group_key": CLUSTER_COLUMN},
        requires=("clustered",),
    ),
    _Rule(
        action_id="differential-expression",
        completed_flag="differential_expression_complete",
        method="differential-expression",
        decision_category=DecisionCategory.DIFFERENTIAL_EXPRESSION,
        intent="quantify which genes distinguish each recovered population",
        rationale=(
            "Test each agent-produced population against the rest with a rank "
            "test, which assumes no distribution the data has to earn."
        ),
        parameters=lambda _: {"method": "wilcoxon", "group_key": CLUSTER_COLUMN},
        requires=("clustered",),
    ),
    _Rule(
        action_id="annotate",
        completed_flag="annotated",
        method="marker_based",
        decision_category=DecisionCategory.ANNOTATION,
        intent="assign a cell-type label to every retained cell from marker evidence",
        rationale="Label each cluster with the canonical PBMC panel it most expresses.",
        parameters=lambda _: {
            "label_vocabulary": sorted(PBMC_MARKER_PANEL),
            "markers": {
                label: list(genes) for label, genes in PBMC_MARKER_PANEL.items()
            },
            "group_key": CLUSTER_COLUMN,
        },
    ),
    # Last, and last for a reason: a report describes the analysis that finished, so
    # every stage that could still add to it belongs above this line. No example
    # catalog offers ``report`` and ``annotate`` together, so the order is currently
    # unobservable -- which is exactly why it is worth getting right here rather than
    # discovering it as a report of unannotated clusters in the first one that does.
    _Rule(
        action_id="report",
        completed_flag="reported",
        method="report",
        decision_category=DecisionCategory.INTERPRETATION,
        intent="present the ranked evidence in a form a reader can check",
        rationale=(
            "Summarize the ranked results this run produced as a readable document."
        ),
        parameters=lambda _: {"top_n": 50},
    ),
)


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
        rule = self._next_rule(observation)
        if rule is None:
            return None
        return ScientificDecision(
            decision_id=f"decision-{order + 1}",
            episode_id=episode_id,
            step_id=f"step-{order + 1}",
            order=order,
            decision_type="method_selection",
            action_category=rule.action_category or rule.action_id,
            decision_category=rule.decision_category,
            intent=rule.intent,
            hypothesis="the declared upstream transformation will improve downstream scientific evidence",
            method=rule.method,
            parameters=rule.parameters(observation),
            rationale=rule.rationale,
            evidence_used=["scientific-observation"],
            confidence=0.7,
            expected_effect={"downstream_quality": 0.7},
            alternatives_considered=list(observation.available_actions),
            timestamp=datetime.now(UTC),
            selected_value=rule.action_id,
            metadata={
                "policy": "deterministic-rule-based",
                "rule_action": rule.action_id,
            },
        )

    @staticmethod
    def _next_rule(observation: ScientificObservation) -> _Rule | None:
        """Find the first stage that is offered, unfinished, and ready to run.

        An unmet ``requires`` returns ``None`` rather than continuing the scan, and
        that is the whole reason this is a separate function: it is the behaviour the
        original if/elif chain had by accident of syntax -- an inner guard that
        failed left the chain with nothing to do, and ``elif`` meant no later branch
        was ever tested. Preserved deliberately, because the alternative reorders the
        analysis. Asked to rank markers before clustering exists, the policy would
        otherwise skip forward and report on results it has not computed.
        """
        available = observation.available_actions
        state = observation.pipeline_state
        for rule in _RULES:
            if rule.action_id not in available or state.get(rule.completed_flag, False):
                continue
            if not all(state.get(flag, False) for flag in rule.requires):
                return None
            return rule
        return None

    @staticmethod
    def _scientific_observation(snapshot: Any) -> ScientificObservation:
        value = snapshot.state.observations.get("scientific-observation")
        if value is None:
            raise RuntimeError("scientific observation is missing from environment state")
        return ScientificObservation.model_validate(value.value)


__all__ = ["BASELINE_MAX_STEPS", "RuleBasedSingleCellAgent", "baseline_budget"]
