"""Agent-driven scientific episode bridge."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.agents import (
    AgentConfiguration,
    AgentHarness,
    AgentRun,
)
from agent_evals.agents.selection import build_agent_adapter, is_universal_runtime
from agent_evals.agents.trajectory import DecisionCategory
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.registry import benchmark_spec_registry
from agent_evals.benchmarks.schema import BenchmarkSpecification
from agent_evals.core.decision_components import (
    OBSERVED_CELL_COUNT,
    OBSERVED_GENE_COUNT,
)
from agent_evals.datasets.preflight import (
    DatasetContractError,
    describe_readiness,
    validate_dataset_contract,
)
from agent_evals.environment.execution import (
    ActionKindRouter,
    WorkspaceObservationBuilder,
)
from agent_evals.environment.models import (
    ActionExecutionResult,
    ActionIntent,
    ArtifactRecord,
    RewardRecord,
)
from agent_evals.environment.ports import (
    CompositeObservationBuilder,
    DeclaredObservationBuilder,
    ExecutionContext,
    ObservationBuilder,
)
from agent_evals.environment.provisioning import (
    provision_environment,
    select_environment,
)
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.evaluation import (
    DecisionEvaluator,
    LocalRewardEvaluator,
    MethodEvaluator,
    MethodSelectionEvaluator,
    ScientificEvaluation,
    ScientificMetricEngine,
    ScoreWeights,
    TrajectoryEvaluator,
    compute_global_agent_score,
    compute_score_confidence,
    describe_score,
)
from agent_evals.evaluation.candidates import (
    build_metric_inputs,
    build_reference_artifacts,
    build_workspace_metric_inputs,
    load_de_table,
)
from agent_evals.evaluation.methods import method_score
from agent_evals.evaluation.metrics.robustness import RobustnessEvaluator
from agent_evals.evaluation.models import MethodScore
from agent_evals.evaluation.profiles import (
    BenchmarkMetricProfile,
    profile_external_scores,
    profile_metric_ids,
    resolve_metric_profile,
)
from agent_evals.evaluation.progress import (
    ProgressSignal,
    ScientificProgressTracker,
    summarize_progress,
)
from agent_evals.evaluation.reference_de import (
    reference_de_metadata,
    scored_group_metadata,
)
from agent_evals.evaluation.scoring import (
    MetricScoreInput,
    WeightedGeometricAggregator,
    aggregate_domains,
    describe_unmeasured_domains,
)
from agent_evals.evaluation.stage_rewards import StageAwareRewardEvaluator
from agent_evals.evaluation.taxonomy import DecisionProfile, decision_ontology
from agent_evals.evaluators.rewards import GlobalReward, RewardEvaluator
from agent_evals.metrics import MetricGroup, MetricWeight
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.results import MetricResult
from agent_evals.scientific.artifacts.storage import LocalArtifactStore
from agent_evals.scientific.artifacts.validation import ArtifactRuleValidator
from agent_evals.scientific.context import ScientificContext
from agent_evals.scientific.executor.scanpy import ScanpyExecutor
from agent_evals.scientific.observations import ScientificObservationBuilder

DEFAULT_RUNTIME_MAX_STEPS = 12


async def _emit_event(
    event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
    event: dict[str, Any],
) -> None:
    """Forward one observable event to an optional sync or async callback."""
    if event_callback is None:
        return
    result = event_callback(event)
    if inspect.isawaitable(result):
        await result


class AgentScientificRun(BaseModel):
    """Persisted result of one agent interacting with the scientific world."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    agent_id: str
    benchmark_id: str
    task_id: str
    episode_id: str
    agent_run: AgentRun
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    local_rewards: list[RewardRecord] = Field(default_factory=list)
    global_reward: GlobalReward
    final_metrics: list[MetricResult] = Field(default_factory=list)
    evaluation: ScientificEvaluation | None = None
    report_path: str | None = None
    #: What the free-execution workspace was and what it failed to guarantee.
    #: ``None`` on the typed tier, where SCAIB runs the science in-process and
    #: there is no workspace to describe. A new optional field rather than a
    #: widened one, so every ``report.json`` written before this existed still
    #: loads against ``extra="forbid"``.
    environment: dict[str, Any] | None = None

    def to_json(self) -> str:
        """Serialize the agent-scientific run."""
        return self.model_dump_json(indent=2)

    def to_markdown(self) -> str:
        """Render an agent-focused benchmark report."""
        local_score = (
            sum(reward.value for reward in self.local_rewards) / len(self.local_rewards)
            if self.local_rewards
            else None
        )
        lines = [
            "# Agent Evaluation Report",
            "",
            f"- Agent: {self.agent_id}",
            f"- Benchmark: {self.benchmark_id}",
            f"- Run: {self.run_id}",
            f"- Episode: {self.episode_id}",
            f"- Agent type: {self.agent_run.manifest.type if self.agent_run.manifest else self.agent_run.adapter_name}",
            f"- Model: {self.agent_run.manifest.model.name if self.agent_run.manifest else self.agent_run.model or 'unspecified'}",
            "",
        ]
        # Printed before the scores, because an unconfined workspace or an
        # unmeasurable outcome dimension changes how every number below should be
        # read, and a reader who reaches the scores first has already formed a
        # view by the time the caveat arrives.
        if self.environment is not None:
            lines.extend(
                [
                    "## Execution environment",
                    "",
                    f"- Environment: `{self.environment.get('environment_id')}` "
                    f"({self.environment.get('backend')})",
                    f"- Agent dataset: `{self.environment.get('agent_dataset')}`",
                    f"- Withheld obs columns: "
                    f"{', '.join(self.environment.get('withheld_obs_columns', [])) or '(none)'}",
                    "",
                    "Not guaranteed by this run:",
                    "",
                    *[
                        f"- {limitation}"
                        for limitation in self.environment.get("limitations", [])
                    ],
                    "",
                ]
            )
        lines.extend(
            [
                "## Decision summary",
                "",
                "| Step | Decision | Method | Execution | Local reward |",
                "| ---: | --- | --- | --- | ---: |",
            ]
        )
        for index, step in enumerate(self.trajectory, start=1):
            decision = step.get("decision", {})
            result = step.get("result") or {}
            reward = step.get("reward") or {}
            lines.append(
                f"| {index} | {decision.get('action_category', '-')} | "
                f"{decision.get('method', '-')} | {result.get('status', 'rejected')} | "
                f"{reward.get('value', '-')} |"
            )
        lines.extend(
            [
                "",
                "## Scores",
                "",
                f"- Local decision score: {local_score if local_score is not None else 'unavailable'}",
                f"- Final pipeline score: {self.global_reward.value if self.global_reward.value is not None else 'unavailable'}",
                "",
                "| Metric | Status | Score |",
                "| --- | --- | ---: |",
            ]
        )
        lines.extend(
            f"| {metric.metric_name} ({metric.metric_id}) | {metric.status.value} | "
            f"{metric.normalized_value if metric.normalized_value is not None else '-'} |"
            for metric in self.final_metrics
        )
        if self.evaluation is not None:
            evaluation = self.evaluation
            lines.extend(
                [
                    "",
                    "## Evaluation dimensions",
                    "",
                    f"- Scientific outcome score: {_unmeasured_or(evaluation.scientific_outcome_score)}",
                    *[
                        # ``unmeasured`` rather than the bare ``None`` this used to
                        # print, which reads as a crash rather than as a gap.
                        f"- {domain.domain.title()} score: "
                        f"{_unmeasured_or(domain.value)}"
                        for domain in evaluation.domain_scores
                    ],
                    f"- Outcome formula: `{evaluation.scientific_outcome_formula}`",
                    f"- Decision score: {_unmeasured_or(evaluation.decision_score)}",
                    f"- Method score: {evaluation.method_score}",
                    f"- Decision quality multiplier: "
                    f"{_unmeasured_or(evaluation.decision_quality_score)}",
                    f"- Trajectory score: {evaluation.trajectory_score}",
                    f"- Final agent score: {_unmeasured_or(evaluation.global_agent_score)}",
                    f"- Global agent score: {_unmeasured_or(evaluation.global_agent_score)}",
                    f"- Score formula: `{evaluation.score_formula}`",
                    "",
                    "### Applicability matrix",
                    "",
                    "| Metric | Eligible | Structural exclusion | Reason |",
                    "| --- | --- | --- | --- |",
                ]
            )
            lines.extend(
                f"| {item.metric_id}@{item.version} | {item.eligible} | "
                f"{item.structurally_ineligible} | {item.reason} |"
                for item in evaluation.applicability
            )
            lines.extend(
                [
                    "",
                    "### Versioned metric results",
                    "",
                    "| Metric | Status | Raw | Normalized |",
                    "| --- | --- | ---: | ---: |",
                ]
            )
            lines.extend(
                f"| {item.metric_id}@{item.version} | {item.status.value} | "
                f"{item.raw_value if item.raw_value is not None else '-'} | "
                f"{item.normalized_value if item.normalized_value is not None else '-'} |"
                for item in evaluation.metric_results
            )
            lines.extend(
                [
                    "",
                    "### Decision timeline",
                    "",
                    "| Decision | Method | Appropriateness | Parameters | Execution | Overall |",
                    "| --- | --- | ---: | ---: | ---: | ---: |",
                ]
            )
            lines.extend(
                f"| {item.decision_id} | {item.method or '-'} | "
                f"{_score_cell(item.appropriateness)} | "
                f"{_score_cell(item.parameter_quality)} | "
                f"{_score_cell(item.execution_quality)} | "
                f"{_score_cell(item.overall)} |"
                for item in evaluation.method_selection_evaluations
            )
            lines.extend(
                [
                    "",
                    "### Local decision rewards",
                    "",
                    "| Decision | Category | Value | Formula |",
                    "| --- | --- | ---: | --- |",
                ]
            )
            lines.extend(
                f"| {item.get('decision_id', '-')} | {item.get('category', '-')} | "
                f"{item.get('value', '-')} | {item.get('formula', '-')} |"
                for item in evaluation.local_decision_rewards
            )
            lines.extend(
                [
                    "",
                    "### Trajectory analysis",
                    "",
                    f"- Method exploration score: {evaluation.trajectory.method_exploration_score}",
                    f"- Alternative coverage: {evaluation.trajectory.alternative_coverage}",
                    f"- Decision regret: {evaluation.trajectory.decision_regret}",
                    f"- Decision efficiency: {evaluation.trajectory.decision_efficiency}",
                    f"- Decision consistency: {evaluation.trajectory.decision_consistency}",
                    f"- Adaptation ability: {evaluation.trajectory.adaptation_ability}",
                    f"- Counterproductive-action signal: {evaluation.trajectory.counterproductive_action_detection}",
                    f"- Short-term gain signal: {evaluation.trajectory.short_term_gain}",
                    f"- Long-term damage signal: {evaluation.trajectory.long_term_damage}",
                    f"- Good: {', '.join(evaluation.trajectory.good_signals) or 'none recorded'}",
                    f"- Bad: {', '.join(evaluation.trajectory.bad_signals) or 'none recorded'}",
                    f"- Recommended improvement: {', '.join(evaluation.trajectory.recommended_improvements) or 'none recorded'}",
                ]
            )
            lines.extend(
                [
                    "",
                    "### Scientific domain scores",
                    "",
                    "| Domain | Score | Included | Excluded | Failed |",
                    "| --- | ---: | --- | --- | --- |",
                ]
            )
            lines.extend(
                f"| {domain.domain} | {domain.value if domain.value is not None else '-'} | "
                f"{', '.join(domain.included_metrics) or '-'} | "
                f"{', '.join(domain.excluded_metrics) or '-'} | "
                f"{', '.join(domain.failed_metrics) or '-'} |"
                for domain in evaluation.domain_scores
            )
            if evaluation.robustness is not None:
                lines.extend(
                    [
                        "",
                        "### Robustness",
                        "",
                        f"- Seeds: {', '.join(str(seed) for seed in evaluation.robustness.seeds)}",
                        "- Seed stability: "
                        f"{_unmeasured_or(evaluation.robustness.seed_stability)}",
                        "- Clustering pairwise ARI: "
                        f"{_unmeasured_or(evaluation.robustness.clustering_pairwise_ari)}",
                        "- Prediction agreement: "
                        f"{_unmeasured_or(evaluation.robustness.annotation_prediction_agreement)}",
                    ]
                )
        return "\n".join(lines) + "\n"


class ScientificActionExecutor:
    """Async environment port that adapts Scanpy results to benchmark artifacts."""

    def __init__(
        self,
        context: ScientificContext,
        event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        expected_outputs: dict[str, list[str]] | None = None,
    ) -> None:
        self.context = context
        self.executor = ScanpyExecutor()
        self.event_callback = event_callback
        self.expected_outputs = expected_outputs or {}

    async def _emit(self, event: dict[str, Any]) -> None:
        """Forward an observable action event without coupling the executor to the API."""
        if self.event_callback is None:
            return
        callback_result = self.event_callback(event)
        if inspect.isawaitable(callback_result):
            await callback_result

    async def execute(
        self,
        intent: ActionIntent,
        context: ExecutionContext,
    ) -> ActionExecutionResult:
        """Execute a validated action without exposing AnnData to the agent."""
        del context
        step = len(self.context.operations) + 1
        await self._emit(
            {
                "type": "action_started",
                "step": step,
                "action_id": intent.action_id,
                "parameters": intent.parameters,
            }
        )
        # Scanpy operations are synchronous and can otherwise block the API
        # event loop, preventing SSE and polling updates from reaching the UI.
        result = await asyncio.to_thread(self.executor.execute, intent, self.context)
        expected_outputs = [
            str(output)
            for output in intent.metadata.get(
                "expected_outputs",
                self.expected_outputs.get(intent.action_id, []),
            )
        ]
        if result.status.value != "succeeded" or not expected_outputs:
            await self._emit(
                {
                    "type": "action_finished",
                    "step": step,
                    "action_id": intent.action_id,
                    "status": result.status.value,
                    "error": result.error,
                    "artifacts": [artifact.artifact_id for artifact in result.artifacts],
                }
            )
            return result
        artifacts = [
            artifact.model_copy(
                update={
                    "artifact_id": expected_outputs[index]
                    if index < len(expected_outputs)
                    else artifact.artifact_id
                }
            )
            for index, artifact in enumerate(result.artifacts)
        ]
        result = result.model_copy(update={"artifacts": artifacts})
        await self._emit(
            {
                "type": "action_finished",
                "step": step,
                "action_id": intent.action_id,
                "status": result.status.value,
                "error": result.error,
                "artifacts": [artifact.artifact_id for artifact in result.artifacts],
            }
        )
        return result


class ScientificLoop:
    """Run an interchangeable agent adapter inside a real scientific episode."""

    def __init__(self, *, cache_dir: Path | str = Path(".cache/datasets")) -> None:
        self.cache_dir = Path(cache_dir)

    async def run(
        self,
        benchmark: str | Path,
        *,
        agent_type: str = "rule-based",
        output_dir: Path | str = Path("results"),
        seed: int = 0,
        max_cells: int | None = None,
        max_steps: int | None = None,
        model: str | None = None,
        provider: str | None = None,
        environment: str | None = None,
        test_mode: bool = False,
        event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> AgentScientificRun:
        """Load data, run the harness, score local/global outcomes, and persist."""
        specification = self._resolve_benchmark(benchmark)
        task = specification.tasks[0]
        from agent_evals.datasets.pbmc import PBMCDataset

        dataset = PBMCDataset(cache_dir=self.cache_dir)
        # Dataset IO/decompression is synchronous; keep the API responsive while
        # the first run populates the cache.
        adata = await asyncio.to_thread(dataset.load, max_cells=max_cells)
        # Validate the benchmark's data contract before spending a model call.
        # Without this an impossible task (batch correction on data with no batch
        # column) reaches the agent, which is then blamed for the harness's gap.
        readiness = describe_readiness(
            adata,
            specification,
            task,
            source=str(dataset.metadata.source),
        )
        for warning in readiness.warnings:
            await _emit_event(
                event_callback,
                {"type": "dataset_warning", "message": warning},
            )
        try:
            validate_dataset_contract(readiness, specification, task)
        except DatasetContractError as error:
            await _emit_event(
                event_callback,
                {"type": "dataset_rejected", "message": str(error)},
            )
            raise
        requested_run_id = str(uuid4())
        pending_root = Path(output_dir) / requested_run_id
        store = LocalArtifactStore(pending_root / "artifacts")
        context = ScientificContext(
            adata=adata,
            dataset_metadata=dataset.metadata.model_dump(),
            artifact_store=store,
            workspace=pending_root,
        )
        # The reference differential-expression evidence, computed once here on the
        # dataset as issued rather than per step. ``context.metadata`` is read by
        # exactly one thing -- the metric context assembled after the run -- so it
        # is an evaluator channel; the reference marker list must never reach an
        # observation, and a test asserts that.
        reference_de, reference_de_gap = await asyncio.to_thread(
            reference_de_metadata,
            specification.metadata.id,
            adata,
            build_reference_artifacts(adata),
        )
        context.metadata.update(reference_de)
        if reference_de_gap is not None:
            # Emitted as a dataset limitation rather than swallowed, so an
            # exclusion downstream is attributable to its real cause instead of
            # looking like a benchmark that never asked for marker recovery.
            await _emit_event(
                event_callback,
                {"type": "dataset_warning", "message": reference_de_gap},
            )
        # Wraps rather than replaces the reward evaluator, so the reward scalar the
        # environment records is unchanged and ``S_t`` rides alongside it as
        # evaluator-side evidence. Reference-derived quality must not become the
        # number an agent optimizes directly.
        reward_evaluator = StageAwareRewardEvaluator(
            RewardEvaluator(),
            ScientificProgressTracker(self._load_metric_profile(specification)),
            context,
            self._progress_metric_ids(specification),
        )
        # Provisioning follows the benchmark's own declaration; ``environment``
        # only selects among what it declares. An opt-in flag would make omitting
        # it a silent misconfiguration, and the router below turns that same
        # mistake into a refusal at construction rather than a run in which every
        # free-execution action failed for the harness's reason.
        selected_environment = select_environment(specification, task, environment)
        provisioned = (
            None
            if selected_environment is None
            else await provision_environment(
                specification,
                selected_environment,
                adata,
                run_root=pending_root,
                task=task,
            )
        )
        if provisioned is not None:
            await _emit_event(
                event_callback,
                {
                    "type": "environment_provisioned",
                    "environment_id": provisioned.environment.id,
                    "workspace_root": str(provisioned.workspace_root),
                    "limitations": provisioned.limitations(),
                },
            )
        # Observation values come from three places and only one of them is the
        # scientific state: a benchmark can declare a value it owns (the DE
        # contrast), the scientific builder measures the dataset, and the
        # workspace builder measures what is on disk and what the last command
        # printed -- neither of which the scientific builder can see, since it
        # holds an in-memory ``AnnData`` and an operation history only the typed
        # executor writes. Argument order is precedence order: a measurement
        # overrides a declaration, never the reverse.
        observation_builder: ObservationBuilder = CompositeObservationBuilder(
            DeclaredObservationBuilder(),
            ScientificObservationBuilder(context),
            *(
                ()
                if provisioned is None
                else (WorkspaceObservationBuilder(provisioned.backend),)
            ),
        )
        episode_environment = ScientificEnvironment(
            specification,
            task_id=task.id,
            # Always routed, even with nothing provisioned: the router is a
            # pass-through when a benchmark declares no free-execution actions,
            # and it refuses at construction when it declares some and has
            # nowhere to run them.
            executor=ActionKindRouter.from_specification(
                specification,
                typed=ScientificActionExecutor(
                    context,
                    event_callback=event_callback,
                    expected_outputs={
                        action.id: list(action.expected_outputs)
                        for action in specification.actions
                    },
                ),
                free=None if provisioned is None else provisioned.executor,
            ),
            observation_builder=observation_builder,
            reward_evaluator=reward_evaluator,
            # Injected here rather than reached for inside the environment, so
            # ``runtime.py`` keeps knowing only the port and stays importable
            # without the science extra installed.
            artifact_validator=ArtifactRuleValidator(),
        )
        adapter = build_agent_adapter(
            agent_type,
            model=model,
            event_callback=event_callback,
            test_mode=test_mode,
        )
        # Test mode replaces the selected legacy agent with the universal GLM
        # runtime, so it must receive the same finite default step cap.
        effective_max_steps = (
            DEFAULT_RUNTIME_MAX_STEPS
            if max_steps is None and (test_mode or is_universal_runtime(agent_type))
            else max_steps
        )
        configuration = AgentConfiguration(
            agent_type=agent_type,
            model=model,
            provider=provider,
            seed=seed,
            max_steps=effective_max_steps,
            workspace={"root": str(pending_root)},
            metadata={
                "dataset_id": task.datasets[0] if task.datasets else "pbmc",
                "scientific_loop": True,
                "run_id": requested_run_id,
            },
        )
        agent_run = await AgentHarness().run(adapter, episode_environment, configuration)
        if provisioned is not None:
            # Before the run directory is renamed below: the backend holds the
            # workspace root by path, and a Windows directory rename fails while
            # anything under it is still open. Nothing is deleted -- the workspace
            # and its artifacts are evidence and outlive the run.
            await provisioned.backend.close()
        trace = getattr(adapter, "decision_trace", [])
        if not trace:
            trace = [
                {
                    "step": action.step,
                    "decision": {
                        "action_id": action.intent.action_id,
                        "method": action.intent.metadata.get("method", action.intent.action_id),
                        "parameters": action.intent.parameters,
                        "rationale": action.intent.rationale,
                    },
                    "intent": action.intent.model_dump(mode="json"),
                    "result": action.result.model_dump(mode="json"),
                }
                for action in agent_run.final_environment_state.state.actions
            ]
        evaluation = await asyncio.to_thread(
            self._evaluate_scientific_run,
            specification,
            task,
            agent_run,
            context,
            store,
            reward_evaluator.signals,
            provisioned,
        )
        metrics = evaluation.metric_results
        global_reward = GlobalReward(
            value=evaluation.scientific_outcome_score,
            components={
                metric.metric_id: float(metric.normalized_value)
                for metric in metrics
                if metric.normalized_value is not None
            },
            metric_ids=[metric.metric_id for metric in metrics],
            status="succeeded"
            if evaluation.scientific_outcome_score is not None
            else "unavailable",
        )
        final_root = Path(output_dir) / agent_run.run_id
        final_root.parent.mkdir(parents=True, exist_ok=True)
        if pending_root != final_root:
            pending_root.rename(final_root)
        run = AgentScientificRun(
            run_id=agent_run.run_id,
            agent_id=agent_run.agent_id,
            benchmark_id=agent_run.benchmark_id,
            task_id=agent_run.task_id,
            episode_id=agent_run.episode_id,
            agent_run=agent_run,
            trajectory=trace,
            artifacts=agent_run.generated_artifacts,
            local_rewards=agent_run.final_environment_state.state.rewards,
            global_reward=global_reward,
            final_metrics=metrics,
            evaluation=evaluation,
            report_path=str(final_root / "report.md"),
            environment=None if provisioned is None else provisioned.describe(),
        )
        (final_root / "trajectory.json").write_text(
            json.dumps(
                {
                    "agent": agent_run.agent_id,
                    "benchmark": agent_run.benchmark_id,
                    "steps": trace,
                    "local_rewards": [
                        reward.model_dump(mode="json") for reward in run.local_rewards
                    ],
                    "global_reward": global_reward.model_dump(mode="json"),
                    "final_reward": global_reward.value,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (final_root / "agent_run.json").write_text(agent_run.to_json(), encoding="utf-8")
        (final_root / "agent.json").write_text(
            json.dumps(
                {
                    "manifest": agent_run.manifest.model_dump(mode="json") if agent_run.manifest else None,
                    "configuration": agent_run.configuration.model_dump(mode="json"),
                    "agent_id": agent_run.agent_id,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (final_root / "actions.json").write_text(
            json.dumps(
                [action.model_dump(mode="json") for action in agent_run.final_environment_state.state.actions],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (final_root / "tool_calls.json").write_text(
            json.dumps(
                [
                    event.model_dump(mode="json")
                    for event in agent_run.trajectory.events
                    if event.event_type.value in {"agent.tool_call", "agent.tool_result"}
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (final_root / "metrics.json").write_text(
            json.dumps(
                {
                    "final_metrics": [metric.model_dump(mode="json") for metric in run.final_metrics],
                    "evaluation": run.evaluation.model_dump(mode="json") if run.evaluation else None,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (final_root / "report.json").write_text(run.to_json(), encoding="utf-8")
        (final_root / "report.md").write_text(run.to_markdown(), encoding="utf-8")
        return run

    @staticmethod
    def _evaluate_scientific_run(
        specification: BenchmarkSpecification,
        task: Any,
        agent_run: AgentRun,
        context: ScientificContext,
        store: LocalArtifactStore,
        progress_signals: Sequence[ProgressSignal] = (),
        provisioned: Any | None = None,
    ) -> ScientificEvaluation:
        """Run versioned evaluation on hidden reference data and visible outputs."""
        adata = context.adata
        # Only columns this run's agent actually wrote may count as predictions.
        # Reading a pre-existing column (bulk_labels, cell_type, or the dataset's
        # own louvain assignment) would score reference biology as agent output.
        prediction_column = context.agent_prediction_column()
        # Built through ``evaluation.candidates`` rather than inline, so the final
        # outcome and the per-step ``S_t`` see the same artifacts. Assembled
        # separately they could disagree for reasons that are the harness's, not
        # the agent's, and the progress signal would measure that disagreement.
        if (
            provisioned is not None
            and not specification.evaluator_executes_task_actions(task)
        ):
            candidate_inputs = build_workspace_metric_inputs(
                {
                    artifact.artifact_id: artifact
                    for artifact in agent_run.generated_artifacts
                },
                provisioned.reference_store,
            )
        else:
            candidate_inputs = build_metric_inputs(
                adata,
                prediction_column=prediction_column,
                cluster_column=context.agent_cluster_column(),
                evaluator_observes_predictions=(
                    specification.evaluator_executes_task_actions(task)
                ),
                # Read off the artifact the agent archived itself, never from
                # ``adata.uns`` -- pbmc68k ships ``rank_genes_groups`` precomputed over
                # the reference labels, so the shortcut reads the answer key.
                de_table=load_de_table(context.artifacts),
            )
        prediction = candidate_inputs.prediction
        candidate_artifacts = candidate_inputs.candidate_artifacts
        reference_artifacts = candidate_inputs.reference_artifacts
        # No placeholder table is archived when there was no candidate to build one
        # from. A CSV of ``__unassigned__`` rows looks exactly like a run that
        # annotated every cell wrongly, and it is the file someone opens to find
        # out why the outcome came back unmeasured.
        prediction_artifact = (
            store.save_table(
                "evaluation-prediction",
                prediction,
                metadata={
                    "hidden_from_agent": True,
                    "source_column": prediction_column,
                    "agent_produced_prediction": prediction_column is not None,
                },
            )
            if prediction is not None
            else None
        )
        metric_ids = [
            item.metric_id
            for group in specification.metric_groups
            for item in group.metrics
        ]
        # No hardcoded fallback list. A benchmark declaring no ``metric_groups``
        # used to fall back to five annotation metrics regardless of what it
        # measured, which is the same defect as the profile fallback one layer
        # down: a DE benchmark got a cell-annotation score. The profile below is
        # the declaration, and it is now benchmark-specific.
        metric_profile = ScientificLoop._load_metric_profile(specification)
        for group in metric_profile.metric_groups.values():
            metric_ids.extend(name for name in group.metrics if name not in metric_ids)
        # A set over every group, not an index into one. This read
        # ``metric_groups["robustness"]`` directly, and neither the integration
        # nor the DE profile declares that group -- so this line raised KeyError
        # for exactly the two benchmarks that resolving profiles correctly makes
        # reachable, which is why the two changes are one commit.
        external_scores = profile_external_scores(metric_profile)
        metric_ids = [name for name in metric_ids if name not in external_scores]
        groups = [
            MetricGroup(
                group_id=group.group_id,
                metrics=[MetricWeight(metric_id=item.metric_id, weight=item.weight, role=item.role) for item in group.metrics],
                aggregation=group.aggregation,
                minimum_required=group.minimum_required,
                contributes_to_primary=group.contributes_to_primary,
            )
            for group in specification.metric_groups
        ]
        metric_context = ScientificMetricContext(
            adata=adata,
            candidate_artifacts=candidate_artifacts,
            reference_artifacts=reference_artifacts,
            metadata={
                **context.metadata,
                **(
                    {"leakage_findings": list(candidate_inputs.leakage_findings)}
                    if candidate_inputs.leakage_findings
                    else {}
                ),
                # Which of the agent's own groups the DE ranking is read from.
                # Resolved by overlap with the hidden reference population rather
                # than by name, because the agent chose its own class names.
                **scored_group_metadata(
                    candidate_artifacts,
                    reference_artifacts,
                    context.metadata,
                ),
                **(
                    {"prediction_artifact_uri": str(prediction_artifact.path)}
                    if prediction_artifact is not None
                    else {}
                ),
            },
            trajectory=agent_run.trajectory.model_dump(mode="json"),
            agent_produced_columns=frozenset(context.agent_produced_columns),
            reference_join_gap=candidate_inputs.reference_join_gap,
        )
        engine = ScientificMetricEngine()
        results, applicability, group_results, _legacy_scientific_score = engine.evaluate(
            metric_ids,
            metric_context,
            groups=groups,
        )
        if prediction_artifact is not None:
            for result in results:
                result.metadata["candidate_evidence_uri"] = str(prediction_artifact.path)
        robustness = RobustnessEvaluator().evaluate(
            [
                {
                    "seed": agent_run.configuration.seed,
                    "cluster_labels": candidate_artifacts.get("cluster_labels"),
                    "predicted_labels": (
                        prediction["predicted_label"].tolist()
                        if prediction is not None
                        else None
                    ),
                    "artifact_checksums": (
                        [prediction_artifact.checksum]
                        if prediction_artifact is not None
                        else []
                    ),
                }
            ]
        )
        metric_inputs = [
            MetricScoreInput.from_metric_result(result) for result in results
        ]
        domain_scores = []
        for domain_name, group in metric_profile.metric_groups.items():
            inputs = list(metric_inputs)
            if group.external_score == "robustness.seed_stability":
                inputs.append(
                    MetricScoreInput.from_external_score(
                        group.external_score, robustness.seed_stability
                    )
                )
            domain_scores.append(
                WeightedGeometricAggregator().aggregate(domain_name, group, inputs)
            )
        scientific = aggregate_domains(domain_scores)
        scientific_score = scientific.value
        decisions = DecisionEvaluator().evaluate(agent_run, task)
        methods = MethodEvaluator().evaluate(agent_run, task, metric_ids, scientific_score)
        local_reward_values = [reward.value for reward in agent_run.final_environment_state.state.rewards]
        progress = summarize_progress(
            progress_signals,
            action_count=len(agent_run.final_environment_state.state.actions),
            token_usage=(
                agent_run.token_usage.total_tokens if agent_run.token_usage else None
            ),
            runtime_seconds=agent_run.wall_clock_seconds,
        )
        trajectory = TrajectoryEvaluator().evaluate(
            agent_run,
            task,
            scientific_score,
            local_rewards=local_reward_values,
            alternative_methods={
                category: profile.alternatives
                for category, profile in specification.decision_evaluation.items()
            },
            progress=progress,
        )
        # ``None``, not 1.0, when there is nothing to score. An agent that runs
        # its own workflow without recording decisions used to collect a free
        # perfect score on this dimension, so the benchmark paid better for less
        # structure. Unmeasured propagates to no global score at all, which is
        # the convention ``compute_global_agent_score`` already follows for a
        # missing scientific outcome.
        decision_value = (
            sum(item.score for item in decisions) / len(decisions) if decisions else None
        )
        method_value = method_score(methods)
        profiles = ScientificLoop._decision_profiles(specification)
        selection_evaluator = MethodSelectionEvaluator(decision_ontology)
        selection_scores: list[MethodScore] = []
        local_decision_rewards: list[dict[str, Any]] = []
        reward_by_step = {
            reward.step: reward.value
            for reward in agent_run.final_environment_state.state.rewards
        }
        normalized_metric_values = {
            result.metric_id: float(result.normalized_value)
            for result in results
            if result.normalized_value is not None
        }
        local_evaluator = LocalRewardEvaluator()
        for decision in agent_run.trajectory.decisions.decisions:
            if decision.parent_decision_id is not None:
                continue
            profile = profiles.get(decision.decision_category)
            selection_scores.append(
                selection_evaluator.evaluate(
                    decision,
                    context.dataset_metadata,
                    profile,
                    results,
                )
            )
            evidence = dict(normalized_metric_values)
            try:
                step_number = int(decision.step_id.rsplit("-", 1)[-1])
            except ValueError:
                step_number = decision.order + 1
            evidence["decision_local_reward"] = reward_by_step.get(step_number, 0.0)
            before, after = ScientificLoop._decision_observations(decision)
            local_decision_rewards.append(
                local_evaluator.evaluate(decision, before, after, evidence).model_dump(
                    mode="json"
                )
            )
        # Only the selections that produced a number. A decision whose every
        # component was unanswerable contributes nothing rather than dragging the
        # mean toward a value the harness never observed.
        scored_selections = [
            item.overall for item in selection_scores if item.overall is not None
        ]
        selection_value = (
            sum(scored_selections) / len(scored_selections)
            if scored_selections
            else None
        )
        decision_quality = (
            None
            if decision_value is None or selection_value is None
            else decision_value * selection_value
        )
        weights = ScoreWeights(
            outcome=specification.scoring.outcome_weight,
            decision=specification.scoring.decision_weight,
            trajectory=specification.scoring.trajectory_weight,
        )
        global_score = compute_global_agent_score(
            scientific_score,
            decision_quality,
            trajectory.trajectory_quality,
            weights=weights,
            confidence=compute_score_confidence(
                ineligible_fraction_decision=_ineligible_fraction_decision(
                    selection_scores
                ),
                ineligible_fraction_trajectory=trajectory.unmeasured_weight,
                decision_penalty=specification.scoring.decision_confidence_penalty,
                trajectory_penalty=specification.scoring.trajectory_confidence_penalty,
            ),
        )
        benchmark_score = global_score.value if global_score is not None else None
        # Both halves, because they answer different questions and either alone
        # leaves an unexplained gap. The join gap says the evaluator could not
        # reach the reference at all; the domain descriptions say which scored
        # quantities went unmeasured and why. A DE run with no reference markers
        # has no join gap -- its candidate never existed -- so before this the
        # entire outcome was ``None`` beside an empty limitation list.
        limitations = [
            *candidate_inputs.limitations,
            *describe_unmeasured_domains(
                domain_scores,
                {result.metric_id: result.eligibility_reason for result in results},
            ),
        ]
        return ScientificEvaluation(
            metric_results=results,
            applicability=applicability,
            groups=group_results,
            domain_scores=domain_scores,
            robustness=robustness,
            decision_evaluations=decisions,
            method_evaluations=methods,
            method_selection_evaluations=selection_scores,
            local_decision_rewards=local_decision_rewards,
            trajectory=trajectory,
            scientific_outcome_score=scientific_score,
            scientific_outcome_formula=scientific.formula,
            outcome_limitations=limitations,
            decision_score=decision_value,
            method_score=method_value,
            decision_quality_score=decision_quality,
            trajectory_score=trajectory.trajectory_quality,
            global_agent_score=benchmark_score,
            benchmark_score=benchmark_score,
            score_formula=_score_formula(
                weights,
                scientific_outcome=scientific_score,
                decision=decision_value,
                selection=selection_value,
            ),
            score_detail=global_score,
        )

    @staticmethod
    def _decision_observations(
        decision: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Report the observed cell and gene counts either side of a decision.

        Sourced from the observed state delta rather than from the agent's claim,
        and ``(None, None)`` when nothing was observed -- which excludes the
        observation-derived reward components rather than scoring them zero.
        """
        delta = getattr(decision, "observed_state_delta", None)
        if delta is None:
            return None, None
        return (
            {
                OBSERVED_CELL_COUNT: delta.n_obs_before,
                OBSERVED_GENE_COUNT: delta.n_vars_before,
            },
            {
                OBSERVED_CELL_COUNT: delta.n_obs_after,
                OBSERVED_GENE_COUNT: delta.n_vars_after,
            },
        )

    @staticmethod
    def _progress_metric_ids(specification: BenchmarkSpecification) -> list[str]:
        """List the metrics worth recomputing every step to track ``S_t``.

        Restricted to the profile's weighted metrics because those are the only
        ones the tracker can aggregate; computing the rest each step would cost
        real time and change no number.
        """
        profile = ScientificLoop._load_metric_profile(specification)
        external = profile_external_scores(profile)
        return sorted(
            name for name in profile_metric_ids(profile) if name not in external
        )

    @staticmethod
    def _load_metric_profile(
        specification: BenchmarkSpecification,
    ) -> BenchmarkMetricProfile:
        """Resolve the scoring profile this benchmark declares.

        Both branches of this function used to return ``pbmc_annotation_profile()``
        -- the second unconditionally -- so an integration or differential-
        expression run was scored on ``clustering.ari`` and
        ``cell_annotation.rare_recall``, and produced a number that looked
        entirely ordinary. See
        :mod:`agent_evals.evaluation.profiles.resolution` for why an unknown
        benchmark now raises instead of defaulting.

        The preferential load from ``configs/metrics/*.yaml`` is gone with it. A
        YAML file and a built-in that mirror each other are two declarations of
        one scoring rule, and they drift *silently*; worse, the guard was
        ``path.exists()``, so deleting or misplacing the file changed the scoring
        rule with no diagnostic. ``load_metric_profile`` remains public for
        third-party profiles, and ``configs/metrics/pbmc_annotation.yaml`` remains
        as the documented example, pinned by a test to the built-in it mirrors.
        """
        return resolve_metric_profile(specification.metadata.id)

    @staticmethod
    def _decision_profiles(
        specification: BenchmarkSpecification,
    ) -> dict[DecisionCategory, DecisionProfile]:
        """Translate declarative YAML profiles into typed ontology profiles."""
        profiles: dict[DecisionCategory, DecisionProfile] = {}
        aliases = {
            "qc": DecisionCategory.QC_STRATEGY,
            "qc_strategy": DecisionCategory.QC_STRATEGY,
            "normalization": DecisionCategory.NORMALIZATION,
            "integration": DecisionCategory.INTEGRATION,
            "annotation": DecisionCategory.ANNOTATION,
        }
        for key, profile in specification.decision_evaluation.items():
            category = aliases.get(key)
            if category is None:
                try:
                    category = DecisionCategory(key)
                except ValueError:
                    continue
            profiles[category] = DecisionProfile(
                category=category,
                allowed_methods=profile.allowed_methods,
                expected_inputs=profile.expected_inputs,
                possible_alternatives=profile.alternatives,
                evaluator_metrics=profile.metrics,
                parameter_ranges=profile.parameter_ranges,
            )
        return profiles

    @staticmethod
    def _resolve_benchmark(reference: str | Path) -> BenchmarkSpecification:
        path = Path(reference)
        if path.exists():
            return load_benchmark(path)
        if not benchmark_spec_registry.list_ids():
            from agent_evals.scientific.benchmarks import register_scientific_benchmarks

            register_scientific_benchmarks()
        return benchmark_spec_registry.get(str(reference))


def _ineligible_fraction_decision(selections: Sequence[MethodScore]) -> float:
    """Share of the decision dimension's components that were unmeasurable.

    Counted over components rather than over decisions because that is where the
    gaps actually are: a decision whose category declares no parameter ranges
    still yields real evidence about method appropriateness and execution, and
    calling the whole decision ineligible would understate the run as badly as
    the old substituted numbers overstated it.

    The other half of the dimension -- ``decision_score`` -- needs no term here.
    It is built from whether the action was allowed and whether it succeeded,
    both of which the harness always observes.
    """
    if not selections:
        return 0.0
    total = 0
    unmeasured = 0
    for item in selections:
        # Three declared components per selection, whatever their values; the
        # denominator has to be what could have been measured, not what was.
        total += 3
        unmeasured += len(item.unmeasured_components)
    return 0.0 if total == 0 else unmeasured / total


def _score_formula(
    weights: ScoreWeights,
    *,
    scientific_outcome: float | None,
    decision: float | None,
    selection: float | None,
) -> str:
    """Describe the score, naming any dimension that could not be measured.

    Names the two halves of the decision dimension separately rather than just
    reporting ``decision_quality`` as absent, because which half went missing is
    the difference between an agent that recorded no decisions and a benchmark
    that declared nothing to score them against.
    """
    unmeasured = [
        name
        for name, value in (
            ("scientific_outcome", scientific_outcome),
            ("decision_score", decision),
            ("method_selection_score", selection),
        )
        if value is None
    ]
    return describe_score(weights, unmeasured)


def _unmeasured_or(value: float | None) -> str:
    """Render an optional score so a gap cannot be mistaken for a failure."""
    return "unmeasured" if value is None else str(value)


def _score_cell(value: float | None) -> str:
    """Render an optional score into a report table cell.

    Spelled out rather than shown as a dash, which the metric table above uses
    for a metric that was attempted and produced nothing. These components were
    never attempted, and a reader comparing two runs needs to be able to tell
    those apart.
    """
    return "unmeasured" if value is None else f"{value:.3f}"


__all__ = [
    "AgentScientificRun",
    "ScientificActionExecutor",
    "ScientificLoop",
]
