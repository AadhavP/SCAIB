"""Agent-driven scientific episode bridge."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.agents import (
    AgentConfiguration,
    AgentHarness,
    AgentRun,
    RuntimeAgentAdapter,
    agent_adapter_registry,
    agent_runtime_registry,
)
from agent_evals.agents.trajectory import DecisionCategory
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.registry import benchmark_spec_registry
from agent_evals.benchmarks.schema import BenchmarkSpecification
from agent_evals.core.config import get_settings
from agent_evals.datasets.preflight import (
    REFERENCE_COLUMN_CANDIDATES,
    DatasetContractError,
    describe_readiness,
    validate_dataset_contract,
)
from agent_evals.environment.models import (
    ActionExecutionResult,
    ActionIntent,
    ArtifactRecord,
    RewardRecord,
)
from agent_evals.environment.ports import ExecutionContext
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.evaluation import (
    DecisionEvaluator,
    LocalRewardEvaluator,
    MethodEvaluator,
    MethodSelectionEvaluator,
    ScientificEvaluation,
    ScientificMetricEngine,
    TrajectoryEvaluator,
    compute_global_agent_score,
)
from agent_evals.evaluation.methods import method_score
from agent_evals.evaluation.metrics.robustness import RobustnessEvaluator
from agent_evals.evaluation.models import MethodScore
from agent_evals.evaluation.profiles import load_metric_profile, pbmc_annotation_profile
from agent_evals.evaluation.scoring import (
    MetricScoreInput,
    WeightedGeometricAggregator,
    aggregate_domains,
)
from agent_evals.evaluation.taxonomy import DecisionProfile, decision_ontology
from agent_evals.evaluators.models import MetricResult
from agent_evals.evaluators.rewards import GlobalReward, RewardEvaluator
from agent_evals.metrics import MetricGroup, MetricWeight
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.results import MetricStatus
from agent_evals.scientific.artifacts.storage import LocalArtifactStore
from agent_evals.scientific.artifacts.validation import ArtifactRuleValidator
from agent_evals.scientific.context import ScientificContext
from agent_evals.scientific.executor.scanpy import ScanpyExecutor
from agent_evals.scientific.metrics import (
    aggregate_objective_score,
    compute_objective_metrics,
)
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
            "## Decision summary",
            "",
            "| Step | Decision | Method | Execution | Local reward |",
            "| ---: | --- | --- | --- | ---: |",
        ]
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
            f"{metric.normalized_score if metric.normalized_score is not None else '-'} |"
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
                        f"- {domain.domain.title()} score: {domain.value}"
                        for domain in evaluation.domain_scores
                    ],
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
                f"| {item.decision_id} | {item.method or '-'} | {item.appropriateness:.3f} | "
                f"{item.parameter_quality:.3f} | {item.execution_quality:.3f} | {item.overall:.3f} |"
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
                        f"- Seed stability: {evaluation.robustness.seed_stability}",
                        f"- Clustering pairwise ARI: {evaluation.robustness.clustering_pairwise_ari}",
                        f"- Prediction agreement: {evaluation.robustness.annotation_prediction_agreement}",
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
        observation_builder = ScientificObservationBuilder(context)
        reward_evaluator = RewardEvaluator()
        environment = ScientificEnvironment(
            specification,
            task_id=task.id,
            executor=ScientificActionExecutor(
                context,
                event_callback=event_callback,
                expected_outputs={
                    action.id: list(action.expected_outputs)
                    for action in specification.actions
                },
            ),
            observation_builder=observation_builder,
            reward_evaluator=reward_evaluator,
            # Injected here rather than reached for inside the environment, so
            # ``runtime.py`` keeps knowing only the port and stays importable
            # without the science extra installed.
            artifact_validator=ArtifactRuleValidator(),
        )
        adapter = _create_scientific_adapter(
            agent_type,
            model=model,
            provider=provider,
            test_mode=test_mode,
            event_callback=event_callback,
        )
        # Test mode replaces the selected legacy agent with the universal GLM
        # runtime, so it must receive the same finite default step cap.
        effective_max_steps = (
            DEFAULT_RUNTIME_MAX_STEPS
            if max_steps is None and (test_mode or agent_type in agent_runtime_registry.list())
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
        agent_run = await AgentHarness().run(adapter, environment, configuration)
        pipeline_parameters: dict[str, Any] = {}
        metrics = await asyncio.to_thread(
            compute_objective_metrics,
            specification.metadata.id,
            context.adata,
            pipeline_parameters,
            set(context.agent_produced_columns),
        )
        global_reward = GlobalReward(
            value=aggregate_objective_score(specification.metadata.id, metrics),
            components={
                metric.metric_id: float(metric.normalized_score)
                for metric in metrics
                if metric.normalized_score is not None
            },
            metric_ids=[metric.metric_id for metric in metrics],
            status="succeeded"
            if aggregate_objective_score(specification.metadata.id, metrics) is not None
            else "unavailable",
        )
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
    def _evaluate_scientific_run(  # noqa: C901
        specification: BenchmarkSpecification,
        task: Any,
        agent_run: AgentRun,
        context: ScientificContext,
        store: LocalArtifactStore,
    ) -> ScientificEvaluation:
        """Run versioned evaluation on hidden reference data and visible outputs."""
        import pandas as pd

        adata = context.adata
        cell_ids = [str(value) for value in adata.obs_names]
        # Only columns this run's agent actually wrote may count as predictions.
        # Reading a pre-existing column (bulk_labels, cell_type, or the dataset's
        # own louvain assignment) would score reference biology as agent output.
        prediction_column = context.agent_prediction_column()
        candidate: dict[str, Any] = {"cell_id": cell_ids}
        if prediction_column is not None:
            candidate["predicted_label"] = [str(value) for value in adata.obs[prediction_column]]
        else:
            candidate["predicted_label"] = ["__unassigned__"] * len(cell_ids)
        prediction = pd.DataFrame(candidate)
        prediction_artifact = store.save_table(
            "evaluation-prediction",
            prediction,
            metadata={
                "hidden_from_agent": True,
                "source_column": prediction_column,
                "agent_produced_prediction": prediction_column is not None,
            },
        )
        candidate_artifacts: dict[str, Any] = {"prediction": prediction}
        cluster_column = context.agent_cluster_column()
        if cluster_column is not None:
            candidate_artifacts["cluster_labels"] = (
                adata.obs[cluster_column].astype(str).to_numpy()
            )
        if "X_pca" in adata.obsm:
            candidate_artifacts["embedding"] = adata.obsm["X_pca"]
        label_column = next(
            (column for column in REFERENCE_COLUMN_CANDIDATES if column in adata.obs),
            None,
        )
        reference_artifacts = (
            {"labels": pd.DataFrame({"reference_label": adata.obs[label_column].astype(str).to_numpy()})}
            if label_column is not None
            else {}
        )
        metric_ids = [
            item.metric_id
            for group in specification.metric_groups
            for item in group.metrics
        ]
        if not metric_ids:
            metric_ids = [
                "cell_annotation.macro_f1",
                "cell_annotation.mcc",
                "cell_annotation.balanced_accuracy",
                "cell_annotation.rare_recall",
                "cell_annotation.accuracy",
            ]
        metric_profile = ScientificLoop._load_metric_profile(specification)
        for group in metric_profile.metric_groups.values():
            metric_ids.extend(name for name in group.metrics if name not in metric_ids)
        metric_ids = [
            name
            for name in metric_ids
            if name != metric_profile.metric_groups["robustness"].external_score
        ]
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
            metadata={**context.metadata, "prediction_artifact_uri": str(prediction_artifact.path)},
            trajectory=agent_run.trajectory.model_dump(mode="json"),
            agent_produced_columns=frozenset(context.agent_produced_columns),
        )
        engine = ScientificMetricEngine()
        results, applicability, group_results, _legacy_scientific_score = engine.evaluate(
            metric_ids,
            metric_context,
            groups=groups,
        )
        for result in results:
            result.metadata["candidate_evidence_uri"] = str(prediction_artifact.path)
        robustness = RobustnessEvaluator().evaluate(
            [
                {
                    "seed": agent_run.configuration.seed,
                    "cluster_labels": candidate_artifacts.get("cluster_labels"),
                    "predicted_labels": prediction["predicted_label"].tolist(),
                    "artifact_checksums": [prediction_artifact.checksum],
                }
            ]
        )
        metric_inputs = [
            MetricScoreInput(
                # ``metric_id`` is the dotted registry id the profiles key on;
                # ``metric_name`` on this model is the human-readable title, and
                # feeding it here would make every profile lookup miss silently.
                name=result.metric_id,
                value=result.normalized_value,
                applicable=result.eligible,
                structurally_ineligible=(
                    result.status is MetricStatus.STRUCTURALLY_INELIGIBLE
                ),
                status=result.status.value,
            )
            for result in results
        ]
        domain_scores = []
        for domain_name, group in metric_profile.metric_groups.items():
            inputs = list(metric_inputs)
            if group.external_score == "robustness.seed_stability":
                inputs.append(
                    MetricScoreInput(
                        name=group.external_score,
                        value=robustness.seed_stability,
                    )
                )
            domain_scores.append(
                WeightedGeometricAggregator().aggregate(domain_name, group, inputs)
            )
        scientific_score = aggregate_domains(domain_scores).value
        decisions = DecisionEvaluator().evaluate(agent_run, task)
        methods = MethodEvaluator().evaluate(agent_run, task, metric_ids, scientific_score)
        local_reward_values = [reward.value for reward in agent_run.final_environment_state.state.rewards]
        trajectory = TrajectoryEvaluator().evaluate(
            agent_run,
            task,
            scientific_score,
            local_rewards=local_reward_values,
            alternative_methods={
                category: profile.alternatives
                for category, profile in specification.decision_evaluation.items()
            },
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
            local_decision_rewards.append(
                local_evaluator.evaluate(decision, None, None, evidence).model_dump(mode="json")
            )
        selection_value = (
            sum(item.overall for item in selection_scores) / len(selection_scores)
            if selection_scores
            else None
        )
        decision_quality = (
            None
            if decision_value is None or selection_value is None
            else decision_value * selection_value
        )
        global_score = compute_global_agent_score(
            scientific_score,
            decision_quality,
            trajectory.trajectory_quality,
        )
        benchmark_score = global_score.value if global_score is not None else None
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
            decision_score=decision_value,
            method_score=method_value,
            decision_quality_score=decision_quality,
            trajectory_score=trajectory.trajectory_quality,
            global_agent_score=benchmark_score,
            benchmark_score=benchmark_score,
            score_formula=_score_formula(
                scientific_outcome=scientific_score,
                decision=decision_value,
                selection=selection_value,
            ),
        )

    @staticmethod
    def _load_metric_profile(specification: BenchmarkSpecification) -> Any:
        """Load the declarative profile, with a typed built-in fallback."""
        if specification.metadata.id == "pbmc-cell-annotation":
            path = (
                Path(__file__).resolve().parents[3]
                / "configs"
                / "metrics"
                / "pbmc_annotation.yaml"
            )
            if path.exists():
                return load_metric_profile(path)
            return pbmc_annotation_profile()
        return pbmc_annotation_profile()

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


def _score_formula(
    *,
    scientific_outcome: float | None,
    decision: float | None,
    selection: float | None,
) -> str:
    """Describe the score, naming any dimension that could not be measured.

    The formula string is persisted into result JSON and read by people
    comparing runs. A run with an unmeasured dimension has no global score, and
    the recorded formula has to say which dimension is missing -- otherwise the
    absent number looks like a crash rather than an honest gap.
    """
    formula = "scientific_outcome * decision_score * method_selection_score * trajectory_score"
    unmeasured = [
        name
        for name, value in (
            ("scientific_outcome", scientific_outcome),
            ("decision_score", decision),
            ("method_selection_score", selection),
        )
        if value is None
    ]
    if not unmeasured:
        return formula
    return f"{formula} (not computed: {', '.join(unmeasured)} unmeasured)"


def _unmeasured_or(value: float | None) -> str:
    """Render an optional score so a gap cannot be mistaken for a failure."""
    return "unmeasured" if value is None else str(value)


def _create_scientific_adapter(
    agent_type: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    test_mode: bool = False,
    event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
) -> Any:
    """Create a legacy adapter or wrap a universal runtime for scientific episodes."""
    if test_mode:
        settings = get_settings()
        test_model = (
            settings.llm_model
            or settings.glm_model
            or os.getenv("LLM_MODEL")
            or os.getenv("GLM_MODEL")
            or model
            or "z-ai/glm-5.2"
        )
        test_base_url = (
            settings.llm_base_url
            or settings.glm_base_url
            or settings.openrouter_base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("GLM_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        test_api_key = (
            settings.llm_api_key
            or settings.glm_api_key
            or settings.openrouter_api_key
            or settings.openai_api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("GLM_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not test_api_key:
            raise RuntimeError(
                "GLM test mode requires LLM_API_KEY, GLM_API_KEY, "
                "OPENROUTER_API_KEY, or OPENAI_API_KEY in the backend environment"
            )
        runtime = agent_runtime_registry.create(
            "openai-compatible",
            model=test_model,
            base_url=test_base_url,
            api_key=test_api_key,
        )
        return RuntimeAgentAdapter(runtime, event_callback=event_callback)
    if agent_type in agent_runtime_registry.list():
        runtime_config: dict[str, object] = {}
        if model is not None and agent_type not in {"gpt-5", "claude-sonnet"}:
            runtime_config["model"] = model
        return RuntimeAgentAdapter(
            agent_runtime_registry.create(agent_type, **runtime_config),
            event_callback=event_callback,
        )
    return agent_adapter_registry.create(agent_type)


__all__ = [
    "AgentScientificRun",
    "ScientificActionExecutor",
    "ScientificLoop",
]
