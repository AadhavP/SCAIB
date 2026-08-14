"""Agent-driven scientific episode bridge."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
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
    qualify_run,
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
    profile_digest,
    profile_external_scores,
    profile_metric_ids,
    resolve_metric_profile,
)
from agent_evals.evaluation.progress import (
    ProgressSignal,
    ScientificProgressTracker,
    summarize_progress,
)
from agent_evals.evaluation.qualification import RunQualification
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
from agent_evals.research.bundle import (
    write_event_ledger,
    write_run_bundle_manifest,
)
from agent_evals.research.certification import (
    ResearchCertification,
    evaluate_research_readiness,
    load_readiness_manifest,
)
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


class RunProvenance(BaseModel):
    """Reproduction manifest for one scored scientific episode.

    A score without the exact benchmark digest, input identity, execution
    backend, and software surface is not a reproducible result. This manifest is
    deliberately separate from the agent-visible observation: it may include
    evaluator-only identities, but it never exposes reference values.
    """

    model_config = ConfigDict(extra="forbid")

    provenance_version: str = "1.0.0"
    benchmark_specification_digest: str
    benchmark_version: str
    benchmark_id: str
    task_id: str
    dataset_id: str | None = None
    dataset_source: str | None = None
    source_dataset_sha256: str | None = None
    source_dataset_checksum_verified: bool | None = None
    agent_dataset_sha256: str | None = None
    reference_manifest_sha256: str | None = None
    #: Shape and subset are part of the dataset identity. A checksum of the
    #: cached source alone is insufficient when a smoke run intentionally loads
    #: only the first N cells.
    loaded_cells: int | None = Field(default=None, ge=0)
    loaded_genes: int | None = Field(default=None, ge=0)
    declared_cells: int | None = Field(default=None, ge=0)
    declared_genes: int | None = Field(default=None, ge=0)
    dataset_shape_verified: bool | None = None
    requested_max_cells: int | None = Field(default=None, ge=1)
    scoring_profile: str | None = None
    scoring_profile_sha256: str | None = None
    dependency_lock_sha256: str | None = None
    source_revision: str | None = None
    python_version: str
    platform: str
    machine: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    environment_backend: str | None = None
    environment_image: str | None = None
    environment_image_digest: str | None = None
    #: Stable hashes of the retained run components. These are not hashes of
    #: private model output; they let a verifier prove that the trajectory,
    #: environment event log, and artifact manifest in an archive are the ones
    #: that produced the reported score.
    trajectory_sha256: str | None = None
    environment_events_sha256: str | None = None
    event_ledger_sha256: str | None = None
    artifact_manifest_sha256: str | None = None
    isolation_report_sha256: str | None = None
    #: SHA-256 of ``archive_manifest.json``. The manifest hashes the materialized
    #: public run components (including agent-visible artifacts and workspace
    #: files), so this is an integrity claim about bytes on disk rather than only
    #: a digest of in-memory Pydantic values.
    archive_manifest_sha256: str | None = None
    #: Canonical digest of the score-bearing result surfaces. The byte manifest
    #: protects the files, while this semantic digest proves that a report's
    #: outcome, decision, trajectory, and qualification were not rewritten
    #: independently of the provenance record.
    result_sha256: str | None = None
    agent_endpoint: str | None = None
    termination_status: str | None = None
    termination_reason: str | None = None
    qualification_status: str | None = None
    research_manifest_sha256: str | None = None
    research_certificate_sha256: str | None = None
    research_certification_status: str | None = None
    step_count: int = Field(default=0, ge=0)
    limitations: list[str] = Field(default_factory=list)


class ArchiveVerification(BaseModel):
    """Independent verification result for a materialized run archive.

    ``archive_manifest.json`` is useful only if a consumer can re-derive it.
    This model keeps that check explicit and distinguishes a changed file from a
    file that was never included, so an archive reader does not mistake a partial
    copy for a clean reproduction.
    """

    model_config = ConfigDict(extra="forbid")

    valid: bool
    manifest_sha256: str | None = None
    checked_files: int = Field(default=0, ge=0)
    missing_files: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    unexpected_files: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


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
    #: Per-step scientific-state evidence used by trajectory scoring. Keeping this
    #: beside the final outcome prevents a report from claiming adaptation while
    #: discarding the ``S_t``/``dS_t`` values that support it.
    progress_signals: list[dict[str, Any]] = Field(default_factory=list)
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
    #: Reproduction metadata is optional for reports written before the manifest
    #: existed, but every new canonical run populates it.
    provenance: RunProvenance | None = None
    #: Independent archive verification, separate from the manifest that was
    #: verified. ``None`` is retained for reports written before this check.
    archive_verification: ArchiveVerification | None = None
    #: Whether this result is suitable for comparison with other certified runs.
    qualification: RunQualification | None = None
    #: Optional benchmark-wide research certificate. It is attached only when a
    #: separate evidence manifest has been evaluated; episode qualification alone
    #: cannot establish calibration, baselines, or cross-run statistics.
    research_certification: ResearchCertification | None = None

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
        if self.provenance is not None:
            lines.extend(
                [
                    "## Reproducibility manifest",
                    "",
                    f"- Benchmark specification digest: `{self.provenance.benchmark_specification_digest}`",
                    f"- Dataset source: `{self.provenance.dataset_source or 'unknown'}`",
                    f"- Source dataset SHA-256: `{self.provenance.source_dataset_sha256 or 'unavailable'}` "
                    f"(verified: `{self.provenance.source_dataset_checksum_verified}`)",
                    f"- Agent-visible dataset SHA-256: `{self.provenance.agent_dataset_sha256 or 'unavailable'}`",
                    f"- Reference manifest SHA-256: `{self.provenance.reference_manifest_sha256 or 'unavailable'}`",
                    f"- Loaded shape: `{self.provenance.loaded_cells or 'unknown'} cells x {self.provenance.loaded_genes or 'unknown'} genes`",
                    f"- Declared shape: `{self.provenance.declared_cells or 'unknown'} cells x {self.provenance.declared_genes or 'unknown'} genes` "
                    f"(verified: `{self.provenance.dataset_shape_verified}`)",
                    f"- Scoring profile: `{self.provenance.scoring_profile or 'unknown'}` "
                    f"({self.provenance.scoring_profile_sha256 or 'unavailable'})",
                    f"- Dependency lock SHA-256: `{self.provenance.dependency_lock_sha256 or 'unavailable'}`",
                    f"- Source revision: `{self.provenance.source_revision or 'unavailable'}`",
                    f"- Python: `{self.provenance.python_version}` on `{self.provenance.platform}/{self.provenance.machine}`",
                    f"- Environment: `{self.provenance.environment_backend or 'typed'}` "
                    f"({self.provenance.environment_image_digest or self.provenance.environment_image or 'in-process'})",
                    f"- Agent endpoint: `{self.provenance.agent_endpoint or 'not applicable'}`",
                    f"- Termination: `{self.provenance.termination_status or 'unknown'}` "
                    f"({self.provenance.termination_reason or 'no reason recorded'})",
                    f"- Qualification: `{self.provenance.qualification_status or 'not recorded'}`",
                    f"- Research certificate: `{self.provenance.research_certification_status or 'not attached'}` "
                    f"({self.provenance.research_manifest_sha256 or 'unavailable'})",
                    f"- Trajectory SHA-256: `{self.provenance.trajectory_sha256 or 'unavailable'}`",
                    f"- Event ledger SHA-256: `{self.provenance.event_ledger_sha256 or 'unavailable'}`",
                    f"- Artifact manifest SHA-256: `{self.provenance.artifact_manifest_sha256 or 'unavailable'}`",
                    f"- Public archive manifest SHA-256: `{self.provenance.archive_manifest_sha256 or 'unavailable'}`",
                    f"- Score-bearing result SHA-256: `{self.provenance.result_sha256 or 'unavailable'}`",
                    "",
                ]
            )
        if self.qualification is not None:
            lines.extend(
                [
                    "## Run qualification",
                    "",
                    f"- Status: **{self.qualification.status.value.upper()}**",
                    f"- Comparable score: `{self.qualification.score_comparable}`",
                    *[
                        f"- {reason}"
                        for reason in self.qualification.reasons
                    ],
                    "",
                ]
            )
        if self.research_certification is not None:
            certificate = self.research_certification
            lines.extend(
                [
                    "## Research-readiness certification",
                    "",
                    f"- Status: **{certificate.status.value.upper()}**",
                    f"- Claim level: `{certificate.claim_level}`",
                    f"- Readiness: `{certificate.readiness_fraction:.1%}`",
                    f"- Evidence manifest SHA-256: `{certificate.manifest_sha256}`",
                    f"- Certificate SHA-256: `{certificate.certificate_sha256 or 'unavailable'}`",
                    *[
                        f"- {reason}"
                        for reason in certificate.blocking_reasons
                    ],
                    "",
                ]
            )
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
        started = time.monotonic()
        result = await asyncio.to_thread(self.executor.execute, intent, self.context)
        elapsed = max(time.monotonic() - started, 0.0)
        # Scanpy's in-process executor does not own a process supervisor, so its
        # default ResourceUsage has no measured wall time. Capture the boundary
        # here instead of presenting a typed action as cost-free telemetry.
        result = result.model_copy(
            update={
                "resource_usage": result.resource_usage.model_copy(
                    update={
                        "wall_time_seconds": max(
                            result.resource_usage.wall_time_seconds,
                            elapsed,
                        )
                    }
                )
            }
        )
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

    async def run(  # noqa: C901
        self,
        benchmark: str | Path,
        *,
        agent_type: str = "rule-based",
        output_dir: Path | str = Path("results"),
        seed: int = 0,
        max_cells: int | None = None,
        task_id: str | None = None,
        dataset_id: str | None = None,
        max_steps: int | None = None,
        model: str | None = None,
        provider: str | None = None,
        agent_endpoint: str | None = None,
        environment: str | None = None,
        test_mode: bool = False,
        research_manifest: Path | str | None = None,
        event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> AgentScientificRun:
        """Load data, run the harness, score local/global outcomes, and persist."""
        specification = self._resolve_benchmark(benchmark)
        if not specification.tasks:
            raise ValueError(
                f"benchmark '{specification.metadata.id}' does not declare any tasks"
            )
        research_certification: ResearchCertification | None = None
        if research_manifest is not None:
            readiness_manifest = load_readiness_manifest(research_manifest)
            if readiness_manifest.benchmark_id != specification.metadata.id:
                raise ValueError(
                    "research readiness manifest benchmark_id does not match the "
                    f"loaded benchmark '{specification.metadata.id}'"
                )
            if readiness_manifest.benchmark_version != specification.metadata.version:
                raise ValueError(
                    "research readiness manifest benchmark_version does not match "
                    f"the loaded benchmark version '{specification.metadata.version}'"
                )
            research_certification = evaluate_research_readiness(
                readiness_manifest,
                evidence_root=Path(research_manifest).parent,
            )
        task = next(
            (item for item in specification.tasks if item.id == task_id),
            specification.tasks[0] if task_id is None else None,
        )
        if task is None:
            available = ", ".join(item.id for item in specification.tasks)
            raise ValueError(
                f"unknown task '{task_id}' for benchmark '{specification.metadata.id}'; "
                f"available tasks: {available}"
            )
        resolved_dataset_id = dataset_id or (task.datasets[0] if task.datasets else None)
        if resolved_dataset_id is not None and resolved_dataset_id not in task.datasets:
            raise ValueError(
                f"dataset '{resolved_dataset_id}' is not supported by task '{task.id}'"
            )
        from agent_evals.datasets.pbmc import AnnDataDataset, PBMCDataset

        try:
            dataset = _resolve_dataset_provider(
                resolved_dataset_id,
                cache_dir=self.cache_dir,
                ann_data_dataset=AnnDataDataset,
                pbmc_dataset=PBMCDataset,
            )
            # Dataset IO/decompression is synchronous; keep the API responsive
            # while the first run populates the cache.
            adata = await asyncio.to_thread(dataset.load, max_cells=max_cells)
        except DatasetContractError as error:
            await _emit_event(
                event_callback,
                {"type": "dataset_rejected", "message": str(error)},
            )
            raise
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
        if readiness.batch_key is not None:
            # Metric applicability is evaluator-owned. The key name is safe to
            # retain here, while the reference labels and metric values remain in
            # the evaluator channel; without this bridge the integration metric
            # definitions could see ``adata.obs['batch']`` but applicability would
            # incorrectly declare the structural batch requirement missing.
            context.metadata["batch"] = readiness.batch_key
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
        try:
            adapter = build_agent_adapter(
                agent_type,
                model=model,
                agent_endpoint=agent_endpoint,
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
                    "dataset_id": resolved_dataset_id or "pbmc",
                    "scientific_loop": True,
                    "run_id": requested_run_id,
                },
            )
        except BaseException:
            # Provisioning starts a long-lived container before the adapter is
            # constructed. A rejected provider, malformed model configuration, or
            # callback setup must not strand that container when no agent call has
            # happened yet.
            if provisioned is not None:
                await provisioned.backend.close()
            raise
        try:
            agent_run = await AgentHarness().run(adapter, episode_environment, configuration)
        finally:
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
        if research_certification is not None:
            evaluation = evaluation.model_copy(
                update={"research_certification": research_certification}
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
        environment_record = (
            provisioned.describe()
            if provisioned is not None
            else _typed_environment_record(specification)
        )
        provenance = _build_run_provenance(
            specification,
            task,
            agent_run,
            dataset,
            resolved_dataset_id,
            provisioned,
            environment_record,
            loaded_cells=int(adata.n_obs),
            loaded_genes=int(adata.n_vars),
            max_cells=max_cells,
        )
        if research_certification is not None:
            provenance = provenance.model_copy(
                update={
                    "research_manifest_sha256": research_certification.manifest_sha256,
                    "research_certificate_sha256": research_certification.certificate_sha256,
                    "research_certification_status": research_certification.status.value,
                }
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
            progress_signals=[
                _progress_signal_payload(signal)
                for signal in reward_evaluator.signals
            ],
            global_reward=global_reward,
            final_metrics=metrics,
            evaluation=evaluation,
            report_path=str(final_root / "report.md"),
            environment=environment_record,
            provenance=provenance,
            research_certification=research_certification,
        )
        (final_root / "progress.json").write_text(
            json.dumps(
                {
                    "signals": run.progress_signals,
                    "trajectory_progress": (
                        run.evaluation.trajectory.model_dump(mode="json")
                        if run.evaluation is not None
                        else None
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (final_root / "environment_events.json").write_text(
            json.dumps(
                [
                    event.model_dump(mode="json")
                    for event in agent_run.final_environment_state.events
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
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
        (final_root / "provenance.json").write_text(
            provenance.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        (final_root / "integrity.json").write_text(
            json.dumps(
                {
                    "trajectory_sha256": provenance.trajectory_sha256,
                    "environment_events_sha256": provenance.environment_events_sha256,
                    "event_ledger_sha256": provenance.event_ledger_sha256,
                    "artifact_manifest_sha256": provenance.artifact_manifest_sha256,
                    "isolation_report_sha256": provenance.isolation_report_sha256,
                    "archive_manifest_sha256": provenance.archive_manifest_sha256,
                    "result_sha256": provenance.result_sha256,
                    "components": {
                        "trajectory": "agent_run.json:trajectory",
                        "environment_events": "environment_events.json",
                        "event_ledger": "events.ndjson",
                        "artifact_manifest": "agent_run.json:generated_artifacts",
                        "isolation": "report.json:environment.isolation",
                        "archive_manifest": "archive_manifest.json",
                        "result": "report.json:score-bearing-result",
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raw_agent_events = [
            {
                "source": "agent-runtime",
                "event_type": getattr(
                    event.event_type,
                    "value",
                    str(event.event_type),
                ),
                "payload": event.model_dump(mode="json"),
            }
            for event in agent_run.raw_events
        ]
        boundary_exchanges = agent_run.metadata.get("boundary_exchanges")
        if isinstance(boundary_exchanges, list):
            raw_agent_events.extend(
                {
                    "source": "agent-boundary",
                    "event_type": "endpoint_exchange",
                    "payload": dict(exchange),
                }
                for exchange in boundary_exchanges
                if isinstance(exchange, Mapping)
            )
        raw_environment_events = [
            {
                "source": "scientific-environment",
                "event_type": getattr(
                    event.event_type,
                    "value",
                    str(event.event_type),
                ),
                "payload": event.model_dump(mode="json"),
            }
            for event in agent_run.final_environment_state.events
        ]
        (final_root / "events.json").write_text(
            json.dumps(
                [*raw_agent_events, *raw_environment_events],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        event_ledger_sha256 = write_event_ledger(
            final_root,
            [*raw_agent_events, *raw_environment_events],
        )
        provenance = provenance.model_copy(
            update={"event_ledger_sha256": event_ledger_sha256}
        )
        run = run.model_copy(update={"provenance": provenance})
        (final_root / "replay.json").write_text(
            json.dumps(
                {
                    "replay_version": "1.0.0",
                    "run_id": run.run_id,
                    "benchmark_id": run.benchmark_id,
                    "task_id": run.task_id,
                    "seed": agent_run.configuration.seed,
                    "deterministic": bool(
                        (task.constraints or specification.constraints).deterministic
                    ),
                    "replay_mode": "event_sourced_observation_only",
                    "event_ledger": "events.ndjson",
                    "event_ledger_sha256": event_ledger_sha256,
                    "trajectory": "agent_run.json",
                    "report": "report.json",
                    "artifacts": [artifact.artifact_id for artifact in run.artifacts],
                    "limitations": [
                        "descriptor validates replay inputs only; it does not execute arbitrary agent code",
                        "clean-room replay still requires the frozen benchmark, dependencies, and dataset/reference package",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
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

        # The in-memory component hashes above establish semantic provenance. Add
        # a second, byte-level manifest after every public component has been
        # materialized so an archive consumer can verify the files it actually
        # received. Metadata files are excluded to avoid a self-referential hash;
        # they are rewritten below from the now-final provenance object.
        archive_manifest_sha256 = _write_archive_manifest(final_root)
        archive_limitations = list(provenance.limitations)
        if archive_manifest_sha256 is None:
            archive_limitations.append(
                "the public archive manifest could not be written or hashed"
            )
        provenance = provenance.model_copy(
            update={
                "archive_manifest_sha256": archive_manifest_sha256,
                "limitations": archive_limitations,
            }
        )
        run = run.model_copy(update={"provenance": provenance})
        # Seed the semantic result digest before the first archive verification.
        # Qualification itself is attached below, so this is a provisional digest
        # for the otherwise complete result. The final digest is recomputed after
        # qualification and the archive is verified a second time; this breaks the
        # otherwise circular dependency between "archive is valid" and the
        # qualification field protected by the result digest.
        provisional_result_sha256 = _sha256_json(_result_integrity_payload(run))
        provenance = provenance.model_copy(
            update={"result_sha256": provisional_result_sha256}
        )
        run = run.model_copy(update={"provenance": provenance})
        (final_root / "provenance.json").write_text(
            provenance.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        (final_root / "integrity.json").write_text(
            json.dumps(
                {
                    "trajectory_sha256": provenance.trajectory_sha256,
                    "environment_events_sha256": provenance.environment_events_sha256,
                    "event_ledger_sha256": provenance.event_ledger_sha256,
                    "artifact_manifest_sha256": provenance.artifact_manifest_sha256,
                    "isolation_report_sha256": provenance.isolation_report_sha256,
                    "archive_manifest_sha256": provenance.archive_manifest_sha256,
                    "result_sha256": provenance.result_sha256,
                    "components": {
                        "trajectory": "agent_run.json:trajectory",
                        "environment_events": "environment_events.json",
                        "event_ledger": "events.ndjson",
                        "artifact_manifest": "agent_run.json:generated_artifacts",
                        "isolation": "report.json:environment.isolation",
                        "archive_manifest": "archive_manifest.json",
                        "result": "report.json:score-bearing-result",
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        # Keep the two human/machine report surfaces consistent with the manifest
        # file. They are excluded from the byte manifest because this rewrite is
        # precisely what records the manifest digest in the report.
        (final_root / "report.json").write_text(run.to_json(), encoding="utf-8")
        (final_root / "report.md").write_text(run.to_markdown(), encoding="utf-8")

        # Qualification happens only after the public archive exists and has been
        # independently re-hashed. A limitation in a report is useful evidence;
        # it must not silently leave an exploratory run looking comparable.
        archive_verification = verify_archive_manifest(final_root)
        qualification = qualify_run(
            agent_termination_status=agent_run.termination_status.value,
            environment=environment_record,
            provenance=provenance.model_dump(mode="json"),
            evaluation=evaluation,
            archive_valid=archive_verification.valid,
            artifacts=agent_run.generated_artifacts,
            state_evidence={
                "backend": (environment_record or {}).get("backend"),
                "action_count": len(agent_run.final_environment_state.state.actions),
                "observed_action_count": sum(
                    action.result.observed_state_delta is not None
                    for action in agent_run.final_environment_state.state.actions
                ),
                "limitations": sorted({
                    limitation
                    for action in agent_run.final_environment_state.state.actions
                    if action.result.observed_state_delta is not None
                    for limitation in action.result.observed_state_delta.limitations
                }),
            },
            cutoff_evidence=(
                agent_run.metadata.get("cutoff")
                if isinstance(agent_run.metadata.get("cutoff"), Mapping)
                else None
            ),
        )
        provenance = provenance.model_copy(
            update={"qualification_status": qualification.status.value}
        )
        evaluation = evaluation.model_copy(update={"qualification": qualification})
        run = run.model_copy(
            update={
                "evaluation": evaluation,
                "provenance": provenance,
                "archive_verification": archive_verification,
                "qualification": qualification,
            }
        )
        # The score-bearing result is hashed only after qualification has been
        # attached. Provenance/environment/archive metadata are deliberately
        # excluded from this payload so the hash is stable under relocation and
        # does not become circular with the manifest digest.
        result_sha256 = _sha256_json(_result_integrity_payload(run))
        provenance = provenance.model_copy(update={"result_sha256": result_sha256})
        run = run.model_copy(update={"provenance": provenance})
        (final_root / "integrity.json").write_text(
            json.dumps(
                {
                    "trajectory_sha256": provenance.trajectory_sha256,
                    "environment_events_sha256": provenance.environment_events_sha256,
                    "event_ledger_sha256": provenance.event_ledger_sha256,
                    "artifact_manifest_sha256": provenance.artifact_manifest_sha256,
                    "isolation_report_sha256": provenance.isolation_report_sha256,
                    "archive_manifest_sha256": provenance.archive_manifest_sha256,
                    "result_sha256": provenance.result_sha256,
                    "components": {
                        "trajectory": "agent_run.json:trajectory",
                        "environment_events": "environment_events.json",
                        "event_ledger": "events.ndjson",
                        "artifact_manifest": "agent_run.json:generated_artifacts",
                        "isolation": "report.json:environment.isolation",
                        "archive_manifest": "archive_manifest.json",
                        "result": "report.json:score-bearing-result",
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (final_root / "provenance.json").write_text(
            provenance.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        (final_root / "report.json").write_text(run.to_json(), encoding="utf-8")
        (final_root / "report.md").write_text(run.to_markdown(), encoding="utf-8")
        # Re-verify after the final qualification/result digest rewrite. The
        # initial verification made qualification auditable; this one is the
        # verification object returned to consumers of the final report.
        archive_verification = verify_archive_manifest(final_root)
        run = run.model_copy(update={"archive_verification": archive_verification})
        (final_root / "report.json").write_text(run.to_json(), encoding="utf-8")
        (final_root / "report.md").write_text(run.to_markdown(), encoding="utf-8")
        # Write the replay-oriented bundle manifest only after the final report
        # rewrite. Any later byte mutation is then visible to verify_run_bundle.
        write_run_bundle_manifest(
            final_root,
            run_id=run.run_id,
            event_ledger_sha256=event_ledger_sha256,
        )
        return run

    @staticmethod
    def _evaluate_scientific_run(  # noqa: C901
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
        profile_requirements = {
            metric_id: entry.required
            for group in metric_profile.metric_groups.values()
            for metric_id, entry in group.metrics.items()
        }
        for result in results:
            if result.metric_id in profile_requirements:
                # The registry role answers what a metric is in isolation; the
                # resolved benchmark profile answers whether this benchmark makes
                # it a certification gate. Persist both rather than treating an
                # optional, structurally inapplicable diagnostic as a failed run.
                result.metadata["profile_required"] = profile_requirements[result.metric_id]
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


def _resolve_dataset_provider(
    dataset_id: str | None,
    *,
    cache_dir: Path,
    ann_data_dataset: Any,
    pbmc_dataset: Any,
) -> Any:
    """Resolve a dataset by declared ID without substituting a different fixture.

    The old loop always instantiated ``PBMCDataset``. That made the integration
    benchmark appear to load ``pbmc-multi-batch`` while actually evaluating the
    annotation fixture, and preflight only caught it later because the fixture
    had no batch column. A benchmark runner must fail before a model call when a
    declared dataset is not provisioned. Operators can provide additional local
    AnnData collections through ``SCAIB_DATASET_<ID>`` or a same-ID file in the
    cache directory; neither path changes the public agent contract.
    """
    key = dataset_id or "pbmc68k"
    if key in {"pbmc68k", "pbmc68k_reduced"}:
        return pbmc_dataset(cache_dir=cache_dir)
    variable = "SCAIB_DATASET_" + "".join(
        character if character.isalnum() else "_" for character in key.upper()
    )
    configured = os.getenv(variable)
    candidate = Path(configured) if configured else cache_dir / f"{key}.h5ad"
    if candidate.is_file():
        return ann_data_dataset(candidate, source=f"configured dataset: {candidate}")
    raise DatasetContractError(
        f"dataset '{key}' has no provisioned AnnData provider. Expected a file at "
        f"'{candidate}' or set {variable}; refusing to substitute the PBMC68k "
        "annotation fixture for a different scientific task."
    )


_ARCHIVE_METADATA_FILES = frozenset(
    {
        "archive_manifest.json",
        "bundle_manifest.json",
        "integrity.json",
        "provenance.json",
        "report.json",
        "report.md",
    }
)


def _sha256_file(path: Path | None) -> str | None:
    """Hash a persisted input without turning a missing file into a run error."""
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def verify_archive_manifest(root: Path | str) -> ArchiveVerification:  # noqa: C901
    """Re-hash a run archive and report whether its public bytes are intact.

    The manifest deliberately excludes metadata files that contain its digest.
    Those files are still checked for a matching declared digest when provenance
    is available, while every scientific artifact and trajectory byte must be
    present and unchanged. This function never raises for a malformed archive;
    an archive is evidence, so the inability to verify it belongs in the result.
    """
    archive_root = Path(root)
    manifest_path = archive_root / "archive_manifest.json"
    if not manifest_path.is_file():
        return ArchiveVerification(
            valid=False,
            limitations=["archive_manifest.json is missing"],
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = payload.get("files")
        if not isinstance(expected, dict) or not all(
            isinstance(path, str) and isinstance(digest, str)
            for path, digest in expected.items()
        ):
            raise ValueError("manifest files must be an object of path to digest")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return ArchiveVerification(
            valid=False,
            limitations=[
                f"archive_manifest.json could not be read ({type(error).__name__}: {error})"
            ],
        )

    actual_manifest_digest = _sha256_file(manifest_path)
    if not archive_root.is_dir() or actual_manifest_digest is None:
        return ArchiveVerification(
            valid=False,
            manifest_sha256=actual_manifest_digest,
            limitations=["the run archive root could not be read"],
        )

    resolved_root = archive_root.resolve()
    missing: list[str] = []
    changed: list[str] = []
    checked = 0
    for relative, digest in sorted(expected.items()):
        candidate = archive_root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            missing.append(relative)
            continue
        if resolved_root not in resolved.parents:
            changed.append(relative)
            continue
        observed = _sha256_file(candidate)
        if observed is None:
            missing.append(relative)
        elif observed != digest:
            changed.append(relative)
        else:
            checked += 1

    expected_paths = set(expected)
    unexpected: list[str] = []
    try:
        for candidate in sorted(archive_root.rglob("*")):
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(archive_root).as_posix()
            if relative in expected_paths or relative in _ARCHIVE_METADATA_FILES:
                continue
            unexpected.append(relative)
    except (OSError, RuntimeError) as error:
        # The expected files were still checked; this is a limitation on the
        # completeness check, not evidence that any expected file changed.
        unexpected.append(f"<archive walk failed: {type(error).__name__}>")

    limitations: list[str] = []
    provenance_path = archive_root / "provenance.json"
    declared: dict[str, Any] | None = None
    if provenance_path.is_file():
        try:
            parsed = json.loads(provenance_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("provenance.json must contain an object")
            declared = parsed
            declared_digest = declared.get("archive_manifest_sha256")
            if declared_digest and declared_digest != actual_manifest_digest:
                changed.append("provenance.json:archive_manifest_sha256")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            limitations.append(
                f"provenance.json could not be checked ({type(error).__name__}: {error})"
            )
    else:
        limitations.append("provenance.json is missing, so no declared manifest digest was available")

    if declared is not None:
        semantic_changed, semantic_limitations = _verify_semantic_components(
            archive_root,
            declared,
        )
        changed.extend(semantic_changed)
        limitations.extend(semantic_limitations)

    return ArchiveVerification(
        valid=not missing and not changed and not unexpected and not limitations,
        manifest_sha256=actual_manifest_digest,
        checked_files=checked,
        missing_files=missing,
        changed_files=sorted(set(changed)),
        unexpected_files=unexpected,
        limitations=limitations,
    )


def _result_integrity_payload(run: AgentScientificRun) -> dict[str, Any]:
    """Return the relocation-stable, score-bearing portion of a run.

    Paths, host details, archive verification, and provenance are intentionally
    excluded. They describe where/how the result was materialized; the digest is
    for the scientific result and qualification that a consumer would compare.
    """
    return _result_integrity_payload_from_mapping(run.model_dump(mode="json"))


def _result_integrity_payload_from_mapping(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the same score payload from a deserialized report archive."""
    return {
        key: payload.get(key)
        for key in (
            "run_id",
            "agent_id",
            "benchmark_id",
            "task_id",
            "episode_id",
            "global_reward",
            "final_metrics",
            "local_rewards",
            "progress_signals",
            "evaluation",
            "qualification",
        )
    }


def _verify_semantic_components(  # noqa: C901
    root: Path,
    provenance: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Verify canonical component hashes in addition to archive file bytes.

    The byte manifest catches changed files, but it intentionally excludes the
    metadata files that carry its own digest. Without this second layer, a
    consumer could edit ``report.json`` to change a score while leaving the
    archive's scientific files untouched. These checks compare independently
    derived canonical JSON values against the evaluator-owned provenance claims.
    """
    changed: list[str] = []
    limitations: list[str] = []

    def read_json(name: str) -> Any | None:
        path = root / name
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            limitations.append(f"{name} could not be checked ({type(error).__name__}: {error})")
            return None

    def compare(name: str, value: Any, field: str) -> None:
        expected = provenance.get(field)
        if expected is None:
            limitations.append(f"provenance field '{field}' is missing")
        elif value is None:
            return
        elif _sha256_json(value) != expected:
            changed.append(f"{name}:{field}")

    agent_run = read_json("agent_run.json")
    if isinstance(agent_run, dict):
        compare("agent_run.json:trajectory", agent_run.get("trajectory"), "trajectory_sha256")
        compare(
            "agent_run.json:generated_artifacts",
            agent_run.get("generated_artifacts"),
            "artifact_manifest_sha256",
        )
    elif agent_run is not None:
        limitations.append("agent_run.json must contain an object")

    environment_events = read_json("environment_events.json")
    if environment_events is not None:
        compare("environment_events.json", environment_events, "environment_events_sha256")

    report = read_json("report.json")
    if isinstance(report, dict):
        environment = report.get("environment")
        isolation = environment.get("isolation") if isinstance(environment, dict) else None
        if isolation is not None:
            compare("report.json:environment.isolation", isolation, "isolation_report_sha256")
        result_digest = provenance.get("result_sha256")
        if result_digest is None:
            limitations.append("provenance field 'result_sha256' is missing")
        elif _sha256_json(_result_integrity_payload_from_mapping(report)) != result_digest:
            changed.append("report.json:score-bearing-result")
    elif report is not None:
        limitations.append("report.json must contain an object")

    integrity = read_json("integrity.json")
    if isinstance(integrity, dict):
        for field in (
            "trajectory_sha256",
            "environment_events_sha256",
            "artifact_manifest_sha256",
            "isolation_report_sha256",
            "archive_manifest_sha256",
            "result_sha256",
        ):
            if integrity.get(field) != provenance.get(field):
                changed.append(f"integrity.json:{field}")
    elif integrity is not None:
        limitations.append("integrity.json must contain an object")

    return changed, limitations


def _typed_environment_record(specification: BenchmarkSpecification) -> dict[str, Any]:
    """Describe the evaluator-owned typed tier with the same honesty as a workspace.

    Typed actions do not grant the agent a shell or a writable workspace: SCAIB's
    registered executor applies the declared operation to an evaluator-owned
    in-memory object. Calling that simply ``in-process`` without a limitation
    would make the two execution tiers look equivalent when they measure
    different capabilities.
    """
    return {
        "environment_id": "typed-in-process",
        "backend": "typed",
        "image": None,
        "image_digest": None,
        "user": None,
        "workspace_root": None,
        "agent_dataset": "evaluator-owned in-memory AnnData",
        "withheld_obs_columns": [],
        "withheld_uns_keys": [],
        "withheld_obsm_keys": [],
        "retained_analysis_keys": [],
        "reference_store": "evaluator-only (not materialized for the agent)",
        "reference_store_scope": "evaluator-process memory",
        "isolation": {
            "backend": "typed",
            "platform": sys.platform,
            "controls": [],
        },
        "limitations": [
            "typed actions execute benchmark-owned implementations in the evaluator process; "
            "this run does not measure agent-authored shell or arbitrary workspace code",
            f"benchmark '{specification.metadata.id}' did not request a free-execution workspace",
        ],
    }


def _write_archive_manifest(root: Path) -> str | None:
    """Write hashes for the public bytes in a completed run archive.

    Metadata surfaces that contain the manifest digest are intentionally excluded
    to avoid a circular hash. Evaluator-only data is normally outside ``root``;
    the containment check also refuses an accidental symlink that points back to
    it. A missing component is omitted rather than assigned a fake digest, and a
    caller can treat a ``None`` return as an archive-integrity limitation.
    """
    files: dict[str, str] = {}
    resolved_root = root.resolve()
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative in _ARCHIVE_METADATA_FILES:
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved != resolved_root and resolved_root not in resolved.parents:
            # Never hash a symlink target outside the public archive.
            continue
        digest = _sha256_file(candidate)
        if digest is not None:
            files[relative] = digest
    manifest = {
        "manifest_version": "1.0.0",
        "scope": "public_run_archive",
        "files": files,
    }
    manifest_path = root / "archive_manifest.json"
    try:
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return None
    return _sha256_file(manifest_path)


def _sha256_json(value: Any) -> str:
    """Hash canonical JSON for an archive component without filesystem order."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_non_negative_int(value: Any) -> int | None:
    """Parse a declared dataset dimension without turning malformed metadata into zero."""
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _dependency_lock_sha256() -> str | None:
    """Hash the first supported dependency lock in a deterministic order."""
    for candidate in (
        Path("uv.lock"),
        Path("poetry.lock"),
        Path("Pipfile.lock"),
        Path("requirements.lock"),
        Path("requirements.txt"),
    ):
        digest = _sha256_file(candidate)
        if digest is not None:
            return digest
    return None


def _source_revision() -> str | None:
    """Return an operator-provided or lightweight local checkout revision.

    The runner must not shell out to Git during a scientific episode. CI and
    deployed workers can set ``SCAIB_SOURCE_REVISION``; for a source checkout we
    also resolve a simple ``.git/HEAD`` ref without invoking an external command.
    """
    configured = os.getenv("SCAIB_SOURCE_REVISION") or os.getenv("GIT_COMMIT")
    if configured:
        return configured
    head = Path(".git/HEAD")
    try:
        value = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value.startswith("ref: "):
        ref = Path(".git") / value[5:]
        try:
            return ref.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return value or None


def _progress_signal_payload(signal: ProgressSignal) -> dict[str, Any]:
    """Serialize the dataclass evidence without turning it into score input."""
    return {
        "step": signal.step,
        "stage": signal.stage.value if signal.stage is not None else None,
        "scientific_state": signal.scientific_state,
        "delta": signal.delta,
        "comparable_metrics": list(signal.comparable_metrics),
        "previous_state_on_comparable": signal.previous_state_on_comparable,
        "current_state_on_comparable": signal.current_state_on_comparable,
        "scored_metrics": list(signal.scored_metrics),
        "limitations": list(signal.limitations),
    }


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    """Capture the versions that can change scientific or protocol semantics."""
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _build_run_provenance(
    specification: BenchmarkSpecification,
    task: Any,
    agent_run: AgentRun,
    dataset: Any,
    dataset_id: str | None,
    provisioned: Any | None,
    environment_record: dict[str, Any] | None,
    *,
    loaded_cells: int,
    loaded_genes: int,
    max_cells: int | None,
) -> RunProvenance:
    """Create the evaluator-owned manifest archived beside every new run."""
    source_path = getattr(dataset, "local_path", None)
    agent_path = getattr(provisioned, "dataset_path", None)
    reference_store = getattr(provisioned, "reference_store", None)
    reference_manifest = (
        Path(reference_store) / "manifest.json"
        if reference_store is not None
        else None
    )
    limitations = list((environment_record or {}).get("limitations", []))
    source_checksum = _sha256_file(Path(source_path) if source_path else None)
    if source_checksum is None:
        limitations.append("the cached source dataset file could not be hashed")
    agent_checksum = _sha256_file(Path(agent_path) if agent_path else None)
    if provisioned is not None and agent_checksum is None:
        limitations.append("the sanitized agent-visible dataset file could not be hashed")
    reference_checksum = _sha256_file(reference_manifest)
    if provisioned is not None and reference_checksum is None:
        limitations.append("the evaluator reference manifest could not be hashed")
    dependency_lock_checksum = _dependency_lock_sha256()
    if dependency_lock_checksum is None:
        limitations.append(
            "no dependency lock file (uv.lock, poetry.lock, or requirements lock) "
            "was available to pin this run"
        )
    source_revision = _source_revision()
    if source_revision is None:
        limitations.append("the source checkout revision was not available")
    dataset_metadata = getattr(dataset, "metadata", None)
    source = getattr(dataset_metadata, "source", None)
    dataset_specification = next(
        (
            item
            for item in specification.datasets
            if item.id == dataset_id
        ),
        None,
    )
    expected_source_checksum = (
        dataset_specification.checksum.value
        if dataset_specification is not None and dataset_specification.checksum is not None
        else None
    )
    source_checksum_verified = (
        None
        if expected_source_checksum is None or source_checksum is None
        else source_checksum.lower() == expected_source_checksum.lower()
    )
    if source_checksum_verified is False:
        limitations.append(
            "the loaded source dataset checksum does not match the benchmark declaration"
        )
    expected_observations = (
        dict(dataset_specification.expected_observations)
        if dataset_specification is not None
        else {}
    )
    declared_cells = _as_non_negative_int(expected_observations.get("cells"))
    declared_genes = _as_non_negative_int(expected_observations.get("genes"))
    dataset_shape_verified = (
        None
        if declared_cells is None or declared_genes is None
        else loaded_cells == declared_cells and loaded_genes == declared_genes
    )
    if dataset_shape_verified is False:
        limitations.append(
            "the loaded dataset shape does not match the benchmark declaration "
            f"({loaded_cells} cells x {loaded_genes} genes loaded; "
            f"{declared_cells} cells x {declared_genes} declared)"
        )
    profile = ScientificLoop._load_metric_profile(specification)
    trajectory_payload = agent_run.trajectory.model_dump(mode="json")
    environment_events_payload = [
        event.model_dump(mode="json")
        for event in agent_run.final_environment_state.events
    ]
    artifact_manifest_payload = [
        artifact.model_dump(mode="json")
        for artifact in agent_run.generated_artifacts
    ]
    isolation_payload = (environment_record or {}).get("isolation")
    agent_endpoint = (
        agent_run.manifest.metadata.get("endpoint")
        if agent_run.manifest is not None
        and isinstance(agent_run.manifest.metadata.get("endpoint"), str)
        else None
    )
    return RunProvenance(
        benchmark_specification_digest=agent_run.final_environment_state.state.specification_digest,
        benchmark_version=specification.metadata.version,
        benchmark_id=specification.metadata.id,
        task_id=task.id,
        dataset_id=dataset_id,
        dataset_source=str(source) if source is not None else None,
        source_dataset_sha256=source_checksum,
        source_dataset_checksum_verified=source_checksum_verified,
        agent_dataset_sha256=agent_checksum,
        reference_manifest_sha256=reference_checksum,
        # Use the object that entered this run, not the cache metadata. The
        # latter describes the full cached fixture even when ``max_cells`` made
        # this a reduced smoke run, which would make the provenance claim the
        # wrong scientific population was evaluated.
        loaded_cells=loaded_cells,
        loaded_genes=loaded_genes,
        declared_cells=declared_cells,
        declared_genes=declared_genes,
        dataset_shape_verified=dataset_shape_verified,
        requested_max_cells=max_cells,
        scoring_profile=profile.benchmark,
        scoring_profile_sha256=profile_digest(profile),
        dependency_lock_sha256=dependency_lock_checksum,
        source_revision=source_revision,
        python_version=platform.python_version(),
        platform=sys.platform,
        machine=platform.machine(),
        package_versions=_package_versions(
            (
                "agent-evals",
                "pydantic",
                "httpx",
                "anndata",
                "scanpy",
                "scikit-learn",
                "scib-metrics",
            )
        ),
        environment_backend=(environment_record or {}).get("backend"),
        environment_image=(environment_record or {}).get("image"),
        environment_image_digest=(environment_record or {}).get("image_digest"),
        trajectory_sha256=_sha256_json(trajectory_payload),
        environment_events_sha256=_sha256_json(environment_events_payload),
        artifact_manifest_sha256=_sha256_json(artifact_manifest_payload),
        isolation_report_sha256=(
            _sha256_json(isolation_payload) if isolation_payload is not None else None
        ),
        agent_endpoint=agent_endpoint,
        termination_status=agent_run.termination_status.value,
        termination_reason=agent_run.termination_reason,
        step_count=agent_run.step_count,
        limitations=limitations,
    )


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
    "ArchiveVerification",
    "RunProvenance",
    "ScientificActionExecutor",
    "ScientificLoop",
    "verify_archive_manifest",
]
