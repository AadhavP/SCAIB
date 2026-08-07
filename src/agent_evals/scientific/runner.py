"""Pipeline-first scientific benchmark runner."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agent_evals.benchmarks.registry import benchmark_spec_registry
from agent_evals.benchmarks.schema import BenchmarkSpecification
from agent_evals.datasets.pbmc import PBMCDataset
from agent_evals.environment.models import ActionIntent, ActionRecord
from agent_evals.evaluators.models import MetricResult
from agent_evals.scientific.artifacts.storage import LocalArtifactStore
from agent_evals.scientific.benchmarks import register_scientific_benchmarks
from agent_evals.scientific.context import ScientificContext
from agent_evals.scientific.executor.scanpy import ScanpyExecutor
from agent_evals.scientific.metrics import compute_objective_metrics
from agent_evals.scientific.pipeline import PipelineSpecification, load_pipeline
from agent_evals.scientific.report import ScientificBenchmarkReport


def _final_score(benchmark_id: str, metric_results: list[MetricResult]) -> float | None:
    """Apply the benchmark's declared objective aggregation when evidence exists."""
    values = {
        metric.metric_id: float(metric.normalized_score)
        for metric in metric_results
        if metric.normalized_score is not None and metric.status.value == "succeeded"
    }
    if "cell-annotation" in benchmark_id:
        required: tuple[str, ...] = ("annotation_ari", "annotation_nmi", "annotation_silhouette")
        return sum(values[key] * weight for key, weight in zip(required, (0.4, 0.4, 0.2), strict=True)) if all(key in values for key in required) else None
    if "batch-correction" in benchmark_id:
        required = ("batch_silhouette", "cell_type_silhouette")
        return sum(values[key] for key in required) / 2 if all(key in values for key in required) else None
    if "differential-expression" in benchmark_id:
        required = ("marker_overlap", "precision_at_k", "auroc")
        return sum(values[key] * weight for key, weight in zip(required, (0.4, 0.3, 0.3), strict=True)) if all(key in values for key in required) else None
    return None


class ScientificPipelineRunner:
    """Execute a declarative pipeline against real PBMC AnnData."""

    def __init__(self, *, cache_dir: Path | str = Path(".cache/datasets")) -> None:
        self.cache_dir = Path(cache_dir)

    def run(
        self,
        benchmark: str | Path,
        pipeline: str | Path | PipelineSpecification,
        *,
        output_dir: Path | str = Path("results"),
        max_cells: int | None = None,
    ) -> ScientificBenchmarkReport:
        """Run, score, and persist one scientific benchmark report."""
        specification = self._resolve_benchmark(benchmark)
        pipeline_spec = load_pipeline(pipeline) if not isinstance(pipeline, PipelineSpecification) else pipeline
        run_id = str(uuid4())
        run_root = Path(output_dir) / run_id
        store = LocalArtifactStore(run_root / "artifacts")
        dataset = PBMCDataset(cache_dir=self.cache_dir)
        requested_cells = max_cells or pipeline_spec.dataset.get("max_cells")
        adata = dataset.load(max_cells=int(requested_cells) if requested_cells else None)
        context = ScientificContext(
            adata=adata,
            dataset_metadata=dataset.metadata.model_dump(),
            artifact_store=store,
            workspace=run_root,
            metadata={"benchmark_version": specification.metadata.version},
        )
        executor = ScanpyExecutor()
        records: list[ActionRecord] = []
        errors: list[str] = []
        started_at = datetime.now(UTC)
        for step_number, step in enumerate(pipeline_spec.steps, start=1):
            intent = ActionIntent(
                action_id=step.operation,
                parameters=step.parameters,
                metadata={"pipeline": pipeline_spec.name, "pipeline_step": step_number},
            )
            result = executor.execute(intent, context)
            records.append(ActionRecord(step=step_number, intent=intent, result=result))
            if result.error:
                errors.append(f"step {step_number} ({step.operation}): {result.error}")
                break
        objective_parameters = {
            key: value
            for step in pipeline_spec.steps
            for key, value in step.parameters.items()
            if key in {"reference_markers", "top_k"}
        }
        metric_results = compute_objective_metrics(specification.metadata.id, context.adata, objective_parameters)
        final_score = _final_score(specification.metadata.id, metric_results)
        report = ScientificBenchmarkReport(
            benchmark_id=specification.metadata.id,
            benchmark_title=specification.metadata.title,
            run_id=run_id,
            dataset_metadata=dataset.metadata.model_dump(),
            pipeline=pipeline_spec,
            trajectory=records,
            artifacts=[artifact.to_artifact_record() for artifact in context.artifacts.values()],
            metric_results=metric_results,
            scores={
                metric.metric_id: float(metric.normalized_score)
                for metric in metric_results
                if metric.normalized_score is not None
            },
            final_score=final_score,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            errors=errors,
        )
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "pipeline.json").write_text(pipeline_spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (run_root / "trajectory.json").write_text(json.dumps([record.model_dump(mode="json") for record in records], indent=2) + "\n", encoding="utf-8")
        (run_root / "metrics.json").write_text(json.dumps([metric.model_dump(mode="json") for metric in metric_results], indent=2, default=str) + "\n", encoding="utf-8")
        (run_root / "report.json").write_text(report.to_json(), encoding="utf-8")
        (run_root / "report.md").write_text(report.to_markdown(), encoding="utf-8")
        return report

    @staticmethod
    def _resolve_benchmark(reference: str | Path) -> BenchmarkSpecification:
        path = Path(reference)
        if path.exists():
            from agent_evals.benchmarks.io import load_benchmark

            return load_benchmark(path)
        if not benchmark_spec_registry.list_ids():
            register_scientific_benchmarks()
        return benchmark_spec_registry.get(str(reference))
