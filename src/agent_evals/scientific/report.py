"""Structured and human-readable scientific benchmark reports."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.environment.models import ActionRecord, ArtifactRecord
from agent_evals.evaluators.models import MetricResult
from agent_evals.scientific.pipeline import PipelineSpecification


class ScientificBenchmarkReport(BaseModel):
    """Complete reproducible record for one pipeline-first benchmark run."""

    model_config = ConfigDict(extra="forbid")

    report_version: str = "1.0.0"
    benchmark_id: str
    benchmark_title: str
    run_id: str
    dataset_metadata: dict[str, Any]
    pipeline: PipelineSpecification
    trajectory: list[ActionRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    metric_results: list[MetricResult] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    final_score: float | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    errors: list[str] = Field(default_factory=list)

    def to_json(self) -> str:
        """Serialize the report as formatted JSON."""
        return self.model_dump_json(indent=2)

    def to_markdown(self) -> str:
        """Render a concise reproducible report."""
        lines = [
            f"# Scientific Benchmark Report: {self.benchmark_title}",
            "",
            f"- Benchmark: {self.benchmark_id}",
            f"- Run: {self.run_id}",
            f"- Dataset: {self.dataset_metadata.get('source', 'PBMC')} ({self.dataset_metadata.get('cells', '?')} cells x {self.dataset_metadata.get('genes', '?')} genes)",
            f"- Pipeline: {self.pipeline.name}",
            "",
            "## Objective metrics",
            "",
            "| Metric | Status | Value | Evidence |",
            "| --- | --- | ---: | --- |",
        ]
        lines.extend(
            f"| {metric.metric_name} ({metric.metric_id}) | {metric.status.value} | {metric.value if metric.value is not None else '-'} | {'; '.join(metric.evidence) or metric.error or '-'} |"
            for metric in self.metric_results
        )
        lines.extend(
            [
                "",
                f"**Final score:** {self.final_score if self.final_score is not None else 'unavailable'}",
                "",
                "## Executed trajectory",
                "",
                "| Step | Operation | Status | Wall time (s) |",
                "| ---: | --- | --- | ---: |",
            ]
        )
        lines.extend(
            f"| {record.step} | {record.intent.action_id} | {record.result.status.value} | {record.result.resource_usage.wall_time_seconds:.3f} |"
            for record in self.trajectory
        )
        if self.errors:
            lines.extend(["", "## Errors", "", *[f"- {error}" for error in self.errors]])
        return "\n".join(lines) + "\n"

