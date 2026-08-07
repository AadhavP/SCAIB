"""Report generator for formatting evaluation results."""

from pathlib import Path

from agent_evals.core.types import EvaluationResult


class ReportGenerator:
    """Generates formatted evaluation reports in JSON, Markdown, or HTML formats."""

    def __init__(self, output_dir: Path = Path("./reports_output")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json(self, result: EvaluationResult, filename: str) -> Path:
        """Export result object as formatted JSON report file."""
        target = self.output_dir / f"{filename}.json"
        with open(target, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        return target

    def generate_markdown(self, result: EvaluationResult, filename: str) -> Path:
        """Export result object as Markdown summary report file."""
        target = self.output_dir / f"{filename}.md"
        lines = [
            f"# Evaluation Report: {result.benchmark_id}",
            "",
            f"- **Agent ID**: {result.agent_id}",
            f"- **Status**: {result.status.value}",
            f"- **Execution Time**: {result.execution_time_seconds:.2f}s",
            "",
            "## Scores",
            "",
            "| Metric | Value | Unit |",
            "| --- | --- | --- |",
        ]
        for score in result.scores:
            unit_str = score.unit if score.unit else "-"
            lines.append(f"| {score.name} | {score.value:.4f} | {unit_str} |")

        if result.error_message:
            lines.extend(
                ["", "## Error Details", "", f"```\n{result.error_message}\n```"]
            )

        with open(target, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return target
