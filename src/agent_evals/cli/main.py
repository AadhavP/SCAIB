"""Typer CLI entrypoint application."""

import asyncio
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from agent_evals import __version__
from agent_evals.agents import (
    AgentAdapter,
    AgentConfiguration,
    AgentHarness,
    AgentRun,
    MockActionExecutor,
    MockObservationBuilder,
    RuntimeAgentAdapter,
    agent_adapter_registry,
    agent_runtime_registry,
)
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.registry import benchmark_registry, benchmark_spec_registry
from agent_evals.core.config import get_settings
from agent_evals.core.logging import configure_logging, get_logger
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.evaluators import EvaluationEngine, EvaluationReport

app = typer.Typer(
    name="agent-evals",
    help="Autonomous AI agent evaluation suite for computational single-cell biology.",
    add_completion=False,
)
benchmark_app = typer.Typer(
    name="benchmark",
    help="Run concrete scientific benchmark pipelines.",
    add_completion=False,
)
app.add_typer(benchmark_app, name="benchmark")
agent_app = typer.Typer(
    name="agent",
    help="Inspect and configure universal agent runtimes.",
    add_completion=False,
)
app.add_typer(agent_app, name="agent")
console = Console()
logger = get_logger("agent_evals.cli")


@benchmark_app.command("run")
def scientific_benchmark_run_command(
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="Benchmark YAML path or registered benchmark ID."),
    pipeline: Path = typer.Option(..., "--pipeline", "-p", exists=True, readable=True, help="Scientific pipeline YAML path."),
    output_dir: Path = typer.Option(Path("results"), "--output-dir", help="Directory for reproducible run artifacts and reports."),
    max_cells: int | None = typer.Option(None, "--max-cells", min=1, help="Optional deterministic prefix subset for local runs."),
) -> None:
    """Execute a real PBMC/Scanpy pipeline and write a scientific report."""
    from agent_evals.scientific.runner import ScientificPipelineRunner

    report = ScientificPipelineRunner().run(
        benchmark,
        pipeline,
        output_dir=output_dir,
        max_cells=max_cells,
    )
    console.print(
        f"[bold green]Scientific run {report.run_id}[/bold green] "
        f"benchmark={report.benchmark_id} score={report.final_score if report.final_score is not None else 'unavailable'} "
        f"report={output_dir / report.run_id / 'report.md'}"
    )


@app.callback()
def common_callback(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose debug logging."
    ),
) -> None:
    """CLI common option initialization callback."""
    settings = get_settings(config)
    log_level = "DEBUG" if verbose else settings.log_level
    configure_logging(log_level=log_level, json_format=settings.log_json)


@app.command("version")
def version_command() -> None:
    """Print agent-evals framework version."""
    console.print(
        f"[bold green]agent-evals[/bold green] version: [cyan]{__version__}[/cyan]"
    )


@app.command("list-benchmarks")
def list_benchmarks_command() -> None:
    """List all registered benchmark IDs."""
    ids = benchmark_registry.list_ids()
    if not ids:
        example_root = Path("examples/benchmarks")
        if example_root.exists():
            benchmark_spec_registry.discover(example_root, replace=True)
            ids = benchmark_spec_registry.list_ids()
    table = Table(title="Registered Benchmarks")
    table.add_column("Benchmark ID", style="cyan")
    if not ids:
        table.add_row("(No benchmarks registered yet)")
    else:
        for b_id in ids:
            table.add_row(b_id)
    console.print(table)


@app.command("list-agents")
def list_agents_command() -> None:
    """List all registered agent adapter types."""
    types = agent_adapter_registry.list_types()
    table = Table(title="Registered Agent Adapters")
    table.add_column("Agent Adapter Type", style="magenta")
    table.add_column("Availability", style="green")
    if not types:
        table.add_row("(No agent adapters registered yet)")
    else:
        availability = agent_adapter_registry.availability()
        for a_type in types:
            table.add_row(a_type, "available" if availability[a_type] else "unavailable")
    console.print(table)


@agent_app.command("list")
def agent_list_command() -> None:
    """List universal runtimes and their declared capabilities."""
    table = Table(title="Available Agents")
    table.add_column("Agent Runtime", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Capabilities", style="green")
    for name in agent_runtime_registry.list():
        manifest = agent_runtime_registry.manifest(name)
        table.add_row(name, manifest.type, ", ".join(manifest.capabilities) or "-")
    if not agent_runtime_registry.list():
        table.add_row("(No universal runtimes registered)", "-", "-")
    console.print(table)


@app.command("run")
def run_command(
    config: Path = typer.Option(
        Path("configs/benchmark_config.yaml"),
        "--config",
        "-c",
        help="Path to benchmark execution YAML specification.",
    ),
    benchmark: str | None = typer.Option(
        None,
        "--benchmark",
        "-b",
        help="Benchmark ID or path to a benchmark YAML specification.",
    ),
    agent: str = typer.Option("mock", "--agent", "-a", help="Harness adapter type."),
    model: str | None = typer.Option(
        None, "--model", help="OpenHands/Litellm model identifier, e.g. anthropic/claude-sonnet-4-5-20250929."
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="Optional model provider prefix when --model has no provider."
    ),
    workspace: Path | None = typer.Option(
        None, "--workspace", help="Controlled workspace root for the agent run."
    ),
    task: str | None = typer.Option(None, "--task", help="Task ID; defaults to the first task."),
    dataset: str | None = typer.Option(None, "--dataset", help="Dataset ID to select."),
    seed: int = typer.Option(0, "--seed", help="Deterministic episode seed."),
    mock_policy: str | None = typer.Option(
        None, "--mock-policy", help="Mock policy: single, good, or bad."
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="AgentRun path, or report format ('json'/'markdown').",
    ),
    report: Path | None = typer.Option(None, "--report", help="Evaluation report output path."),
    output_dir: Path = typer.Option(
        Path("results"),
        "--output-dir",
        help="Directory for agent-scientific run artifacts.",
    ),
    max_cells: int | None = typer.Option(
        None,
        "--max-cells",
        min=1,
        help="Optional deterministic PBMC prefix subset for local runs.",
    ),
) -> None:
    """Execute one benchmark task through a framework-neutral harness."""
    benchmark_reference = benchmark or "examples/benchmarks/pbmc-cell-annotation.yaml"
    console.print(f"[bold yellow]Running benchmark:[/bold yellow] {benchmark_reference}")
    logger.info(
        "Executing benchmark run command",
        config_path=str(config),
        benchmark=benchmark_reference,
        agent=agent,
    )
    if agent == "rule-based":
        from agent_evals.environment.scientific_loop import ScientificLoop

        scientific_run = asyncio.run(
            ScientificLoop().run(
                benchmark_reference,
                agent_type=agent,
                output_dir=output_dir,
                seed=seed,
                max_cells=max_cells,
            )
        )
        console.print(
            f"[bold green]Scientific agent run {scientific_run.run_id}[/bold green] "
            f"status={scientific_run.agent_run.termination_status.value} "
            f"final_score={scientific_run.evaluation.global_agent_score if scientific_run.evaluation and scientific_run.evaluation.global_agent_score is not None else 'unavailable'} "
            f"report={scientific_run.report_path}"
        )
        return
    output_format = output if output in {"json", "markdown"} else "json"
    run_output = None if output in {"json", "markdown"} else Path(output) if output else None
    run = asyncio.run(
        _run_benchmark(
            benchmark_reference=benchmark_reference,
            agent_type=agent,
            model=model,
            provider=provider,
            workspace=workspace,
            task_id=task,
            dataset_id=dataset,
            seed=seed,
            output=run_output,
            mock_policy=mock_policy,
        )
    )
    specification = load_benchmark(_resolve_benchmark_path(benchmark_reference))
    evaluation = EvaluationEngine().evaluate(specification, run)
    report_target = report or Path("runs") / f"{run.run_id}.evaluation.{output_format}"
    _write_report(evaluation, report_target, output_format)
    console.print(
        f"[bold green]Run {run.run_id}[/bold green] "
        f"status={run.termination_status.value} steps={run.step_count} "
        f"metrics={len(evaluation.metric_results)} report={report_target}"
    )


async def _run_benchmark(
    *,
    benchmark_reference: str,
    agent_type: str,
    model: str | None,
    provider: str | None,
    workspace: Path | None,
    task_id: str | None,
    dataset_id: str | None,
    seed: int,
    output: Path | None,
    mock_policy: str | None,
) -> AgentRun:
    """Resolve a benchmark, execute an adapter, and persist its AgentRun."""
    path = _resolve_benchmark_path(benchmark_reference)
    specification = load_benchmark(path)
    resolved_task = task_id or specification.tasks[0].id
    adapter: AgentAdapter
    if agent_type in agent_runtime_registry.list():
        runtime_config: dict[str, object] = {}
        if model is not None and agent_type not in {"gpt-5", "claude-sonnet"}:
            runtime_config["model"] = model
        adapter = RuntimeAgentAdapter(agent_runtime_registry.create(agent_type, **runtime_config))
    else:
        adapter = agent_adapter_registry.create(agent_type)
    environment = ScientificEnvironment(
        specification,
        task_id=resolved_task,
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )
    configuration = AgentConfiguration(
        agent_type=agent_type,
        model=model,
        provider=provider,
        seed=seed,
        workspace={"root": str(workspace)} if workspace else {},
        metadata={
            key: value
            for key, value in {
                "dataset_id": dataset_id,
                "mock_policy": (
                    (mock_policy or "good") if agent_type == "mock" else None
                ),
            }.items()
            if value is not None
        },
    )
    run = await AgentHarness().run(adapter, environment, configuration)
    target = output or Path("runs") / f"{run.run_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(run.to_json(), encoding="utf-8")
    return run


@app.command("evaluate")
def evaluate_command(
    run_id: str = typer.Argument(..., help="AgentRun JSON path or run identifier."),
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="Benchmark YAML path or ID."),
    output: str = typer.Option("json", "--output", "-o", help="Report format: json or markdown."),
    report: Path | None = typer.Option(None, "--report", help="Evaluation report output path."),
) -> None:
    """Evaluate an existing persisted AgentRun without rerunning the agent."""
    run_path = Path(run_id)
    if not run_path.exists():
        run_path = Path("runs") / f"{run_id}.json"
    run = AgentRun.from_json(run_path.read_text(encoding="utf-8"))
    specification = load_benchmark(_resolve_benchmark_path(benchmark))
    evaluation = EvaluationEngine().evaluate(specification, run)
    output_format = output.lower()
    if output_format not in {"json", "markdown"}:
        raise typer.BadParameter("output must be 'json' or 'markdown'")
    target = report or Path("runs") / f"{run.run_id}.evaluation.{output_format}"
    _write_report(evaluation, target, output_format)
    console.print(f"[bold green]Evaluation report:[/bold green] {target}")


def _resolve_benchmark_path(reference: str) -> Path:
    path = Path(reference)
    if path.exists():
        return path
    return Path("examples/benchmarks") / f"{reference}.yaml"


def _write_report(report: EvaluationReport, target: Path, output_format: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        target.write_text(report.to_json(), encoding="utf-8")
        return
    target.write_text(_report_markdown(report), encoding="utf-8")


def _report_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# Evaluation Report: {report.benchmark_id}",
        "",
        f"- Task: `{report.task_id}`",
        f"- Agent: `{report.agent_id}`",
        f"- Run: `{report.run_id}`",
        "",
        "## Decision evaluations",
        "",
        "| Decision | Level | Valid | Score |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {item.decision_id} | {item.level.value} | {item.valid} | {item.score:.3f} |"
        for item in report.decision_evaluations
    )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Level | Status | Raw value | Normalized |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {item.metric_id} | {item.level.value} | {item.status.value} | "
        f"{item.raw_value if item.raw_value is not None else '-'} | "
        f"{item.normalized_score if item.normalized_score is not None else '-'} |"
        for item in report.metric_results
    )
    return "\n".join(lines) + "\n"


@app.command("serve")
def serve_command(
    host: str | None = typer.Option(
        None, "--host", "-h", help="Bind host address for REST API server."
    ),
    port: int | None = typer.Option(
        None, "--port", "-p", help="Bind port for REST API server."
    ),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload for development."
    ),
) -> None:
    """Launch the FastAPI backend server."""
    settings = get_settings()
    server_host = host or settings.api.host
    server_port = port or settings.api.port

    console.print(
        f"[bold green]Launching API server on[/bold green] http://{server_host}:{server_port}"
    )
    uvicorn.run(
        "agent_evals.api.main:app",
        host=server_host,
        port=server_port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
