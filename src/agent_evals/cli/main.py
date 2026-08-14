"""Typer CLI entrypoint application."""

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
import uvicorn
import yaml
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
    agent_adapter_registry,
    agent_runtime_registry,
)
from agent_evals.agents.selection import build_agent_adapter, is_universal_runtime
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.registry import benchmark_registry, benchmark_spec_registry
from agent_evals.benchmarks.schema import BenchmarkSpecification
from agent_evals.cli.environments import env_app
from agent_evals.cli.references import resolve_benchmark_path
from agent_evals.core.config import get_settings
from agent_evals.core.logging import configure_logging, get_logger
from agent_evals.environment.execution import free_execution_action_ids
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.evaluators import EvaluationEngine, EvaluationReport
from agent_evals.research import (
    ResearchCertification,
    StudyArm,
    StudyPlan,
    build_starter_manifest,
    build_study_report,
    default_protocol_fixtures,
    dump_readiness_manifest,
    evaluate_research_readiness,
    load_readiness_manifest,
    run_interoperability_suite,
    run_synthetic_conformance_sync,
    verify_research_certification,
    verify_run_bundle,
)

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
app.add_typer(env_app, name="env")
research_app = typer.Typer(
    name="research",
    help="Validate research-readiness evidence and reproducible study artifacts.",
    add_completion=False,
)
app.add_typer(research_app, name="research")
console = Console()
logger = get_logger("agent_evals.cli")


@research_app.command("init")
def research_init_command(
    benchmark_id: str = typer.Option(..., "--benchmark-id", help="Frozen benchmark identifier."),
    benchmark_version: str = typer.Option(..., "--benchmark-version", help="Frozen benchmark version."),
    output: Path = typer.Option(
        Path("research-readiness.yaml"),
        "--output",
        "-o",
        help="JSON or YAML checklist path to create.",
    ),
) -> None:
    """Create the explicit all-missing research certification checklist."""
    manifest = build_starter_manifest(
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
    )
    dump_readiness_manifest(manifest, output)
    console.print(
        f"[bold yellow]Research checklist created[/bold yellow] {output} "
        f"digest={manifest.canonical_digest()}"
    )


@research_app.command("certify")
def research_certify_command(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        "-m",
        exists=True,
        readable=True,
        help="Research-readiness evidence manifest (JSON or YAML).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional JSON certification output path.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with status 1 unless every required gate is certified.",
    ),
) -> None:
    """Evaluate evidence gates without turning missing evidence into a pass."""
    readiness = load_readiness_manifest(manifest)
    certification = evaluate_research_readiness(readiness, evidence_root=manifest.parent)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(certification.to_json() + "\n", encoding="utf-8")
    table = Table(title="Research Readiness")
    table.add_column("Gate", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Evidence", style="green")
    for gate in certification.gates:
        table.add_row(
            gate.gate.value,
            gate.status.value,
            ", ".join(gate.evidence_ids) or "-",
        )
    console.print(table)
    console.print(
        f"status={certification.status.value} "
        f"readiness={certification.readiness_fraction:.1%} "
        f"manifest={certification.manifest_sha256} "
        f"certificate={certification.certificate_sha256 or 'unavailable'}"
    )
    if certification.blocking_reasons:
        for reason in certification.blocking_reasons:
            console.print(f"  [yellow]{reason}[/yellow]")
    if strict and not certification.research_grade:
        raise typer.Exit(code=1)


@research_app.command("verify")
def research_verify_command(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        "-m",
        exists=True,
        readable=True,
        help="Research-readiness evidence manifest (JSON or YAML).",
    ),
    certificate: Path = typer.Option(
        ...,
        "--certificate",
        "-c",
        exists=True,
        readable=True,
        help="Previously emitted JSON certification.",
    ),
    evidence_root: Path | None = typer.Option(
        None,
        "--evidence-root",
        help="Root for local evidence paths; defaults to the manifest directory.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with status 1 if certificate or manifest integrity fails.",
    ),
) -> None:
    """Recompute a certificate and its evidence hashes without network access."""
    readiness = load_readiness_manifest(manifest)
    try:
        parsed_certificate = ResearchCertification.model_validate_json(
            certificate.read_text(encoding="utf-8")
        )
    except Exception as error:
        console.print(f"[red]INVALID certificate[/red] {certificate}: {error}")
        raise typer.Exit(code=1) from error
    integrity = verify_research_certification(
        parsed_certificate,
        manifest=readiness,
        evidence_root=evidence_root or manifest.parent,
    )
    console.print(
        f"certificate={'VALID' if integrity.valid else 'INVALID'} "
        f"status={parsed_certificate.status.value} "
        f"research_grade={parsed_certificate.research_grade} "
        f"digest={'ok' if integrity.certificate_digest_matches else 'mismatch'}"
    )
    for item in integrity.evidence:
        console.print(f"  evidence {item.evidence_id}: {item.status}")
    for limitation in integrity.limitations:
        console.print(f"  [yellow]{limitation}[/yellow]")
    if strict and not integrity.valid:
        raise typer.Exit(code=1)


@research_app.command("protocol-check")
def research_protocol_check_command(
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with status 1 if the offline interoperability suite fails.",
    ),
) -> None:
    """Run provider-neutral structured/text/opaque-agent protocol fixtures."""
    report = run_interoperability_suite(
        default_protocol_fixtures(),
        opaque_multi_agent_fixture_ids={"black-box-text-action"},
    )
    for fixture in report.fixtures:
        status = "PASS" if fixture.passed else "FAIL"
        console.print(f"{status} {fixture.fixture_id}")
        for finding in fixture.findings:
            console.print(f"  {finding}")
    console.print(f"protocol_suite={'PASS' if report.passed else 'INCOMPLETE'}")
    if strict and not report.passed:
        raise typer.Exit(code=1)


@research_app.command("conformance")
def research_conformance_command(
    output: Path = typer.Option(
        Path("research") / "synthetic-conformance",
        "--output",
        "-o",
        help="Directory for the synthetic run bundle.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with status 1 when any conformance check fails.",
    ),
) -> None:
    """Run the dataset-independent endpoint-to-bundle conformance fixture."""
    report = run_synthetic_conformance_sync(output)
    status = "PASS" if report.passed else "FAIL"
    console.print(
        f"{status} synthetic-conformance run={report.run_id} "
        f"bundle={report.bundle_path}"
    )
    for check, passed in report.checks.items():
        console.print(f"  {'PASS' if passed else 'FAIL'} {check}")
    for limitation in report.limitations:
        console.print(f"  [yellow]limitation: {limitation}[/yellow]")
    if strict and not report.passed:
        raise typer.Exit(code=1)


@research_app.command("stats")
def research_stats_command(
    plan: Path = typer.Option(
        ...,
        "--plan",
        exists=True,
        readable=True,
        help="StudyPlan JSON or YAML.",
    ),
    arms: Path = typer.Option(
        ...,
        "--arms",
        exists=True,
        readable=True,
        help="JSON/YAML list of StudyArm records.",
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Study report JSON path."),
) -> None:
    """Generate deterministic arm summaries, paired CIs, and ablation results."""
    plan_payload = _load_structured_file(plan)
    arms_payload = _load_structured_file(arms)
    plan_model = StudyPlan.model_validate(plan_payload)
    raw_arms = arms_payload.get("arms") if isinstance(arms_payload, dict) else arms_payload
    if not isinstance(raw_arms, list):
        raise typer.BadParameter("--arms must contain a list or an 'arms' list")
    arm_models = [StudyArm.model_validate(item) for item in raw_arms]
    report = build_study_report(plan_model, arm_models)
    target = output or Path("research") / f"{plan_model.study_id}.statistics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(
        f"[bold green]Study report written[/bold green] {target} "
        f"ready={report.research_ready} comparisons={len(report.statistics.comparisons)}"
    )


def _load_structured_file(path: Path) -> Any:
    """Load JSON or YAML for research protocol artifacts."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)
    if payload is None:
        raise typer.BadParameter(f"structured file '{path}' is empty")
    return payload


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
        None,
        "--model",
        help="Model identifier for the selected --agent runtime; see 'agent-evals agent list'.",
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="Optional model provider prefix when --model has no provider."
    ),
    agent_endpoint: str | None = typer.Option(
        None,
        "--agent-endpoint",
        help="URL of a black-box agent endpoint when --agent http-step is selected.",
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
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        min=1,
        help="Maximum agent-environment interaction steps.",
    ),
    environment: str | None = typer.Option(
        None,
        "--environment",
        "-e",
        help=(
            "Declared environment ID to run the agent's own code in; defaults to "
            "the task's own choice. See 'agent-evals env inspect'."
        ),
    ),
    research_manifest: Path | None = typer.Option(
        None,
        "--research-manifest",
        help="Attach and validate a benchmark-wide research-readiness manifest.",
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
    if agent == "rule-based" or is_universal_runtime(agent):
        from agent_evals.environment.scientific_loop import (
            DEFAULT_RUNTIME_MAX_STEPS,
            ScientificLoop,
        )

        runtime_max_steps = (
            DEFAULT_RUNTIME_MAX_STEPS
            if max_steps is None and is_universal_runtime(agent)
            else max_steps
        )

        scientific_run = asyncio.run(
            ScientificLoop().run(
                benchmark_reference,
                agent_type=agent,
                output_dir=output_dir,
                seed=seed,
                max_cells=max_cells,
                task_id=task,
                dataset_id=dataset,
                max_steps=runtime_max_steps,
                model=model,
                provider=provider,
                agent_endpoint=agent_endpoint,
                environment=environment,
                research_manifest=research_manifest,
            )
        )
        console.print(
            f"[bold green]Scientific agent run {scientific_run.run_id}[/bold green] "
            f"status={scientific_run.agent_run.termination_status.value} "
            f"final_score={scientific_run.evaluation.global_agent_score if scientific_run.evaluation and scientific_run.evaluation.global_agent_score is not None else 'unavailable'} "
            f"report={scientific_run.report_path}"
        )
        return
    if environment is not None:
        # Refused rather than ignored. This path executes through
        # ``MockActionExecutor``, which touches no workspace at all, so accepting
        # the flag would report a run against an environment that was never
        # provisioned -- and a free-execution benchmark would come back with the
        # mock's synthetic artifacts as though the agent had produced them.
        console.print(
            f"[red]--environment is not supported by the '{agent}' adapter[/red]; "
            "it runs against a mock executor and provisions no workspace. Use "
            "--agent rule-based or a universal runtime ('agent-evals agent list')."
        )
        raise typer.Exit(code=2)
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
            max_steps=max_steps,
            output=run_output,
            mock_policy=mock_policy,
        )
    )
    specification = load_benchmark(resolve_benchmark_path(benchmark_reference))
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
    max_steps: int | None,
    output: Path | None,
    mock_policy: str | None,
) -> AgentRun:
    """Resolve a benchmark, execute an adapter, and persist its AgentRun."""
    path = resolve_benchmark_path(benchmark_reference)
    specification = load_benchmark(path)
    resolved_task = task_id or specification.tasks[0].id
    adapter: AgentAdapter = build_agent_adapter(agent_type, model=model)
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
        max_steps=max_steps,
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
    specification = load_benchmark(resolve_benchmark_path(benchmark))
    evaluation = EvaluationEngine().evaluate(specification, run)
    output_format = output.lower()
    if output_format not in {"json", "markdown"}:
        raise typer.BadParameter("output must be 'json' or 'markdown'")
    target = report or Path("runs") / f"{run.run_id}.evaluation.{output_format}"
    _write_report(evaluation, target, output_format)
    console.print(f"[bold green]Evaluation report:[/bold green] {target}")


@app.command("verify-run")
def verify_run_command(
    run: Path = typer.Argument(..., exists=True, file_okay=False, help="Materialized run archive directory."),
) -> None:
    """Re-hash a persisted run archive and fail if its public bytes drifted."""
    from agent_evals.environment.scientific_loop import verify_archive_manifest

    verification = verify_archive_manifest(run)
    if verification.valid:
        console.print(
            f"[bold green]VALID[/bold green] {run} "
            f"({verification.checked_files} public files checked)"
        )
        return
    console.print(f"[bold red]INVALID[/bold red] {run}")
    if verification.missing_files:
        console.print(f"  missing: {', '.join(verification.missing_files)}")
    if verification.changed_files:
        console.print(f"  changed: {', '.join(verification.changed_files)}")
    if verification.unexpected_files:
        console.print(f"  unexpected: {', '.join(verification.unexpected_files)}")
    for limitation in verification.limitations:
        console.print(f"  limitation: {limitation}")
    raise typer.Exit(code=1)


@app.command("verify-bundle")
def verify_bundle_command(
    bundle: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Materialized public run bundle directory.",
    ),
    strict_replay: bool = typer.Option(
        False,
        "--strict-replay",
        help="Also require a valid replay descriptor and referenced public files.",
    ),
) -> None:
    """Verify the replay-oriented event ledger and content-addressed bundle."""
    verification = verify_run_bundle(bundle)
    if verification.valid:
        console.print(
            f"[bold green]VALID[/bold green] {bundle} "
            f"({verification.checked_files} files, event ledger valid, "
            f"replay_ready={verification.replay_ready})"
        )
        for limitation in verification.replay_limitations:
            console.print(f"  replay limitation: {limitation}")
        if strict_replay and not verification.replay_ready:
            raise typer.Exit(code=1)
        return
    console.print(f"[bold red]INVALID[/bold red] {bundle}")
    if verification.missing_files:
        console.print(f"  missing: {', '.join(verification.missing_files)}")
    if verification.changed_files:
        console.print(f"  changed: {', '.join(verification.changed_files)}")
    if verification.unexpected_files:
        console.print(f"  unexpected: {', '.join(verification.unexpected_files)}")
    for limitation in verification.limitations:
        console.print(f"  limitation: {limitation}")
    for limitation in verification.replay_limitations:
        console.print(f"  replay limitation: {limitation}")
    raise typer.Exit(code=1)


@app.command("validate-benchmark")
def validate_benchmark_command(
    benchmark: str = typer.Option(
        ..., "--benchmark", "-b", help="Benchmark YAML path or registered ID."
    ),
) -> None:
    """Check a benchmark YAML without running it, and summarize what it declares.

    ``load_benchmark`` already runs the full integrity check -- cross-references,
    artifact contracts, environment/task consistency, scoring weights, cutoff
    coherence -- so this command adds no validation of its own. What it adds is a
    way to *reach* that check, which until now required starting a run.
    """
    path = resolve_benchmark_path(benchmark)
    if not path.exists():
        console.print(f"[red]Benchmark not found:[/red] {benchmark}")
        raise typer.Exit(code=2)
    try:
        specification = load_benchmark(path)
    except Exception as error:
        console.print(f"[red]INVALID[/red] {path}")
        console.print(f"  {error}")
        raise typer.Exit(code=1) from error
    console.print(f"[bold green]VALID[/bold green] {path}")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="dim")
    table.add_column("Value")
    table.add_row("id", specification.metadata.id)
    table.add_row("version", specification.metadata.version)
    table.add_row("tasks", ", ".join(task.id for task in specification.tasks))
    table.add_row("actions", str(len(specification.actions)))
    table.add_row("metrics", str(len(specification.metrics)))
    table.add_row("artifacts", str(len(specification.artifacts)))
    table.add_row(
        "environments",
        ", ".join(spec.id for spec in specification.environments) or "(none)",
    )
    free_actions = sorted(free_execution_action_ids(specification))
    table.add_row("free-execution actions", ", ".join(free_actions) or "(none)")
    weights = specification.scoring
    table.add_row(
        "score weights",
        f"outcome={weights.outcome_weight:.3f} decision={weights.decision_weight:.3f} "
        f"trajectory={weights.trajectory_weight:.3f}",
    )
    table.add_row("cutoff", _cutoff_summary(specification))
    console.print(table)


def _cutoff_summary(specification: BenchmarkSpecification) -> str:
    """Render the declared live budget, naming what is left unbounded.

    An undeclared budget is reported rather than omitted: a cutoff that never
    fires is the failure mode this project treats as worse than one never
    declared, so the CLI has to be able to show its absence.
    """
    cutoff = specification.cutoff
    declared = {
        "max_steps": cutoff.max_steps,
        "wall_time_s": cutoff.max_wall_time_seconds,
        "tokens": cutoff.max_total_tokens,
        "cost_usd": cutoff.max_cost_usd,
        "consecutive_failures": cutoff.max_consecutive_failures,
        "repeated_decisions": cutoff.max_repeated_decisions,
        "stagnation_window": cutoff.stagnation_window,
    }
    present = [f"{name}={value}" for name, value in declared.items() if value is not None]
    absent = [name for name, value in declared.items() if value is None]
    summary = ", ".join(present) or "(nothing declared)"
    if absent:
        summary += f" | unbounded: {', '.join(absent)}"
    return summary


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


@app.command("worker-health")
def worker_health_command() -> None:
    """Exit successfully only while the durable worker lease is alive."""
    from agent_evals.api.job_store import JobStoreError, SQLiteJobStore

    store: SQLiteJobStore | None = None
    try:
        store = SQLiteJobStore(get_settings().storage.job_db_path)
        store.ping()
        if not store.worker_lease_active():
            raise JobStoreError("no active scientific worker lease")
    except JobStoreError as error:
        console.print(f"[red]Worker unhealthy:[/red] {error}")
        raise typer.Exit(code=1) from error
    finally:
        if store is not None:
            store.close()
    console.print("[green]Worker healthy[/green]")


@app.command("worker")
def worker_command() -> None:
    """Run the durable evaluation worker without serving HTTP traffic."""
    from agent_evals.api.routes import job_manager

    async def run_worker() -> None:
        manager = job_manager
        await manager.start(execute_jobs=True)
        console.print(
            "[bold green]Evaluation worker is running[/bold green] "
            f"store={get_settings().storage.job_db_path}"
        )
        try:
            await asyncio.Event().wait()
        finally:
            await manager.shutdown()
            manager.close()

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        console.print("[yellow]Evaluation worker stopped[/yellow]")


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
    workers: int | None = typer.Option(
        None,
        "--workers",
        min=1,
        max=32,
        help="Number of Uvicorn worker processes (ignored when --reload is enabled).",
    ),
) -> None:
    """Launch the FastAPI backend server."""
    settings = get_settings()
    server_host = host or settings.api.host
    server_port = port or settings.api.port
    server_workers = 1 if reload else (workers or settings.api.workers)
    if server_workers > 1 and settings.api.execute_jobs_in_process:
        raise typer.BadParameter(
            "multiple API workers require "
            "AGENT_EVALS_API__EXECUTE_JOBS_IN_PROCESS=false and a dedicated "
            "'agent-evals worker' process"
        )

    console.print(
        f"[bold green]Launching API server on[/bold green] http://{server_host}:{server_port}"
    )
    uvicorn.run(
        "agent_evals.api.main:app",
        host=server_host,
        port=server_port,
        reload=reload,
        workers=server_workers,
    )


if __name__ == "__main__":
    app()
