"""Commands for inspecting the execution environments a benchmark asks for.

These read a benchmark and report what it declares; they run nothing. The point
is to answer, before a paid run starts, two questions a YAML file cannot answer
by itself:

- Which execution tiers can this host actually provide right now?
- Of the isolation a benchmark asks for, how much would this host really impose?

The second is the one that matters. A benchmark declaring ``internet_access:
false`` and ``max_memory_mb: 16384`` looks equally confident on every platform,
and on Windows neither is enforceable through the local tier. Reporting that
ahead of time is the difference between knowing a run was unconfined and
discovering it in the run record afterwards -- or not at all.
"""

from __future__ import annotations

import shutil
import sys

import typer
from rich.console import Console
from rich.table import Table

from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.schema import (
    BenchmarkSpecification,
    ConstraintSpecification,
    EnvironmentBackend,
    EnvironmentSpecification,
)
from agent_evals.cli.references import resolve_benchmark_path
from agent_evals.environment.execution import (
    IsolationOutcome,
    docker_runtime_available,
    free_execution_action_ids,
    isolation_from_constraints,
    local_isolation_controls,
)

console = Console()

env_app = typer.Typer(
    name="env",
    help="Inspect and validate the execution environments a benchmark declares.",
    add_completion=False,
)

#: The container tier drives this binary over ``subprocess``; without it on
#: PATH a benchmark asking for that tier cannot be honoured on this host.
CONTAINER_BINARY = "docker"


def _container_available() -> bool:
    """Whether the container tier could be provided on this host."""
    return docker_runtime_available(CONTAINER_BINARY)


def _environment_status(environment: EnvironmentSpecification) -> str:
    """Return the host status for one declared environment, including its image."""
    if environment.backend is EnvironmentBackend.LOCAL:
        return "available"
    if shutil.which(CONTAINER_BINARY) is None:
        return f"unavailable ('{CONTAINER_BINARY}' not on PATH)"
    if not docker_runtime_available(CONTAINER_BINARY):
        return "unavailable (Docker daemon is stopped or inaccessible)"
    if environment.image and not docker_runtime_available(
        CONTAINER_BINARY, image=environment.image
    ):
        return f"unavailable (image '{environment.image}' is not available)"
    return "available"


def backend_availability() -> dict[EnvironmentBackend, str]:
    """Report, per backend, whether this host can provide it and why not."""
    return {
        EnvironmentBackend.LOCAL: "available",
        EnvironmentBackend.CONTAINER: (
            "available"
            if _container_available()
            else (
                f"unavailable ('{CONTAINER_BINARY}' not on PATH)"
                if shutil.which(CONTAINER_BINARY) is None
                else "unavailable (Docker daemon is stopped or inaccessible)"
            )
        ),
    }


@env_app.command("list")
def env_list_command() -> None:
    """List the execution tiers SCAIB implements and their host availability."""
    table = Table(title="Execution Backends")
    table.add_column("Backend", style="cyan")
    table.add_column("Host status")
    table.add_column("Isolation this host can impose")
    availability = backend_availability()
    for backend, status in availability.items():
        table.add_row(backend.value, status, _backend_isolation_summary(backend))
    console.print(table)
    console.print(f"Platform: [bold]{sys.platform}[/bold]")


#: The probe used to ask the local tier what it would enforce. Every control the
#: tier has a mechanism for is requested, because the column reports what this
#: host *can* impose and a control nobody asked for answers ``not_requested`` --
#: which would drop it from both halves of the summary and read as no gap. The
#: values do not reach the verdict; only their presence does.
_SUMMARY_PROBE = ConstraintSpecification(
    internet_access=False,
    max_memory_mb=1024,
    max_runtime_seconds=60,
)


def _backend_isolation_summary(backend: EnvironmentBackend) -> str:
    """Summarize which controls a backend can impose on this host.

    Both halves are reported, present *and* missing, because either alone is
    misread: a list of guarantees implies the rest were not asked for, and a
    list of gaps implies everything else held.

    Derived from :func:`local_isolation_controls` rather than assembled here.
    This function used to hand-list ``filesystem_scope`` among the controls the
    local tier imposes, while the tier itself reports it ``unenforceable`` on
    every host and says why -- a local process keeps the host write permissions
    of the user running the benchmark. So the pre-run summary promised
    confinement that no run ever had, and nothing could notice, because a
    summary blocks nothing.
    """
    if backend is EnvironmentBackend.CONTAINER:
        if not _container_available():
            return "-"
        # Deliberately no list. The controls this tier imposes are flags on a
        # daemon this command does not talk to, and some of them -- a cgroup
        # memory ceiling in particular -- are accepted and then ignored by a
        # host without the matching cgroup support. Naming them here would be
        # the unchecked claim the whole isolation layer exists to prevent.
        return "resolved against the container runtime at run time"
    reports = local_isolation_controls(isolation_from_constraints(_SUMMARY_PROBE))
    present = [
        report.control.value
        for report in reports
        if report.outcome is IsolationOutcome.ENFORCED
    ]
    missing = [
        report.control.value
        for report in reports
        if report.outcome
        in (IsolationOutcome.UNENFORCEABLE, IsolationOutcome.FAILED)
    ]
    return f"{', '.join(present) or '(none)'} (no {', '.join(missing)})"


@env_app.command("inspect")
def env_inspect_command(
    benchmark: str = typer.Option(
        ...,
        "--benchmark",
        "-b",
        help="Benchmark ID or path to a benchmark YAML specification.",
    ),
    environment: str | None = typer.Option(
        None,
        "--environment",
        "-e",
        help="Restrict output to one declared environment ID.",
    ),
) -> None:
    """Show a benchmark's declared environments and their real isolation."""
    specification = _load(benchmark)
    selected = _select_environments(specification, environment)
    if not selected:
        console.print(
            f"[yellow]Benchmark '{specification.metadata.id}' declares no "
            "execution environments; it runs on the typed action tier only.[/yellow]"
        )
        return
    for spec in selected:
        _print_environment(specification, spec)


def _print_environment(
    specification: BenchmarkSpecification,
    environment: EnvironmentSpecification,
) -> None:
    """Render one environment: what it declares, then what is really enforced."""
    console.print(f"\n[bold cyan]{environment.id}[/bold cyan] — {environment.name}")
    console.print(f"  {environment.description}")
    detail = Table(show_header=False, box=None, padding=(0, 2))
    detail.add_column("Field", style="dim")
    detail.add_column("Value")
    detail.add_row("backend", environment.backend.value)
    detail.add_row("image", environment.image or "-")
    detail.add_row("languages", ", ".join(environment.languages))
    detail.add_row("host status", _environment_status(environment))
    tasks = sorted(
        task.id for task in specification.tasks if task.environment == environment.id
    )
    detail.add_row("used by tasks", ", ".join(tasks) or "(none)")
    free_actions = sorted(free_execution_action_ids(specification))
    detail.add_row("free-execution actions", ", ".join(free_actions) or "(none)")
    console.print(detail)
    task_constraints = next(
        (
            task.constraints or specification.constraints
            for task in specification.tasks
            if task.environment == environment.id
        ),
        specification.constraints,
    )
    _print_isolation(task_constraints, environment.backend)


def _print_isolation(
    constraints: ConstraintSpecification,
    backend: EnvironmentBackend,
) -> None:
    """Report per control what the benchmark asked for and what it would get.

    The rows are the local tier's own report, not a re-derivation. Assembled
    here from :func:`describe_process_limits` alone, the table listed only the
    resource ceilings plus a hand-written network row, and silently omitted
    ``filesystem_scope`` -- the one control this tier most notably does not
    impose, and the one an operator reading a table headed "Isolation on this
    host" would take the omission as absence of a problem.

    The container tier's answers depend on a daemon this command deliberately
    does not talk to, so they are reported as undetermined rather than assumed:
    predicting enforcement without checking would be exactly the claim this
    whole layer exists to stop.
    """
    request = isolation_from_constraints(constraints)
    table = Table(title="Isolation on this host", title_style="bold")
    table.add_column("Control", style="cyan")
    table.add_column("Requested")
    table.add_column("Outcome")
    table.add_column("Detail")
    if backend is EnvironmentBackend.CONTAINER:
        table.add_row(
            "(all)",
            "see constraints",
            "undetermined",
            "resolved against the Docker runtime at run time",
        )
        console.print(table)
        return
    for report in local_isolation_controls(request):
        table.add_row(
            report.control.value,
            report.requested or "-",
            _outcome_style(report.outcome),
            report.mechanism or report.detail or "-",
        )
    console.print(table)


def _outcome_style(outcome: IsolationOutcome) -> str:
    """Colour an outcome so an absent guarantee is not read as a present one."""
    if outcome is IsolationOutcome.ENFORCED:
        return f"[green]{outcome.value}[/green]"
    if outcome is IsolationOutcome.NOT_REQUESTED:
        return outcome.value
    return f"[yellow]{outcome.value}[/yellow]"


@env_app.command("validate")
def env_validate_command(
    benchmark: str = typer.Option(
        ...,
        "--benchmark",
        "-b",
        help="Benchmark ID or path to a benchmark YAML specification.",
    ),
) -> None:
    """Check that this host can provide every environment a benchmark declares.

    Exits non-zero when it cannot. A benchmark whose container image is
    unavailable here is not a benchmark this host can run, and finding that out
    from an exit code before a run is better than from a failed action during
    one.
    """
    specification = _load(benchmark)
    statuses = {spec.id: _environment_status(spec) for spec in specification.environments}
    availability = {
        spec.id: statuses[spec.id] for spec in specification.environments
    }
    problems = [
        f"environment '{spec.id}' requests the "
        f"'{spec.backend.value}' backend: {statuses[spec.id]}"
        for spec in specification.environments
        if statuses[spec.id] != "available"
    ]
    unenforced = _unenforceable_controls(specification)
    for spec in specification.environments:
        console.print(
            f"[green]OK[/green] environment '{spec.id}' "
            f"({spec.backend.value}) — {availability[spec.id]}"
        )
    if not specification.environments:
        console.print(
            f"[yellow]Benchmark '{specification.metadata.id}' declares no "
            "execution environments.[/yellow]"
        )
    if unenforced:
        # A warning, not a failure: an unconfined run is still a run, and
        # refusing to start would make the honest report useless. What must not
        # happen is the gap going unmentioned.
        console.print(
            "[yellow]WARNING[/yellow] this host cannot enforce "
            f"{', '.join(unenforced)} through the local tier; the run record "
            "will report those controls as unenforceable."
        )
    for problem in problems:
        console.print(f"[red]FAIL[/red] {problem}")
    if problems:
        raise typer.Exit(code=1)


def _unenforceable_controls(specification: BenchmarkSpecification) -> list[str]:
    """Name the controls the local tier cannot impose on this host.

    Reads the same report the run record will carry, so the warning an operator
    sees before the run and the gaps recorded during it are the same list.
    """
    if not any(
        spec.backend is EnvironmentBackend.LOCAL for spec in specification.environments
    ):
        return []
    request = isolation_from_constraints(specification.constraints)
    return [
        report.control.value
        for report in local_isolation_controls(request)
        if report.outcome
        in (IsolationOutcome.UNENFORCEABLE, IsolationOutcome.FAILED)
    ]


def _load(reference: str) -> BenchmarkSpecification:
    """Load a benchmark, turning a bad reference into a CLI error."""
    path = resolve_benchmark_path(reference)
    if not path.exists():
        console.print(f"[red]Benchmark not found:[/red] {reference}")
        raise typer.Exit(code=2)
    try:
        return load_benchmark(path)
    except Exception as error:
        console.print(f"[red]Invalid benchmark[/red] {path}: {error}")
        raise typer.Exit(code=2) from error


def _select_environments(
    specification: BenchmarkSpecification,
    environment_id: str | None,
) -> list[EnvironmentSpecification]:
    """Return the requested environment, or all of them, erroring on unknown."""
    if environment_id is None:
        return list(specification.environments)
    matches = [spec for spec in specification.environments if spec.id == environment_id]
    if not matches:
        declared = ", ".join(spec.id for spec in specification.environments) or "(none)"
        console.print(
            f"[red]Unknown environment[/red] '{environment_id}'; "
            f"benchmark declares: {declared}"
        )
        raise typer.Exit(code=2)
    return matches


__all__ = ["CONTAINER_BINARY", "backend_availability", "env_app"]
