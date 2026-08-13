"""Tests for the ``env`` commands, whose only job is an honest isolation report.

This module exists because the commands it covers make *claims* and enforce
nothing. An operator runs ``env list`` and ``env validate`` before spending money
on a run, decides from the output whether the run will be confined, and then
reads a paper that repeats the decision. Nothing downstream compares those
claims to what the run record ends up saying, so a wrong claim here is invisible
in exactly the way every other non-blocking defect in this project has been.

It was wrong. ``env list`` reported ``filesystem_scope`` among the controls the
local tier imposes, while :meth:`LocalProcessBackend.isolation_report` recorded
that same control as ``unenforceable`` on every host and explained why -- a local
process keeps the host write permissions of the user running the benchmark.
``env inspect`` omitted the control from its table altogether, and ``env
validate``'s warning left it out of the list of gaps. So the pre-run summary
promised confinement no run ever had.

The tests are therefore written as *comparisons against the backend's own
report* rather than as expected strings. A hand-written expectation would have
been just as wrong as the code, and would have made the drift permanent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.schema import (
    ConstraintSpecification,
    EnvironmentBackend,
)
from agent_evals.cli import environments as env_module
from agent_evals.cli.environments import (
    CONTAINER_BINARY,
    _backend_isolation_summary,
    _outcome_style,
    _unenforceable_controls,
    backend_availability,
)
from agent_evals.cli.main import app
from agent_evals.environment.execution import (
    IsolationOutcome,
    LocalProcessBackend,
    isolation_from_constraints,
    local_isolation_controls,
)

runner = CliRunner()

FREE_BENCHMARK = "examples/benchmarks/pbmc-cell-annotation-free.yaml"
TYPED_BENCHMARK = "examples/benchmarks/pbmc-cell-annotation.yaml"

#: Wide enough that no table cell wraps. A wrapped cell splits the very control
#: name these tests assert on across two lines, so a real regression would be
#: reported as a missing substring and a real fix as a passing test at random.
WIDE = {"COLUMNS": "220"}


def invoke(*args: str, env: dict[str, str] | None = None):
    """Run the CLI with a terminal wide enough to read the tables it prints."""
    return runner.invoke(app, list(args), env={**WIDE, **(env or {})})


@pytest.fixture
def container_benchmark(tmp_path: Path) -> Path:
    """A real benchmark whose only environment asks for the container tier.

    Derived from the shipped free benchmark rather than hand-written, so it stays
    a benchmark that genuinely loads: ``validate_integrity`` couples environments
    to tasks, actions, and artifacts, and a minimal hand-rolled file would be
    testing the fixture rather than the command.
    """
    document = yaml.safe_load(Path(FREE_BENCHMARK).read_text(encoding="utf-8"))
    for environment in document["environments"]:
        environment["backend"] = "container"
        environment["image"] = "scaib/exec:test"
    path = tmp_path / "container-benchmark.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def no_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the host look like one with no container runtime installed."""
    monkeypatch.setattr(shutil, "which", lambda name: None)


@pytest.fixture
def with_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the host look like one with a container runtime installed."""
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


# ---------------------------------------------------------------------------
# Host availability. Reported from the host, never asserted.
# ---------------------------------------------------------------------------


def test_the_container_tier_is_unavailable_when_its_binary_is_absent(
    no_container: None,
) -> None:
    status = backend_availability()[EnvironmentBackend.CONTAINER]
    assert status != "available"
    assert CONTAINER_BINARY in status
    assert "PATH" in status


def test_the_container_tier_is_available_when_its_binary_is_present(
    with_container: None,
) -> None:
    assert backend_availability()[EnvironmentBackend.CONTAINER] == "available"


def test_the_local_tier_is_available_whether_or_not_a_runtime_is_installed(
    no_container: None,
) -> None:
    """The local tier needs nothing installed, so its status cannot depend on it."""
    assert backend_availability()[EnvironmentBackend.LOCAL] == "available"


def test_an_unavailable_container_tier_summarizes_no_isolation_at_all(
    no_container: None,
) -> None:
    assert _backend_isolation_summary(EnvironmentBackend.CONTAINER) == "-"


def test_an_available_container_tier_does_not_name_controls_it_did_not_check(
    with_container: None,
) -> None:
    """The container tier's controls are flags on a daemon nobody asked.

    A cgroup memory ceiling in particular can be accepted and then ignored by a
    host lacking the matching cgroup support, so naming it as imposed would be
    the unchecked claim this whole layer exists to prevent.
    """
    summary = _backend_isolation_summary(EnvironmentBackend.CONTAINER)
    assert "run time" in summary
    assert "enforced" not in summary
    for control in ("network", "resident_memory", "filesystem_scope"):
        assert control not in summary


# ---------------------------------------------------------------------------
# The anti-drift claim: the summary and the run record are one computation.
# ---------------------------------------------------------------------------


def _summary_halves(summary: str) -> tuple[set[str], set[str]]:
    """Split ``present (no missing)`` back into the two sets it renders."""
    present, _, missing = summary.partition(" (no ")
    return (
        {item.strip() for item in present.split(",") if item.strip() != "(none)"},
        {item.strip() for item in missing.rstrip(")").split(",") if item.strip()},
    )


def test_the_summary_names_exactly_the_controls_the_backend_reports_enforced(
    tmp_path: Path,
) -> None:
    """The pre-run claim must equal the post-run record, not merely resemble it.

    Compared against a real :class:`LocalProcessBackend` rather than against a
    literal, because a literal would have agreed with the defect this test was
    written for.
    """
    request = isolation_from_constraints(env_module._SUMMARY_PROBE)
    backend = LocalProcessBackend(tmp_path, isolation=request)
    present, _ = _summary_halves(_backend_isolation_summary(EnvironmentBackend.LOCAL))
    assert present == {control.value for control in backend.isolation_report().enforced}


def test_the_summary_names_exactly_the_controls_the_backend_reports_unenforced(
    tmp_path: Path,
) -> None:
    request = isolation_from_constraints(env_module._SUMMARY_PROBE)
    backend = LocalProcessBackend(tmp_path, isolation=request)
    _, missing = _summary_halves(_backend_isolation_summary(EnvironmentBackend.LOCAL))
    assert missing == {
        control.value for control in backend.isolation_report().unenforced
    }


def test_the_summary_probe_requests_every_control_the_tier_has_a_mechanism_for() -> (
    None
):
    """A control nobody asked for answers ``not_requested`` and drops out.

    That is the failure mode the probe's shape guards: dropped from both halves,
    an unrequested control reads as the absence of a problem rather than as an
    unanswered question. The network is the one that matters -- the local tier can
    never deny it -- so a probe that allowed the network would silently stop
    reporting the tier's most consequential gap.
    """
    assert env_module._SUMMARY_PROBE.internet_access is False
    assert env_module._SUMMARY_PROBE.max_memory_mb is not None
    assert env_module._SUMMARY_PROBE.max_runtime_seconds is not None


@pytest.mark.parametrize("platform_name", ["win32", "linux", "darwin"])
def test_the_local_tier_never_claims_to_confine_writes(platform_name: str) -> None:
    """No host makes a bare subprocess stay inside the workspace.

    Pinning the working directory aims writes at the workspace; it does not
    confine them. The distinction is the whole reason ``filesystem_scope`` is
    named after its mechanism.
    """
    reports = {
        report.control.value: report.outcome
        for report in local_isolation_controls(
            isolation_from_constraints(env_module._SUMMARY_PROBE),
            platform_name=platform_name,
        )
    }
    assert reports["filesystem_scope"] is IsolationOutcome.UNENFORCEABLE


@pytest.mark.parametrize("platform_name", ["win32", "linux", "darwin"])
def test_the_local_tier_never_claims_to_deny_the_network(platform_name: str) -> None:
    reports = {
        report.control.value: report.outcome
        for report in local_isolation_controls(
            isolation_from_constraints(env_module._SUMMARY_PROBE),
            platform_name=platform_name,
        )
    }
    assert reports["network"] is IsolationOutcome.UNENFORCEABLE


@pytest.mark.parametrize("platform_name", ["win32", "linux", "darwin"])
def test_the_local_tier_does_claim_the_environment_scrub(platform_name: str) -> None:
    """The presence complement: a real guarantee must still be reported as one.

    Without this, reporting *everything* as unenforceable would pass every
    absence assertion above while making the whole report useless.
    """
    reports = {
        report.control.value: report.outcome
        for report in local_isolation_controls(
            isolation_from_constraints(env_module._SUMMARY_PROBE),
            platform_name=platform_name,
        )
    }
    assert reports["environment"] is IsolationOutcome.ENFORCED


def test_a_posix_host_reports_its_resource_ceilings_as_enforced() -> None:
    """Linux is where the paper's runs happen, and there the ceilings are real."""
    reports = {
        report.control.value: report.outcome
        for report in local_isolation_controls(
            isolation_from_constraints(env_module._SUMMARY_PROBE),
            platform_name="linux",
        )
    }
    assert reports["address_space"] is IsolationOutcome.ENFORCED
    assert reports["cpu_time"] is IsolationOutcome.ENFORCED


def test_a_windows_host_reports_the_same_ceilings_as_unenforceable() -> None:
    reports = {
        report.control.value: report.outcome
        for report in local_isolation_controls(
            isolation_from_constraints(env_module._SUMMARY_PROBE),
            platform_name="win32",
        )
    }
    assert reports["address_space"] is IsolationOutcome.UNENFORCEABLE
    assert reports["cpu_time"] is IsolationOutcome.UNENFORCEABLE


def test_an_unrequested_ceiling_is_reported_by_nobody() -> None:
    """A benchmark that asks for no memory ceiling must not be told it has one."""
    controls = {
        report.control.value
        for report in local_isolation_controls(
            isolation_from_constraints(ConstraintSpecification()),
            platform_name="linux",
        )
    }
    assert "address_space" not in controls
    assert "cpu_time" not in controls
    # The two unconditional controls remain, because this tier applies both on
    # every execution whether or not a benchmark mentions them.
    assert {"filesystem_scope", "environment"} <= controls


# ---------------------------------------------------------------------------
# env list
# ---------------------------------------------------------------------------


def test_list_reports_both_tiers_and_the_host_platform() -> None:
    result = invoke("env", "list")
    assert result.exit_code == 0
    assert "local" in result.stdout
    assert "container" in result.stdout
    assert "Platform:" in result.stdout


def test_list_names_the_local_tier_gaps_it_would_hide(no_container: None) -> None:
    result = invoke("env", "list")
    assert result.exit_code == 0
    assert "filesystem_scope" in result.stdout
    assert "network" in result.stdout


# ---------------------------------------------------------------------------
# env inspect
# ---------------------------------------------------------------------------


def test_inspect_reports_the_declared_environment_and_its_tasks() -> None:
    result = invoke("env", "inspect", "-b", FREE_BENCHMARK)
    assert result.exit_code == 0
    assert "local-python" in result.stdout
    assert "cell-annotation-free" in result.stdout


def test_inspect_names_the_control_the_local_tier_cannot_impose() -> None:
    """The row that was missing entirely, in the table headed for this host."""
    result = invoke("env", "inspect", "-b", FREE_BENCHMARK)
    assert result.exit_code == 0
    assert "filesystem_scope" in result.stdout
    assert "unenforceable" in result.stdout


def test_inspect_reports_the_enforced_control_too() -> None:
    result = invoke("env", "inspect", "-b", FREE_BENCHMARK)
    assert result.exit_code == 0
    assert "environment" in result.stdout
    assert "enforced" in result.stdout


def test_inspect_reports_the_container_tier_as_undetermined(
    container_benchmark: Path, with_container: None
) -> None:
    result = invoke("env", "inspect", "-b", str(container_benchmark))
    assert result.exit_code == 0
    assert "undetermined" in result.stdout
    assert "unenforceable" not in result.stdout


def test_inspect_accepts_a_declared_environment_id() -> None:
    result = invoke("env", "inspect", "-b", FREE_BENCHMARK, "-e", "local-python")
    assert result.exit_code == 0
    assert "local-python" in result.stdout


def test_inspect_rejects_an_unknown_environment_id() -> None:
    result = invoke("env", "inspect", "-b", FREE_BENCHMARK, "-e", "no-such-env")
    assert result.exit_code == 2
    assert "no-such-env" in result.stdout
    assert "local-python" in result.stdout


def test_inspect_reports_a_typed_benchmark_as_declaring_no_environments() -> None:
    """The typed tier is not a defect, so this is a message and not an error."""
    result = invoke("env", "inspect", "-b", TYPED_BENCHMARK)
    assert result.exit_code == 0
    assert "no execution environments" in result.stdout


def test_inspect_resolves_a_bare_benchmark_id() -> None:
    result = invoke("env", "inspect", "-b", "pbmc-cell-annotation-free")
    assert result.exit_code == 0
    assert "local-python" in result.stdout


def test_inspect_exits_two_on_a_benchmark_that_does_not_exist() -> None:
    result = invoke("env", "inspect", "-b", "no-such-benchmark")
    assert result.exit_code == 2
    assert "not found" in result.stdout


def test_inspect_exits_two_on_a_benchmark_that_does_not_load(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("metadata: {id: broken}\n", encoding="utf-8")
    result = invoke("env", "inspect", "-b", str(path))
    assert result.exit_code == 2
    assert "Invalid benchmark" in result.stdout


# ---------------------------------------------------------------------------
# env validate
# ---------------------------------------------------------------------------


def test_validate_passes_a_benchmark_whose_backend_this_host_provides() -> None:
    result = invoke("env", "validate", "-b", FREE_BENCHMARK)
    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_validate_warns_about_unenforceable_controls_without_failing() -> None:
    """An unconfined run is still a run; refusing to start would hide the gap.

    What must not happen is the gap going unmentioned, so the exit code stays 0
    and the warning is the deliverable.
    """
    result = invoke("env", "validate", "-b", FREE_BENCHMARK)
    assert result.exit_code == 0
    assert "WARNING" in result.stdout
    assert "filesystem_scope" in result.stdout


def test_the_validate_warning_lists_what_the_run_record_will_report(
    tmp_path: Path,
) -> None:
    """Same comparison as the summary test, at the level an operator acts on."""
    specification = load_benchmark(Path(FREE_BENCHMARK))
    backend = LocalProcessBackend(
        tmp_path,
        isolation=isolation_from_constraints(specification.constraints),
    )
    assert set(_unenforceable_controls(specification)) == {
        control.value for control in backend.isolation_report().unenforced
    }


def test_validate_fails_when_a_declared_backend_is_unavailable(
    container_benchmark: Path, no_container: None
) -> None:
    result = invoke("env", "validate", "-b", str(container_benchmark))
    assert result.exit_code == 1
    assert "FAIL" in result.stdout


def test_validate_passes_the_same_benchmark_when_the_backend_is_available(
    container_benchmark: Path, with_container: None
) -> None:
    """The complement, without which the test above passes on any always-fail."""
    result = invoke("env", "validate", "-b", str(container_benchmark))
    assert result.exit_code == 0
    assert "FAIL" not in result.stdout


def test_validate_warns_about_no_local_controls_for_a_container_only_benchmark(
    container_benchmark: Path, with_container: None
) -> None:
    """A benchmark with no local environment has no local gaps to report."""
    specification = load_benchmark(container_benchmark)
    assert _unenforceable_controls(specification) == []


def test_validate_reports_a_typed_benchmark_as_declaring_no_environments() -> None:
    result = invoke("env", "validate", "-b", TYPED_BENCHMARK)
    assert result.exit_code == 0
    assert "no execution environments" in result.stdout


def test_validate_exits_two_on_a_benchmark_that_does_not_exist() -> None:
    result = invoke("env", "validate", "-b", "no-such-benchmark")
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# Rendering. An absent guarantee must not look like a present one.
# ---------------------------------------------------------------------------


def test_an_unenforceable_outcome_is_not_styled_like_an_enforced_one() -> None:
    assert _outcome_style(IsolationOutcome.ENFORCED) != _outcome_style(
        IsolationOutcome.UNENFORCEABLE
    )
    assert "green" in _outcome_style(IsolationOutcome.ENFORCED)
    assert "yellow" in _outcome_style(IsolationOutcome.UNENFORCEABLE)


def test_a_failed_control_is_styled_as_a_gap_not_as_a_guarantee() -> None:
    assert "yellow" in _outcome_style(IsolationOutcome.FAILED)


def test_a_control_nobody_requested_is_not_styled_as_either() -> None:
    """An unasked question is neither a guarantee nor a gap."""
    assert (
        _outcome_style(IsolationOutcome.NOT_REQUESTED)
        == IsolationOutcome.NOT_REQUESTED.value
    )


def test_every_outcome_renders_its_own_value() -> None:
    """Styling must not lose the word, whatever colour it wears."""
    for outcome in IsolationOutcome:
        assert outcome.value in _outcome_style(outcome)
