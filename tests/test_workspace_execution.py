"""Tests for running an agent's own code in an observed, isolated workspace.

Three properties here carry more weight than the rest, because each is a place
the implementation could look finished while being wrong:

1. **A failing execution is a result, not an exception.** Every test that drives
   a failure asserts on a returned status. If any of them started raising, the
   environment would record "executor error" and blame the harness for the
   agent's bug.
2. **Isolation is reported, never assumed.** The platform-degradation path is
   asserted for *both* platforms via an injected platform name, because the host
   running these tests only ever exercises one of them.
3. **A declared artifact is verified, not believed.** The agent says what its
   code produces; the executor must check.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from agent_evals.benchmarks.schema import ConstraintSpecification
from agent_evals.core.exceptions import SandboxExecutionError
from agent_evals.environment.execution.backend import (
    POSIX_ENVIRONMENT_ALLOWLIST,
    WINDOWS_ENVIRONMENT_ALLOWLIST,
    CommandRequest,
    Language,
    WorkspaceBackend,
    environment_allowlist,
    interpreter_argv,
)
from agent_evals.environment.execution.container import (
    CONTAINER_WORKSPACE,
    OOM_EXIT_CODE,
    ContainerBackend,
    build_exec_argv,
    build_run_argv,
    docker_available,
)
from agent_evals.environment.execution.executor import (
    WorkspaceActionExecutor,
    command_timeout,
    declared_artifacts,
    deterministic_environment,
    isolation_from_constraints,
)
from agent_evals.environment.execution.fingerprint import (
    DigestMethod,
    fingerprint_workspace,
)
from agent_evals.environment.execution.isolation import (
    IsolationControl,
    IsolationOutcome,
    IsolationReport,
    IsolationRequest,
    describe_process_limits,
    environment_report,
    filesystem_report,
    network_report,
    supports_resource_limits,
)
from agent_evals.environment.execution.local import (
    BACKEND_NAME,
    LocalProcessBackend,
    _CappedBuffer,
    _read_peak_rss_mb,
    resolve_within,
)
from agent_evals.environment.models import (
    ActionIntent,
    ActionStatus,
    EpisodeSnapshot,
    EpisodeState,
    ExecutionStatus,
)
from agent_evals.environment.ports import ActionExecutor, ExecutionContext

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

HAS_BASH = shutil.which("bash") is not None


def make_context(**constraints: object) -> ExecutionContext:
    """Build the minimum viable execution context for the executor."""
    state = EpisodeState(
        episode_id="episode-1",
        benchmark_id="bench",
        benchmark_version="1.0.0",
        task_id="task-1",
        seed=42,
        specification_digest="digest",
    )
    return ExecutionContext(
        snapshot=EpisodeSnapshot(state=state),
        constraints=ConstraintSpecification(**constraints),  # type: ignore[arg-type]
    )


async def started_backend(root: Path, **kwargs: object) -> LocalProcessBackend:
    """Return a started local backend rooted at ``root``."""
    backend = LocalProcessBackend(root, **kwargs)  # type: ignore[arg-type]
    await backend.start()
    return backend


def python_request(source: str, **kwargs: object) -> CommandRequest:
    """Build a Python command request with a short default ceiling."""
    options: dict[str, object] = {"timeout_seconds": 30.0}
    options.update(kwargs)
    return CommandRequest(command=source, language=Language.PYTHON, **options)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Workspace fingerprints
# --------------------------------------------------------------------------


def test_fingerprint_hashes_content_and_records_the_method(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "clusters.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    # Act
    fingerprint = fingerprint_workspace(tmp_path)

    # Assert
    entry = fingerprint.files["outputs/clusters.csv"]
    assert entry.method is DigestMethod.SHA256
    assert entry.is_proof
    assert fingerprint.total_bytes == entry.size_bytes
    assert fingerprint.paths == {"outputs/clusters.csv"}


def test_fingerprint_falls_back_to_size_and_mtime_for_large_files(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "big.h5ad").write_bytes(b"x" * 64)

    # Act
    fingerprint = fingerprint_workspace(tmp_path, max_content_bytes=8)

    # Assert
    entry = fingerprint.files["big.h5ad"]
    assert entry.method is DigestMethod.SIZE_MTIME
    assert not entry.is_proof, "size+mtime equality is evidence, never proof"


def test_fingerprint_changes_when_content_changes(tmp_path: Path) -> None:
    # Arrange
    target = tmp_path / "result.json"
    target.write_text("{}", encoding="utf-8")
    before = fingerprint_workspace(tmp_path)

    # Act
    target.write_text('{"n": 1}', encoding="utf-8")
    after = fingerprint_workspace(tmp_path)

    # Assert
    assert before.files["result.json"].digest != after.files["result.json"].digest


def test_fingerprint_skips_churn_directories_and_missing_roots(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mod.pyc").write_bytes(b"\x00")
    (tmp_path / "kept.txt").write_text("keep", encoding="utf-8")

    # Act
    fingerprint = fingerprint_workspace(tmp_path)
    absent = fingerprint_workspace(tmp_path / "nope")

    # Assert
    assert fingerprint.paths == {"kept.txt"}
    assert absent.files == {}


def test_fingerprint_records_symlinks_as_unreadable_instead_of_following(
    tmp_path: Path,
) -> None:
    # Arrange
    outside = tmp_path / "outside.txt"
    outside.write_text("reference labels", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        (workspace / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation requires privileges this host lacks")

    # Act
    fingerprint = fingerprint_workspace(workspace)

    # Assert
    assert fingerprint.files == {}
    assert "link.txt" in fingerprint.unreadable


# --------------------------------------------------------------------------
# Path containment
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate",
    [
        "../escape.csv",
        "../../reference.csv.gz",
        "",
        " outputs/x.csv",
        ".",
    ],
)
def test_paths_that_escape_or_name_the_root_are_refused(
    tmp_path: Path,
    candidate: str,
) -> None:
    assert resolve_within(tmp_path, candidate) is None


def test_paths_inside_the_workspace_resolve(tmp_path: Path) -> None:
    resolved = resolve_within(tmp_path, "outputs/clusters.csv")
    assert resolved is not None
    assert resolved.name == "clusters.csv"
    assert tmp_path.resolve() in resolved.parents


# --------------------------------------------------------------------------
# Isolation reporting
# --------------------------------------------------------------------------


def test_resource_limits_are_enforced_on_posix_and_unenforceable_on_windows() -> None:
    # Arrange
    request = IsolationRequest(
        max_memory_mb=2048,
        max_cpu_seconds=60,
        max_processes=16,
        max_file_size_mb=512,
    )

    # Act
    posix = describe_process_limits(request, platform_name="linux")
    windows = describe_process_limits(request, platform_name="win32")

    # Assert
    assert {report.outcome for report in posix} == {IsolationOutcome.ENFORCED}
    assert {report.outcome for report in windows} == {IsolationOutcome.UNENFORCEABLE}
    assert {report.control for report in posix} == {
        IsolationControl.ADDRESS_SPACE,
        IsolationControl.CPU_TIME,
        IsolationControl.PROCESS_COUNT,
        IsolationControl.FILE_SIZE,
    }
    assert all("container backend" in (report.detail or "") for report in windows)
    assert supports_resource_limits("linux")
    assert not supports_resource_limits("win32")


def test_unrequested_limits_are_reported_as_nothing_rather_than_enforced() -> None:
    request = IsolationRequest()
    assert describe_process_limits(request, platform_name="linux") == ()
    assert not request.requests_any_limit
    assert IsolationRequest(max_memory_mb=8).requests_any_limit


def test_address_space_is_a_distinct_control_from_resident_memory() -> None:
    # A cgroup bounds resident memory; RLIMIT_AS bounds virtual address space.
    # Reporting both as "memory" would hide why a run died.
    (report,) = describe_process_limits(
        IsolationRequest(max_memory_mb=1024),
        platform_name="linux",
    )
    assert report.control is IsolationControl.ADDRESS_SPACE
    assert report.control is not IsolationControl.RESIDENT_MEMORY
    assert report.mechanism == "RLIMIT_AS"


def test_a_denied_network_is_unenforceable_locally_and_enforced_in_a_container() -> None:
    local = network_report(network_access=False, enforceable=False)
    contained = network_report(
        network_access=False,
        enforceable=True,
        mechanism="docker --network=none",
    )
    allowed = network_report(network_access=True, enforceable=False)

    assert local.outcome is IsolationOutcome.UNENFORCEABLE
    assert contained.outcome is IsolationOutcome.ENFORCED
    assert allowed.outcome is IsolationOutcome.NOT_REQUESTED


def test_filesystem_scope_distinguishes_aiming_from_confining() -> None:
    unconfined = filesystem_report(enforceable=False)
    confined = filesystem_report(enforceable=True, mechanism="docker bind mounts")

    assert unconfined.outcome is IsolationOutcome.UNENFORCEABLE
    assert "refused" in (unconfined.detail or "")
    assert confined.outcome is IsolationOutcome.ENFORCED


def test_the_environment_scrub_promises_credentials_are_unreachable() -> None:
    report = environment_report(allowlisted=("PATH", "HOME"))
    assert report.outcome is IsolationOutcome.ENFORCED
    assert report.requested == "2 variable(s) passed through"
    assert "API keys" in (report.detail or "")


def test_a_report_summarises_which_controls_were_real() -> None:
    # Arrange
    report = IsolationReport(
        backend="local",
        platform="win32",
        controls=[
            network_report(network_access=False, enforceable=False),
            filesystem_report(enforceable=False),
            environment_report(allowlisted=("PATH",)),
        ],
    )

    # Assert
    assert report.enforced == {IsolationControl.ENVIRONMENT}
    assert report.unenforced == {
        IsolationControl.NETWORK,
        IsolationControl.FILESYSTEM_SCOPE,
    }
    assert not report.is_complete
    assert report.outcome_for(IsolationControl.CPU_TIME) is IsolationOutcome.NOT_REQUESTED
    assert (
        IsolationReport(backend="container", platform="linux", controls=[]).is_complete
    ), "a report with nothing unenforced is complete"


def test_the_local_backends_report_names_its_platform_not_the_hosts() -> None:
    backend = LocalProcessBackend(Path("."), platform_name="win32")
    report = backend.isolation_report()
    assert report.backend == BACKEND_NAME
    assert report.platform == "win32"


# --------------------------------------------------------------------------
# Interpreter dispatch and the environment allowlist
# --------------------------------------------------------------------------


def test_source_is_read_from_standard_input_for_every_language() -> None:
    # A script *file* would appear in the workspace as a change SCAIB caused,
    # contaminating the before/after diff provenance is derived from.
    assert interpreter_argv(Language.PYTHON, python_executable="py") == ("py", "-u", "-")
    assert interpreter_argv(
        Language.BASH,
        python_executable="py",
        shell_executable="sh",
    ) == ("sh", "-s")


def test_the_allowlist_is_platform_specific_and_never_a_denylist() -> None:
    assert environment_allowlist("win32") == WINDOWS_ENVIRONMENT_ALLOWLIST
    assert environment_allowlist("linux") == POSIX_ENVIRONMENT_ALLOWLIST
    assert "SYSTEMROOT" in WINDOWS_ENVIRONMENT_ALLOWLIST, "Python needs it for TLS"
    for allowlist in (WINDOWS_ENVIRONMENT_ALLOWLIST, POSIX_ENVIRONMENT_ALLOWLIST):
        assert not any("KEY" in name or "TOKEN" in name for name in allowlist)


# --------------------------------------------------------------------------
# Output capping
# --------------------------------------------------------------------------


def test_a_capped_buffer_keeps_what_fits_and_always_admits_truncation() -> None:
    buffer = _CappedBuffer(limit=5)
    buffer.feed(b"abc")
    assert not buffer.truncated
    buffer.feed(b"defgh")
    assert buffer.text() == "abcde"
    assert buffer.truncated
    buffer.feed(b"ignored")
    assert buffer.text() == "abcde"


def test_a_capped_buffer_tolerates_a_split_multibyte_character() -> None:
    buffer = _CappedBuffer(limit=2)
    buffer.feed("é".encode())
    buffer.feed(b"\xc3")
    assert buffer.text()
    assert buffer.truncated


def test_peak_memory_is_none_rather_than_guessed_when_unmeasurable() -> None:
    # Over-reporting a memory figure would fail runs against a ceiling for
    # something an earlier step did.
    assert _read_peak_rss_mb(-1) is None


# --------------------------------------------------------------------------
# The local backend
# --------------------------------------------------------------------------


async def test_the_local_backend_satisfies_the_workspace_backend_port(
    tmp_path: Path,
) -> None:
    backend = await started_backend(tmp_path)
    assert isinstance(backend, WorkspaceBackend)
    await backend.close()


async def test_using_the_backend_before_starting_it_is_a_harness_fault(
    tmp_path: Path,
) -> None:
    # This is the one thing that raises: a lifecycle fault is SCAIB's problem,
    # not evidence about the agent.
    backend = LocalProcessBackend(tmp_path)
    with pytest.raises(SandboxExecutionError, match="before start"):
        await backend.run(python_request("print(1)"))


async def test_a_successful_execution_reports_output_and_zero_exit(
    tmp_path: Path,
) -> None:
    # Arrange
    backend = await started_backend(tmp_path)

    # Act
    outcome = await backend.run(python_request("print('hello science')"))

    # Assert
    assert outcome.succeeded
    assert outcome.status is ExecutionStatus.SUCCESS
    assert outcome.exit_code == 0
    assert "hello science" in outcome.stdout
    assert outcome.error is None
    assert not outcome.truncated
    assert outcome.resource_usage.wall_time_seconds >= 0.0


async def test_code_runs_with_the_workspace_as_its_working_directory(
    tmp_path: Path,
) -> None:
    # Arrange
    backend = await started_backend(tmp_path)

    # Act
    outcome = await backend.run(
        python_request("open('made-here.txt', 'w').write('ok')")
    )

    # Assert
    assert outcome.succeeded
    assert (tmp_path / "made-here.txt").read_text(encoding="utf-8") == "ok"
    fingerprint = await backend.fingerprint()
    assert "made-here.txt" in fingerprint.paths


async def test_a_raising_script_is_an_error_result_not_an_exception(
    tmp_path: Path,
) -> None:
    # Arrange
    backend = await started_backend(tmp_path)

    # Act
    outcome = await backend.run(python_request("raise ValueError('bad qc threshold')"))

    # Assert
    assert outcome.status is ExecutionStatus.ERROR
    assert not outcome.succeeded
    assert outcome.exit_code == 1
    assert "bad qc threshold" in outcome.stderr
    assert "exited with code 1" in (outcome.error or "")


async def test_a_syntax_error_before_stdin_is_read_still_reports_stderr(
    tmp_path: Path,
) -> None:
    # Arrange
    backend = await started_backend(tmp_path)

    # Act
    outcome = await backend.run(python_request("def broken(:\n    pass\n"))

    # Assert
    assert outcome.status is ExecutionStatus.ERROR
    assert "SyntaxError" in outcome.stderr


async def test_a_runaway_execution_times_out_instead_of_hanging(
    tmp_path: Path,
) -> None:
    # Arrange
    backend = await started_backend(tmp_path)
    request = python_request(
        "import time\nprint('before', flush=True)\ntime.sleep(30)\n",
        timeout_seconds=1.0,
    )

    # Act
    outcome = await backend.run(request)

    # Assert
    assert outcome.status is ExecutionStatus.TIMEOUT
    assert "exceeded its 1s limit" in (outcome.error or "")
    assert "before" in outcome.stdout, "partial output survives the kill"


async def test_output_above_the_ceiling_is_dropped_and_the_loss_is_recorded(
    tmp_path: Path,
) -> None:
    # Arrange
    backend = await started_backend(tmp_path)
    request = python_request("print('x' * 100_000)", max_output_bytes=256)

    # Act
    outcome = await backend.run(request)

    # Assert
    assert outcome.succeeded
    assert len(outcome.stdout) <= 256
    assert outcome.stdout_truncated
    assert outcome.truncated, "scoring must never read a clipped stream as empty"


async def test_the_execution_cannot_read_the_evaluators_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-be-visible")
    monkeypatch.setenv("SCAIB_EVALUATOR_SECRET", "hidden-reference-path")
    backend = await started_backend(tmp_path)
    source = (
        "import os\n"
        "print(os.environ.get('ANTHROPIC_API_KEY', 'ABSENT'))\n"
        "print(os.environ.get('SCAIB_EVALUATOR_SECRET', 'ABSENT'))\n"
    )

    # Act
    outcome = await backend.run(python_request(source))

    # Assert
    assert outcome.succeeded
    assert "sk-should-never-be-visible" not in outcome.stdout
    assert "hidden-reference-path" not in outcome.stdout
    assert outcome.stdout.count("ABSENT") == 2


async def test_requested_variables_are_layered_over_the_allowlist(
    tmp_path: Path,
) -> None:
    # Arrange
    backend = await started_backend(tmp_path, extra_environment={"SCAIB_STAGE": "qc"})
    source = (
        "import os\n"
        "print(os.environ['SCAIB_STAGE'], os.environ['PYTHONHASHSEED'])\n"
    )

    # Act
    outcome = await backend.run(python_request(source, env={"PYTHONHASHSEED": "7"}))

    # Assert
    assert outcome.succeeded
    assert "qc 7" in outcome.stdout


async def test_a_missing_interpreter_is_reported_as_a_named_harness_error(
    tmp_path: Path,
) -> None:
    # Arrange
    backend = await started_backend(
        tmp_path,
        shell_executable="scaib-nonexistent-shell",
    )

    # Act
    outcome = await backend.run(
        CommandRequest(command="echo hi", language=Language.BASH)
    )

    # Assert
    assert outcome.status is ExecutionStatus.ERROR
    assert "could not start bash interpreter" in (outcome.error or "")
    assert "scaib-nonexistent-shell" in (outcome.error or "")


@pytest.mark.skipif(not HAS_BASH, reason="bash is not on PATH")
async def test_shell_source_also_runs_from_standard_input(tmp_path: Path) -> None:
    # Arrange
    backend = await started_backend(tmp_path)

    # Act
    outcome = await backend.run(
        CommandRequest(
            command="echo shell-ran > from-shell.txt",
            language=Language.BASH,
            timeout_seconds=30.0,
        )
    )

    # Assert
    assert outcome.succeeded, outcome.error
    assert (tmp_path / "from-shell.txt").exists()


async def test_a_memory_error_under_a_limit_is_classified_as_oom(
    tmp_path: Path,
) -> None:
    # Arrange: the classification is asserted directly because provoking a real
    # allocation failure would be slow and platform-dependent, while the
    # OOM-vs-crash distinction is scored and must be right on every platform.
    backend = await started_backend(
        tmp_path,
        isolation=IsolationRequest(max_memory_mb=512),
    )

    # Act
    status, error = backend._classify(
        exit_code=1,
        timed_out=False,
        timeout_seconds=30.0,
        stderr_text="Traceback...\nMemoryError\n",
    )
    killed, kill_error = backend._classify(
        exit_code=-9,
        timed_out=False,
        timeout_seconds=30.0,
        stderr_text="",
    )

    # Assert
    assert status is ExecutionStatus.OOM
    assert "512 MB" in (error or "")
    assert killed is ExecutionStatus.OOM
    assert "inferred from the signal, not measured" in (kill_error or "")


async def test_a_signal_kill_without_a_memory_limit_is_terminated_not_oom(
    tmp_path: Path,
) -> None:
    # Arrange
    backend = await started_backend(tmp_path)

    # Act
    status, error = backend._classify(
        exit_code=-15,
        timed_out=False,
        timeout_seconds=30.0,
        stderr_text="",
    )

    # Assert
    assert status is ExecutionStatus.TERMINATED
    assert "signal 15" in (error or "")


# --------------------------------------------------------------------------
# The executor behind the ActionExecutor port
# --------------------------------------------------------------------------


async def test_the_executor_satisfies_the_existing_action_executor_port(
    tmp_path: Path,
) -> None:
    # This is what keeps episodes, decisions, trajectories, and scoring working
    # unchanged above this layer.
    executor = WorkspaceActionExecutor(await started_backend(tmp_path))
    assert isinstance(executor, ActionExecutor)


async def test_a_declared_artifact_that_exists_is_recorded_with_a_checksum(
    tmp_path: Path,
) -> None:
    # Arrange
    executor = WorkspaceActionExecutor(await started_backend(tmp_path))
    intent = ActionIntent(
        action_id="cluster",
        parameters={
            "code": (
                "import pathlib\n"
                "pathlib.Path('outputs').mkdir(exist_ok=True)\n"
                "pathlib.Path('outputs/clusters.csv').write_text('cell,cluster\\n')\n"
            ),
            "produces": {"clusters": "outputs/clusters.csv"},
        },
    )

    # Act
    result = await executor.execute(intent, make_context())

    # Assert
    assert result.status is ActionStatus.SUCCEEDED
    assert result.execution_status is ExecutionStatus.SUCCESS
    (artifact,) = result.artifacts
    assert artifact.artifact_id == "clusters"
    assert artifact.format == "csv"
    assert artifact.kind == "table"
    assert (artifact.checksum or "").startswith("sha256:")
    assert not artifact.validated, "existence is not scientific validity"
    assert result.outputs == {"clusters": "outputs/clusters.csv"}


async def test_a_declared_artifact_that_was_not_produced_fails_the_action(
    tmp_path: Path,
) -> None:
    # The never-trust-claims principle, at the point a claim first enters.
    executor = WorkspaceActionExecutor(await started_backend(tmp_path))
    intent = ActionIntent(
        action_id="cluster",
        parameters={
            "code": "print('I totally wrote the file')",
            "produces": ["outputs/clusters.csv"],
        },
    )

    # Act
    result = await executor.execute(intent, make_context())

    # Assert
    assert result.status is ActionStatus.FAILED
    assert result.execution_status is ExecutionStatus.PARTIAL
    assert "were not produced" in (result.error or "")
    assert "outputs/clusters.csv" in (result.error or "")
    assert result.outputs == {}


async def test_a_declared_path_outside_the_workspace_is_refused_before_running(
    tmp_path: Path,
) -> None:
    # Arrange
    workspace = tmp_path / "workspace"
    executor = WorkspaceActionExecutor(await started_backend(workspace))
    canary = tmp_path / "reference.csv"
    canary.write_text("cell_type\n", encoding="utf-8")
    intent = ActionIntent(
        action_id="annotate",
        parameters={
            "code": "raise AssertionError('must never run')",
            "produces": {"labels": "../reference.csv"},
        },
    )

    # Act
    result = await executor.execute(intent, make_context())

    # Assert
    assert result.status is ActionStatus.FAILED
    assert "resolve outside the workspace" in (result.error or "")
    assert "labels -> ../reference.csv" in (result.error or "")
    assert result.observations == [], "nothing ran, so there is nothing to observe"


async def test_an_intent_without_code_fails_with_an_actionable_message(
    tmp_path: Path,
) -> None:
    # Arrange
    executor = WorkspaceActionExecutor(await started_backend(tmp_path))

    # Act
    empty = await executor.execute(ActionIntent(action_id="qc"), make_context())
    blank = await executor.execute(
        ActionIntent(action_id="qc", parameters={"code": "   "}),
        make_context(),
    )

    # Assert
    for result in (empty, blank):
        assert result.status is ActionStatus.FAILED
        assert result.execution_status is ExecutionStatus.ERROR
        assert "requires a non-empty 'code' parameter" in (result.error or "")


async def test_an_unsupported_language_is_rejected_rather_than_mis_run(
    tmp_path: Path,
) -> None:
    # R is not installed; a stubbed language would fail as a harness error and
    # be scored against the agent.
    executor = WorkspaceActionExecutor(await started_backend(tmp_path))
    intent = ActionIntent(
        action_id="qc",
        parameters={"code": "library(Seurat)", "language": "r"},
    )

    result = await executor.execute(intent, make_context())

    assert result.status is ActionStatus.FAILED
    assert "unsupported language 'r'" in (result.error or "")
    assert "python" in (result.error or "")


async def test_a_malformed_produces_declaration_is_rejected(tmp_path: Path) -> None:
    executor = WorkspaceActionExecutor(await started_backend(tmp_path))
    intent = ActionIntent(
        action_id="qc",
        parameters={"code": "print(1)", "produces": "outputs/clusters.csv"},
    )

    result = await executor.execute(intent, make_context())

    assert result.status is ActionStatus.FAILED
    assert "must be a list of paths or a mapping" in (result.error or "")


async def test_a_failed_execution_surfaces_stderr_to_the_agent(tmp_path: Path) -> None:
    # Arrange
    executor = WorkspaceActionExecutor(await started_backend(tmp_path))
    intent = ActionIntent(
        action_id="qc",
        parameters={"code": "raise RuntimeError('n_genes filter removed all cells')"},
    )

    # Act
    result = await executor.execute(intent, make_context())

    # Assert
    assert result.status is ActionStatus.FAILED
    assert result.execution_status is ExecutionStatus.ERROR
    stderr = next(
        item for item in result.observations if item.observation_id == "execution-stderr"
    )
    assert "n_genes filter removed all cells" in str(stderr.value)
    assert stderr.visible_to_agent, "an agent that cannot see its error cannot debug"


async def test_the_isolation_report_is_recorded_but_hidden_from_the_agent(
    tmp_path: Path,
) -> None:
    # An agent that can read which controls went unenforced has been handed a
    # map of what it can get away with.
    executor = WorkspaceActionExecutor(await started_backend(tmp_path))
    intent = ActionIntent(action_id="qc", parameters={"code": "print('ok')"})

    result = await executor.execute(intent, make_context())

    isolation = next(
        item
        for item in result.observations
        if item.observation_id == "execution-isolation"
    )
    assert not isolation.visible_to_agent
    assert isolation.value["backend"] == BACKEND_NAME
    visible = {item.observation_id for item in result.observations if item.visible_to_agent}
    assert visible == {"execution-stdout", "execution-stderr", "execution-status"}


async def test_execution_telemetry_never_leaks_into_the_artifact_contract(
    tmp_path: Path,
) -> None:
    # ``_normalize_result`` derives satisfied ``expected_outputs`` from
    # ``outputs``, so a stray telemetry key there could accidentally satisfy a
    # declared artifact name.
    executor = WorkspaceActionExecutor(await started_backend(tmp_path))
    intent = ActionIntent(action_id="qc", parameters={"code": "print('ok')"})

    result = await executor.execute(intent, make_context())

    assert result.status is ActionStatus.SUCCEEDED
    assert result.outputs == {}
    assert "stdout" not in result.outputs


async def test_an_intent_declaring_nothing_succeeds_with_no_artifacts(
    tmp_path: Path,
) -> None:
    executor = WorkspaceActionExecutor(await started_backend(tmp_path))
    intent = ActionIntent(action_id="explore", parameters={"code": "print(2 + 2)"})

    result = await executor.execute(intent, make_context())

    assert result.status is ActionStatus.SUCCEEDED
    assert result.artifacts == []


# --------------------------------------------------------------------------
# Constraint translation
# --------------------------------------------------------------------------


def test_benchmark_constraints_become_a_backend_neutral_isolation_request() -> None:
    # Arrange
    constraints = ConstraintSpecification(
        internet_access=False,
        max_memory_mb=8192,
        max_runtime_seconds=600,
    )

    # Act
    request = isolation_from_constraints(constraints)

    # Assert
    assert not request.network_access
    assert request.max_memory_mb == 8192
    assert request.max_cpu_seconds == 600


def test_a_single_command_cannot_consume_the_whole_episode_budget() -> None:
    assert command_timeout(ConstraintSpecification(max_runtime_seconds=600)) == 300.0
    assert command_timeout(ConstraintSpecification()) == 300.0
    assert command_timeout(ConstraintSpecification(max_runtime_seconds=1)) == 1.0


def test_a_deterministic_benchmark_seeds_the_interpreter_before_it_starts() -> None:
    # PYTHONHASHSEED has to be in the environment: a seed set from inside the
    # script is already too late to affect string hashing.
    seeded = deterministic_environment(
        ConstraintSpecification(deterministic=True, random_seed=42)
    )
    assert seeded == {"PYTHONHASHSEED": "42", "SCAIB_RANDOM_SEED": "42"}
    assert deterministic_environment(ConstraintSpecification()) == {}


def test_both_produces_spellings_are_accepted() -> None:
    listed = declared_artifacts(
        ActionIntent(action_id="a", parameters={"produces": ["outputs/x.csv"]})
    )
    mapped = declared_artifacts(
        ActionIntent(action_id="a", parameters={"produces": {"x": "outputs/x.csv"}})
    )
    assert listed == {"outputs/x.csv": "outputs/x.csv"}
    assert mapped == {"x": "outputs/x.csv"}
    assert declared_artifacts(ActionIntent(action_id="a")) == {}


# --------------------------------------------------------------------------
# The container backend
# --------------------------------------------------------------------------


def test_the_container_run_argv_pins_swap_to_the_memory_ceiling() -> None:
    # Without --memory-swap, Docker grants swap equal to the memory limit, so a
    # run nominally capped at 8 GB may use 16 GB.
    argv = build_run_argv(
        image="scaib-exec:latest",
        workspace=Path("/tmp/ws"),
        isolation=IsolationRequest(
            network_access=False,
            max_memory_mb=8192,
            max_processes=64,
        ),
        cpu_limit=2.0,
    )

    assert "--memory=8192m" in argv
    assert "--memory-swap=8192m" in argv
    assert "--network=none" in argv
    assert "--pids-limit=64" in argv
    assert "--cpus=2" in argv
    assert argv[-1] == "infinity"
    assert "--rm" in argv


def test_the_container_mounts_the_dataset_read_only(tmp_path: Path) -> None:
    # So the agent cannot rewrite the data it is scored on.
    argv = build_run_argv(
        image="img",
        workspace=tmp_path,
        isolation=IsolationRequest(),
        input_dir=tmp_path,
    )
    mounts = [item for item in argv if CONTAINER_WORKSPACE in item]
    assert any(item.endswith(f"{CONTAINER_WORKSPACE}/inputs:ro") for item in mounts)


def test_an_unconstrained_container_declares_no_limit_flags(tmp_path: Path) -> None:
    argv = build_run_argv(image="img", workspace=tmp_path, isolation=IsolationRequest())
    assert not any(item.startswith("--memory") for item in argv)
    assert "--network=none" not in argv
    assert not any(item.startswith("--pids-limit") for item in argv)


def test_the_container_exec_argv_carries_env_and_reads_stdin() -> None:
    argv = build_exec_argv(
        container_id="abc123",
        language=Language.PYTHON,
        env={"PYTHONHASHSEED": "42", "SCAIB_STAGE": "qc"},
    )
    assert argv[:3] == ("docker", "exec", "--interactive")
    assert "--env" in argv
    assert "PYTHONHASHSEED=42" in argv
    assert argv[-3:] == ("python", "-u", "-")
    assert "abc123" in argv

    shell = build_exec_argv(container_id="abc123", language=Language.BASH, env={})
    assert shell[-2:] == ("bash", "-s")


def test_the_container_backend_claims_real_isolation(tmp_path: Path) -> None:
    # Arrange
    backend = ContainerBackend(
        tmp_path,
        image="img",
        isolation=IsolationRequest(
            network_access=False,
            max_memory_mb=4096,
            max_processes=32,
        ),
    )

    # Act
    report = backend.isolation_report()

    # Assert
    assert report.outcome_for(IsolationControl.NETWORK) is IsolationOutcome.ENFORCED
    assert (
        report.outcome_for(IsolationControl.FILESYSTEM_SCOPE)
        is IsolationOutcome.ENFORCED
    )
    assert (
        report.outcome_for(IsolationControl.RESIDENT_MEMORY) is IsolationOutcome.ENFORCED
    )
    assert (
        report.outcome_for(IsolationControl.PROCESS_COUNT) is IsolationOutcome.ENFORCED
    )
    assert report.is_complete


def test_an_unlimited_container_does_not_claim_a_memory_ceiling(tmp_path: Path) -> None:
    backend = ContainerBackend(tmp_path, image="img")
    report = backend.isolation_report()
    assert (
        report.outcome_for(IsolationControl.RESIDENT_MEMORY)
        is IsolationOutcome.NOT_REQUESTED
    )
    assert report.outcome_for(IsolationControl.NETWORK) is IsolationOutcome.NOT_REQUESTED


def test_the_container_maps_exit_137_to_oom_only_under_a_memory_limit(
    tmp_path: Path,
) -> None:
    limited = ContainerBackend(
        tmp_path,
        image="img",
        isolation=IsolationRequest(max_memory_mb=2048),
    )
    unlimited = ContainerBackend(tmp_path, image="img")

    oom, oom_error = limited._classify(
        exit_code=OOM_EXIT_CODE,
        timed_out=False,
        timeout_seconds=30.0,
    )
    killed, _ = unlimited._classify(
        exit_code=OOM_EXIT_CODE,
        timed_out=False,
        timeout_seconds=30.0,
    )
    timeout, timeout_error = limited._classify(
        exit_code=None,
        timed_out=True,
        timeout_seconds=30.0,
    )
    success, _ = limited._classify(exit_code=0, timed_out=False, timeout_seconds=30.0)

    assert oom is ExecutionStatus.OOM
    assert "2048 MB" in (oom_error or "")
    assert killed is ExecutionStatus.ERROR
    assert timeout is ExecutionStatus.TIMEOUT
    assert "30s limit" in (timeout_error or "")
    assert success is ExecutionStatus.SUCCESS


async def test_the_container_backend_refuses_to_run_before_it_is_started(
    tmp_path: Path,
) -> None:
    backend = ContainerBackend(tmp_path, image="img")
    assert backend.container_id is None
    with pytest.raises(SandboxExecutionError, match="before start"):
        await backend.run(python_request("print(1)"))
    await backend.close()  # idempotent when nothing was started


async def test_a_missing_container_runtime_is_a_named_harness_fault(
    tmp_path: Path,
) -> None:
    backend = ContainerBackend(
        tmp_path,
        image="img",
        executable="scaib-nonexistent-docker",
    )
    with pytest.raises(SandboxExecutionError, match="on PATH"):
        await backend.start()


def test_docker_availability_is_probed_rather_than_assumed() -> None:
    assert docker_available("scaib-nonexistent-docker") is False
    assert isinstance(docker_available(), bool)


@pytest.mark.skipif(not docker_available(), reason="Docker is not installed")
@pytest.mark.skipif(sys.platform == "win32", reason="needs a Linux container host")
async def test_a_real_container_runs_code_and_enforces_its_network_denial(
    tmp_path: Path,
) -> None:
    # Arrange
    backend = ContainerBackend(
        tmp_path,
        image="python:3.12-slim",
        isolation=IsolationRequest(network_access=False, max_memory_mb=512),
    )
    await backend.start()

    # Act
    try:
        ran = await backend.run(python_request("print('in a container')"))
        networked = await backend.run(
            python_request(
                "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=5)\n"
            )
        )
    finally:
        await backend.close()

    # Assert
    assert ran.succeeded, ran.error
    assert "in a container" in ran.stdout
    assert not networked.succeeded, "a denied network must actually be denied"
    assert backend.container_id is None
