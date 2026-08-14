"""Tests for Typer CLI commands."""

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from agent_evals.agents.trajectory import RunTerminationStatus
from agent_evals.cli.main import app
from agent_evals.environment.scientific_loop import DEFAULT_RUNTIME_MAX_STEPS
from agent_evals.research.bundle import write_event_ledger, write_run_bundle_manifest

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "agent-evals version" in result.stdout


def test_cli_list_benchmarks() -> None:
    result = runner.invoke(app, ["list-benchmarks"])
    assert result.exit_code == 0
    assert "Registered Benchmarks" in result.stdout


def test_cli_list_agents() -> None:
    result = runner.invoke(app, ["list-agents"])
    assert result.exit_code == 0
    assert "Registered Agent Adapters" in result.stdout


def test_cli_agent_list_runtimes() -> None:
    result = runner.invoke(app, ["agent", "list"])
    assert result.exit_code == 0
    assert "Available Agents" in result.stdout
    assert "openai" in result.stdout


def test_cli_research_readiness_lifecycle(tmp_path: Path) -> None:
    manifest = tmp_path / "readiness.yaml"
    certificate = tmp_path / "certificate.json"

    initialized = runner.invoke(
        app,
        [
            "research",
            "init",
            "--benchmark-id",
            "toy",
            "--benchmark-version",
            "1.0.0",
            "--output",
            str(manifest),
        ],
    )
    assert initialized.exit_code == 0, initialized.stdout
    assert manifest.is_file()

    certified = runner.invoke(
        app,
        [
            "research",
            "certify",
            "--manifest",
            str(manifest),
            "--output",
            str(certificate),
        ],
    )
    assert certified.exit_code == 0, certified.stdout
    assert "status=blocked" in certified.stdout
    assert certificate.is_file()

    verified = runner.invoke(
        app,
        [
            "research",
            "verify",
            "--manifest",
            str(manifest),
            "--certificate",
            str(certificate),
            "--strict",
        ],
    )
    # Strict verification checks integrity, not whether the intentionally empty
    # starter checklist has empirical evidence; a blocked but untampered
    # certificate is therefore valid and remains visibly BLOCKED above.
    assert verified.exit_code == 0, verified.stdout
    assert "certificate=VALID" in verified.stdout

    protocol = runner.invoke(app, ["research", "protocol-check", "--strict"])
    assert protocol.exit_code == 0, protocol.stdout
    assert "protocol_suite=PASS" in protocol.stdout


def test_cli_runs_synthetic_research_conformance(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "research",
            "conformance",
            "--output",
            str(tmp_path / "conformance"),
            "--strict",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "PASS synthetic-conformance" in result.stdout
    assert "bundle_hash_chain_and_manifest_verify" in result.stdout


def test_cli_validates_benchmarks_and_public_bundles(tmp_path: Path) -> None:
    benchmark = runner.invoke(
        app,
        [
            "validate-benchmark",
            "--benchmark",
            "examples/benchmarks/pbmc-cell-annotation.yaml",
        ],
    )
    assert benchmark.exit_code == 0, benchmark.stdout
    assert "VALID" in benchmark.stdout

    (tmp_path / "report.json").write_text("{}\n", encoding="utf-8")
    write_event_ledger(
        tmp_path,
        [{"source": "test", "event_type": "complete", "payload": {}}],
    )
    write_run_bundle_manifest(tmp_path, run_id="cli-run")
    bundle = runner.invoke(app, ["verify-bundle", str(tmp_path)])

    assert bundle.exit_code == 0, bundle.stdout
    assert "VALID" in bundle.stdout


def test_cli_run_routes_universal_runtime_to_scientific_loop(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeScientificLoop:
        async def run(self, benchmark: str, **kwargs: object) -> object:
            calls.update({"benchmark": benchmark, **kwargs})
            return SimpleNamespace(
                run_id="run-1",
                agent_run=SimpleNamespace(termination_status=RunTerminationStatus.COMPLETED),
                evaluation=SimpleNamespace(global_agent_score=0.5),
                report_path="results/run-1/report.md",
            )

    monkeypatch.setattr("agent_evals.environment.scientific_loop.ScientificLoop", FakeScientificLoop)

    result = runner.invoke(
        app,
        ["run", "--agent", "openai", "--model", "gpt-5", "--max-cells", "12"],
    )

    assert result.exit_code == 0
    assert calls["agent_type"] == "openai"
    assert calls["model"] == "gpt-5"
    assert calls["max_cells"] == 12
    assert calls["max_steps"] == DEFAULT_RUNTIME_MAX_STEPS
