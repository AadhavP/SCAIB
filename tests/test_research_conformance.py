"""Dataset-independent conformance tests for the research loop."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_evals.research.conformance import (
    build_synthetic_conformance_specification,
    run_synthetic_conformance,
)


def test_synthetic_conformance_specification_has_no_dataset_dependency() -> None:
    specification = build_synthetic_conformance_specification()

    assert specification.datasets == []
    assert specification.metadata.id == "synthetic-conformance"
    assert specification.required_task_artifacts(specification.tasks[0]) == {"result"}


def test_synthetic_conformance_runs_endpoint_loop_and_verifies_bundle(
    tmp_path: Path,
) -> None:
    report = asyncio.run(run_synthetic_conformance(tmp_path))

    assert report.passed is True
    assert report.bundle_valid is True
    assert report.artifact_verified is True
    assert report.checks["replay_descriptor_is_ready"] is True
    assert report.endpoint_phases == [
        "initialize",
        "plan",
        "observation",
        "observation",
        "terminate",
    ]
    bundle = Path(report.bundle_path)
    assert (bundle / "bundle_manifest.json").is_file()
    assert (bundle / "events.ndjson").is_file()
    assert (bundle / "replay.json").is_file()
    exchange = json.loads((bundle / "endpoint_exchange.json").read_text())
    assert exchange["request_count"] == 5
    assert all(item["request_id"] for item in exchange["requests"])


def test_synthetic_conformance_is_repeatable_as_a_contract(tmp_path: Path) -> None:
    first = asyncio.run(run_synthetic_conformance(tmp_path / "first"))
    second = asyncio.run(run_synthetic_conformance(tmp_path / "second"))

    assert first.passed and second.passed
    assert first.checks == second.checks
    assert first.endpoint_phases == second.endpoint_phases
    assert first.action_count == second.action_count
