"""Tests for the replay-oriented public run bundle contract."""

from __future__ import annotations

import json
from pathlib import Path

from agent_evals.research.bundle import (
    BUNDLE_MANIFEST_FILENAME,
    EVENT_LEDGER_FILENAME,
    read_event_ledger,
    verify_replay_descriptor,
    verify_run_bundle,
    write_event_ledger,
    write_run_bundle_manifest,
)


def _write_bundle(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text('{"score":0.8}\n', encoding="utf-8")
    write_event_ledger(
        tmp_path,
        [
            {
                "source": "agent",
                "event_type": "observation",
                "payload": {"step": 0},
            },
            {
                "source": "environment",
                "event_type": "action_finished",
                "payload": {"step": 1, "status": "succeeded"},
            },
        ],
    )
    write_run_bundle_manifest(tmp_path, run_id="run-1")


def test_bundle_round_trip_is_independently_verifiable(tmp_path: Path) -> None:
    _write_bundle(tmp_path)

    verification = verify_run_bundle(tmp_path)

    assert verification.valid is True
    assert verification.event_ledger_valid is True
    assert verification.event_chain_valid is True
    assert verification.checked_files == 2
    assert read_event_ledger(tmp_path / EVENT_LEDGER_FILENAME)[1].sequence == 1


def test_bundle_detects_changed_and_unexpected_files(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "report.json").write_text('{"score":0.2}\n', encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("late mutation", encoding="utf-8")

    verification = verify_run_bundle(tmp_path)

    assert verification.valid is False
    assert "report.json" in verification.changed_files
    assert "untracked.txt" in verification.unexpected_files


def test_bundle_detects_manifest_timestamp_tampering(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    manifest_path = tmp_path / BUNDLE_MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["created_at"] = "2030-01-01T00:00:00Z"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    verification = verify_run_bundle(tmp_path)

    assert verification.valid is False
    assert any("self-digest" in item for item in verification.limitations)


def test_manifest_rejects_a_caller_supplied_ledger_digest_mismatch(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text("{}\n", encoding="utf-8")
    write_event_ledger(tmp_path, [{"source": "test", "event_type": "done"}])

    try:
        write_run_bundle_manifest(tmp_path, run_id="run-1", event_ledger_sha256="0" * 64)
    except ValueError as error:
        assert "does not match" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("a stale event ledger digest must be rejected")


def test_bundle_detects_manifest_claim_tampering(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    manifest_path = tmp_path / BUNDLE_MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["run_id"] = "rewritten"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    verification = verify_run_bundle(tmp_path)

    assert verification.valid is False
    assert any("self-digest" in item for item in verification.limitations)


def test_bundle_detects_event_chain_tampering(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    ledger = tmp_path / EVENT_LEDGER_FILENAME
    lines = ledger.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[1])
    payload["payload"]["status"] = "rewritten"
    lines[1] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\\n".join(lines) + "\\n", encoding="utf-8")

    verification = verify_run_bundle(tmp_path)

    assert verification.valid is False
    assert verification.event_ledger_valid is False
    assert verification.event_chain_valid is False
    assert any("event ledger" in item for item in verification.limitations)


def test_replay_descriptor_rejects_unsafe_or_malformed_references(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "trajectory.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "replay.json").write_text(
        json.dumps(
            {
                "replay_version": "1.0.0",
                "run_id": "run-1",
                "benchmark_id": "benchmark",
                "task_id": "task-1",
                "seed": 0,
                "deterministic": True,
                "replay_mode": "event_sourced",
                "event_ledger": "../events.ndjson",
                "event_ledger_sha256": "not-a-digest",
                "trajectory": "trajectory.json",
                "report": "report.json",
            }
        ),
        encoding="utf-8",
    )

    verification = verify_replay_descriptor(tmp_path)

    assert verification.valid is False
    assert any("SHA-256" in item for item in verification.limitations)
    assert any("safe relative path" in item for item in verification.limitations)


def test_event_ledger_rejects_non_contiguous_sequences(tmp_path: Path) -> None:
    ledger = tmp_path / EVENT_LEDGER_FILENAME
    ledger.write_text(
        '{"event_id":"event-00000000","event_type":"one","payload":{},'
        '"sequence":0,"source":"test"}\n'
        '{"event_id":"event-00000002","event_type":"three","payload":{},'
        '"sequence":2,"source":"test"}\n',
        encoding="utf-8",
    )

    try:
        read_event_ledger(ledger)
    except ValueError as error:
        assert "not contiguous" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("non-contiguous event sequences must be rejected")
