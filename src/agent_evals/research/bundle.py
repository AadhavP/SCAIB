"""Immutable public run bundles for replay and research audit.

A report is an interpretation of a run.  The bundle is the evidence from which
that interpretation can be independently checked: a canonical event ledger plus
content-addressed public files and a manifest.  This module deliberately does
not execute agent code or scientific metrics; it only materializes and verifies
bytes that an evaluator has already produced.

The manifest excludes itself from its file list and hashes its canonical claims
instead.  That avoids a self-referential byte hash while still detecting edits to
the manifest's claims.  The event ledger uses canonical JSON lines so a copied
archive can be checked without importing the original runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BUNDLE_VERSION = "1.0.0"
BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"
EVENT_LEDGER_FILENAME = "events.ndjson"
REPLAY_DESCRIPTOR_FILENAME = "replay.json"
_SHA256_LENGTH = 64


def _sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of exact bytes."""
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically for hashes and event lines."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str | None:
    """Hash a regular file, returning ``None`` when it cannot be read."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


class BundleFile(BaseModel):
    """Content identity of one public bundle file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def valid_digest(cls, value: str) -> str:
        """Reject malformed file identities before they enter a manifest."""
        if len(value) != _SHA256_LENGTH or any(
            character not in "0123456789abcdefABCDEF" for character in value
        ):
            raise ValueError("bundle file sha256 must be a 64-character hexadecimal digest")
        return value.lower()


class BundleEvent(BaseModel):
    """One canonical, public event in a run ledger.

    ``event_sha256`` forms a hash chain over observable events. It is not a
    signature and does not prove who produced an event, but it does make an
    omitted, reordered, or edited event detectable without importing SCAIB.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    event_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    previous_event_sha256: str | None = None
    event_sha256: str | None = None

    @field_validator("previous_event_sha256", "event_sha256")
    @classmethod
    def valid_event_digest(cls, value: str | None) -> str | None:
        """Reject malformed chain links before they become evidence."""
        if value is not None and (
            len(value) != _SHA256_LENGTH
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise ValueError("event sha256 must be a 64-character hexadecimal digest")
        return value.lower() if value is not None else None

    def canonical_payload(self) -> dict[str, Any]:
        """Return the event claims covered by ``event_sha256``."""
        return self.model_dump(mode="json", exclude={"event_sha256"})


def _event_digest(event: BundleEvent) -> str:
    """Hash one event without recursively hashing its own digest."""
    return _sha256_bytes(_canonical_json(event.canonical_payload()))


class ReplayVerification(BaseModel):
    """Structural verification of the inputs needed for later replay.

    This is intentionally separate from bundle integrity. Hashes prove that a
    bundle was not changed; a replay descriptor proves that it contains enough
    public identity and event references for a clean-room tool to attempt a
    replay. It never claims that arbitrary agent code or a scientific metric was
    re-executed.
    """

    model_config = ConfigDict(extra="forbid")

    valid: bool
    descriptor_present: bool = False
    referenced_files: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class BundleVerification(BaseModel):
    """Independent verification result for a public run bundle."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    manifest_sha256: str | None = None
    checked_files: int = Field(default=0, ge=0)
    missing_files: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    unexpected_files: list[str] = Field(default_factory=list)
    event_ledger_valid: bool = False
    event_chain_valid: bool = False
    replay_ready: bool = False
    replay_limitations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class RunBundleManifest(BaseModel):
    """Canonical claims about a materialized public run bundle."""

    model_config = ConfigDict(extra="forbid")

    bundle_version: str = BUNDLE_VERSION
    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_ledger: str = EVENT_LEDGER_FILENAME
    event_ledger_sha256: str
    files: dict[str, BundleFile] = Field(default_factory=dict)
    manifest_sha256: str | None = None

    @field_validator("event_ledger_sha256", "manifest_sha256")
    @classmethod
    def valid_manifest_digest(cls, value: str | None) -> str | None:
        """Reject malformed ledger and manifest digests."""
        if value is not None and (
            len(value) != _SHA256_LENGTH
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise ValueError("bundle digest must be a 64-character hexadecimal digest")
        return value.lower() if value is not None else None

    @model_validator(mode="after")
    def validate_file_paths(self) -> RunBundleManifest:
        """Keep manifest paths relative and unique by construction."""
        for path in self.files:
            candidate = Path(path)
            if candidate.is_absolute() or path in {"", "."}:
                raise ValueError(f"bundle file path must be relative: {path!r}")
            if "\\" in path or any(part == ".." for part in candidate.parts):
                raise ValueError(f"bundle file path escapes its root: {path!r}")
        if self.event_ledger not in self.files:
            raise ValueError("bundle manifest must include the event ledger in files")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        """Return all persisted claims except the recursive self-digest."""
        # ``created_at`` is part of the immutable run evidence. Excluding it made
        # timestamp tampering invisible even though every other manifest claim
        # was protected by the self-digest.
        return self.model_dump(mode="json", exclude={"manifest_sha256"})

    def canonical_digest(self) -> str:
        """Hash the manifest claims that a verifier can recompute."""
        return _sha256_bytes(_canonical_json(self.canonical_payload()))

    def verify_integrity(self) -> bool:
        """Return whether the persisted self-digest matches its claims."""
        return self.manifest_sha256 == self.canonical_digest()




def _event_payload(event: Any) -> dict[str, Any]:
    """Convert a Pydantic or mapping event to a public JSON object."""
    if hasattr(event, "model_dump"):
        event = event.model_dump(mode="json")
    if isinstance(event, Mapping):
        return {
            str(key): value
            for key, value in event.items()
            if key not in {
                "sequence",
                "event_id",
                "previous_event_sha256",
                "event_sha256",
            }
        }
    return {"value": str(event)}


def write_event_ledger(
    root: Path | str,
    events: Sequence[Mapping[str, Any] | Any],
    *,
    filename: str = EVENT_LEDGER_FILENAME,
) -> str:
    """Write canonical JSONL events and return the resulting byte digest.

    The ledger intentionally contains normalized observable events supplied by
    the caller. It does not attempt to retain private chain-of-thought. A caller
    that needs raw response retention must provide a separately governed,
    redacted artifact and reference it by digest.
    """
    bundle_root = Path(root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    lines: list[bytes] = []
    previous: str | None = None
    for sequence, event in enumerate(events):
        payload = _event_payload(event)
        event_type = payload.pop("event_type", None) or payload.pop("type", None)
        source = payload.pop("source", None) or "runtime"
        event = BundleEvent(
            sequence=sequence,
            event_id=f"event-{sequence:08d}",
            source=str(source),
            event_type=str(event_type or "unknown"),
            payload=payload,
            previous_event_sha256=previous,
        )
        event = event.model_copy(update={"event_sha256": _event_digest(event)})
        previous = event.event_sha256
        lines.append(_canonical_json(event.model_dump(mode="json")) + b"\n")
    target = bundle_root / filename
    target.write_bytes(b"".join(lines))
    digest = _sha256_file(target)
    if digest is None:  # pragma: no cover - write_bytes either succeeds or raises
        raise OSError(f"event ledger could not be read after writing: {target}")
    return digest


def read_event_ledger(  # noqa: C901
    path: Path | str,
    *,
    require_chain: bool = False,
) -> list[BundleEvent]:
    """Read and validate a canonical event ledger without executing anything.

    ``require_chain`` is used by strict bundle verification. The permissive
    default keeps old hand-authored fixture ledgers readable, while every ledger
    produced by :func:`write_event_ledger` carries and verifies the chain.
    """
    events: list[BundleEvent] = []
    previous: str | None = None
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = BundleEvent.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"event ledger line {line_number} is invalid: {error}"
                ) from error
            if event.sequence != len(events):
                raise ValueError(
                    f"event ledger sequence is not contiguous at line {line_number}"
                )
            chain_present = (
                event.previous_event_sha256 is not None or event.event_sha256 is not None
            )
            if require_chain and not chain_present:
                raise ValueError(
                    f"event ledger line {line_number} has no tamper-evident hash chain"
                )
            if chain_present:
                if event.event_sha256 is None:
                    raise ValueError(
                        f"event ledger line {line_number} is missing event_sha256"
                    )
                if event.previous_event_sha256 != previous:
                    raise ValueError(
                        f"event ledger chain is broken at line {line_number}"
                    )
                if _event_digest(event) != event.event_sha256:
                    raise ValueError(
                        f"event ledger event_sha256 does not match line {line_number}"
                    )
                previous = event.event_sha256
            events.append(event)
    if require_chain and not events:
        raise ValueError("event ledger is empty and cannot establish a run chain")
    return events


def write_run_bundle_manifest(
    root: Path | str,
    *,
    run_id: str,
    event_ledger_sha256: str | None = None,
) -> RunBundleManifest:
    """Hash every public file and write a canonical bundle manifest.

    ``bundle_manifest.json`` is excluded from its own file table. All other
    files, including reports and artifacts, are content-addressed. Unexpected
    files added after this function returns are therefore detectable.
    """
    bundle_root = Path(root)
    ledger_path = bundle_root / EVENT_LEDGER_FILENAME
    observed_ledger_digest = _sha256_file(ledger_path)
    if observed_ledger_digest is None:
        raise FileNotFoundError(f"event ledger is missing: {ledger_path}")
    if event_ledger_sha256 is not None:
        if (
            len(event_ledger_sha256) != _SHA256_LENGTH
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in event_ledger_sha256
            )
        ):
            raise ValueError("event_ledger_sha256 must be a 64-character hexadecimal digest")
        if event_ledger_sha256.lower() != observed_ledger_digest:
            raise ValueError(
                "event_ledger_sha256 does not match the bytes on disk"
            )
    ledger_digest = observed_ledger_digest
    files: dict[str, BundleFile] = {}
    resolved_root = bundle_root.resolve()
    for candidate in sorted(bundle_root.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(bundle_root).as_posix()
        if relative == BUNDLE_MANIFEST_FILENAME:
            continue
        resolved = candidate.resolve()
        if resolved_root not in resolved.parents:
            raise ValueError(f"bundle file escapes root through symlink: {relative}")
        digest = _sha256_file(candidate)
        if digest is None:
            raise OSError(f"bundle file could not be hashed: {candidate}")
        files[relative] = BundleFile(
            sha256=digest,
            size_bytes=candidate.stat().st_size,
        )
    manifest = RunBundleManifest(
        run_id=run_id,
        event_ledger_sha256=ledger_digest,
        files=files,
    )
    manifest = manifest.model_copy(update={"manifest_sha256": manifest.canonical_digest()})
    (bundle_root / BUNDLE_MANIFEST_FILENAME).write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_replay_descriptor(root: Path | str) -> ReplayVerification:  # noqa: C901
    """Check replay metadata and referenced public files without executing code."""
    bundle_root = Path(root)
    descriptor_path = bundle_root / REPLAY_DESCRIPTOR_FILENAME
    if not descriptor_path.is_file():
        return ReplayVerification(
            valid=False,
            limitations=[f"{REPLAY_DESCRIPTOR_FILENAME} is missing"],
        )
    try:
        payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return ReplayVerification(
            valid=False,
            descriptor_present=True,
            limitations=[
                f"replay descriptor is invalid: {type(error).__name__}: {error}"
            ],
        )
    if not isinstance(payload, dict):
        return ReplayVerification(
            valid=False,
            descriptor_present=True,
            limitations=["replay descriptor must contain an object"],
        )
    required = {
        "replay_version",
        "run_id",
        "benchmark_id",
        "task_id",
        "seed",
        "deterministic",
        "replay_mode",
        "event_ledger",
        "event_ledger_sha256",
        "trajectory",
        "report",
    }
    limitations = [
        f"replay descriptor is missing '{name}'"
        for name in sorted(required - set(payload))
    ]
    string_fields = (
        "replay_version",
        "run_id",
        "benchmark_id",
        "task_id",
        "replay_mode",
    )
    for field_name in string_fields:
        value = payload.get(field_name)
        if field_name in payload and (
            not isinstance(value, str) or not value.strip()
        ):
            limitations.append(
                f"replay descriptor field '{field_name}' must be a non-empty string"
            )
    seed = payload.get("seed")
    if "seed" in payload and (not isinstance(seed, int) or isinstance(seed, bool)):
        limitations.append("replay descriptor field 'seed' must be an integer")
    deterministic = payload.get("deterministic")
    if "deterministic" in payload and not isinstance(deterministic, bool):
        limitations.append("replay descriptor field 'deterministic' must be boolean")
    expected_digest = payload.get("event_ledger_sha256")
    if "event_ledger_sha256" in payload and not _is_sha256(expected_digest):
        limitations.append(
            "replay descriptor field 'event_ledger_sha256' must be a SHA-256 digest"
        )
    private_keys = {
        "api_key",
        "authorization",
        "access_token",
        "chain_of_thought",
        "private_reasoning",
    }
    leaked = sorted(private_keys.intersection(payload))
    if leaked:
        limitations.append(
            "replay descriptor contains prohibited private fields: "
            + ", ".join(leaked)
        )
    referenced: list[str] = []
    resolved_root = bundle_root.resolve()
    for key in ("event_ledger", "trajectory", "report"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            if key in payload:
                limitations.append(
                    f"replay descriptor field '{key}' must be a non-empty path"
                )
            continue
        candidate_path = Path(value)
        if (
            candidate_path.is_absolute()
            or "\\" in value
            or any(part in {"", ".", ".."} for part in candidate_path.parts)
        ):
            limitations.append(f"replay file path is not a safe relative path: {value}")
            continue
        referenced.append(value)
        candidate = bundle_root / value
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            limitations.append(f"replay file is missing: {value}")
            continue
        if resolved_root not in resolved.parents:
            limitations.append(f"replay file escapes bundle root: {value}")
    ledger_name = payload.get("event_ledger")
    if isinstance(ledger_name, str) and ledger_name in referenced and _is_sha256(expected_digest):
        ledger_path = bundle_root / ledger_name
        actual = _sha256_file(ledger_path)
        if actual != expected_digest:
            limitations.append("replay descriptor event ledger digest does not match")
        else:
            try:
                read_event_ledger(ledger_path, require_chain=True)
            except (OSError, ValueError) as error:
                limitations.append(
                    f"replay event ledger is not replay-ready: {type(error).__name__}: {error}"
                )
    return ReplayVerification(
        valid=not limitations,
        descriptor_present=True,
        referenced_files=referenced,
        limitations=limitations,
    )


def _is_sha256(value: Any) -> bool:
    """Return whether a value is a canonical 64-hex SHA-256 digest."""
    return isinstance(value, str) and len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def verify_run_bundle(root: Path | str) -> BundleVerification:  # noqa: C901
    """Verify manifest integrity, file bytes, paths, and event ordering."""
    bundle_root = Path(root)
    manifest_path = bundle_root / BUNDLE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return BundleVerification(
            valid=False,
            limitations=[f"{BUNDLE_MANIFEST_FILENAME} is missing"],
        )
    try:
        manifest = RunBundleManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        return BundleVerification(
            valid=False,
            limitations=[f"bundle manifest is invalid: {type(error).__name__}: {error}"],
        )
    limitations: list[str] = []
    if not manifest.verify_integrity():
        limitations.append("bundle manifest self-digest does not match its claims")
    missing: list[str] = []
    changed: list[str] = []
    checked = 0
    resolved_root = bundle_root.resolve()
    for relative, expected in sorted(manifest.files.items()):
        candidate = bundle_root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            missing.append(relative)
            continue
        if resolved_root not in resolved.parents:
            changed.append(relative)
            continue
        observed = _sha256_file(candidate)
        if observed is None:
            missing.append(relative)
        elif observed != expected.sha256 or candidate.stat().st_size != expected.size_bytes:
            changed.append(relative)
        else:
            checked += 1
    expected_paths = set(manifest.files)
    unexpected: list[str] = []
    try:
        for candidate in sorted(bundle_root.rglob("*")):
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(bundle_root).as_posix()
            if relative not in expected_paths and relative != BUNDLE_MANIFEST_FILENAME:
                unexpected.append(relative)
    except (OSError, RuntimeError) as error:
        limitations.append(f"bundle walk failed: {type(error).__name__}: {error}")
    event_valid = False
    event_chain_valid = False
    ledger_path = bundle_root / manifest.event_ledger
    actual_ledger_digest = _sha256_file(ledger_path)
    if actual_ledger_digest != manifest.event_ledger_sha256:
        limitations.append("event ledger digest does not match the bundle manifest")
    else:
        try:
            read_event_ledger(ledger_path, require_chain=True)
            event_valid = True
            event_chain_valid = True
        except (OSError, ValueError) as error:
            limitations.append(
                f"event ledger could not be validated: {type(error).__name__}: {error}"
            )
    replay = verify_replay_descriptor(bundle_root)
    return BundleVerification(
        valid=(
            not limitations
            and not missing
            and not changed
            and not unexpected
            and event_valid
            and event_chain_valid
        ),
        manifest_sha256=manifest.manifest_sha256,
        checked_files=checked,
        missing_files=missing,
        changed_files=sorted(set(changed)),
        unexpected_files=unexpected,
        event_ledger_valid=event_valid,
        event_chain_valid=event_chain_valid,
        replay_ready=replay.valid,
        replay_limitations=replay.limitations,
        limitations=limitations,
    )


__all__ = [
    "BUNDLE_MANIFEST_FILENAME",
    "BUNDLE_VERSION",
    "EVENT_LEDGER_FILENAME",
    "REPLAY_DESCRIPTOR_FILENAME",
    "BundleEvent",
    "BundleFile",
    "BundleVerification",
    "ReplayVerification",
    "RunBundleManifest",
    "read_event_ledger",
    "verify_replay_descriptor",
    "verify_run_bundle",
    "write_event_ledger",
    "write_run_bundle_manifest",
]
