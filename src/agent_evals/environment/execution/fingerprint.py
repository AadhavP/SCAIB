"""Content fingerprints for a workspace directory tree.

Once the agent runs its own code, SCAIB cannot know what an action did by
inspecting the action -- it only knows what the action *said* it would do.
Provenance therefore has to come from observing the workspace before and after,
which is what a fingerprint is for: a value that changes when the files change
and is cheap enough to take on every step.

The cheapness is the constraint that shapes this module.  A single-cell run
writes ``.h5ad`` files in the hundreds of megabytes, so hashing every byte of
every file on every step would make the observation cost dominate the science.
Large files therefore fall back to a size-and-mtime identity, and the fallback
is recorded per file rather than hidden: ``size_mtime`` equality is evidence
that a file did not change, while ``sha256`` equality is proof.  A consumer that
needs proof can ask which it got.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from pydantic import Field

from agent_evals.environment.models import KeyDelta, RuntimeModel

#: Files at or below this size are hashed by content. Above it, hashing every
#: step costs more than the provenance is worth; see :class:`DigestMethod`.
CONTENT_DIGEST_MAX_BYTES = 128 * 1024 * 1024

#: Directory names never worth fingerprinting: churn with no scientific content.
IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".ipynb_checkpoints",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }
)

_READ_CHUNK_BYTES = 1024 * 1024


class DigestMethod(StrEnum):
    """How one file's identity was established."""

    #: Content hash. Equality is proof the bytes are identical.
    SHA256 = "sha256"
    #: Size plus modification time. Equality is *evidence* the file did not
    #: change, not proof: an in-place edit that preserves size within the
    #: filesystem's timestamp granularity is invisible to it.
    SIZE_MTIME = "size_mtime"


class FileFingerprint(RuntimeModel):
    """Identity of one file at one point in time."""

    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    digest: str = Field(min_length=1)
    method: DigestMethod

    @property
    def is_proof(self) -> bool:
        """Whether comparing this digest can prove content equality."""
        return self.method is DigestMethod.SHA256


class WorkspaceFingerprint(RuntimeModel):
    """Identity of every fingerprinted file under a workspace root."""

    root: str
    files: dict[str, FileFingerprint] = Field(default_factory=dict)
    #: Paths that exist but could not be read, with the reason. A file that
    #: vanished mid-walk or is permission-denied must be visible as such rather
    #: than silently absent, which would read as "the agent deleted it".
    unreadable: dict[str, str] = Field(default_factory=dict)

    @property
    def paths(self) -> frozenset[str]:
        """Workspace-relative paths covered by this fingerprint."""
        return frozenset(self.files)

    @property
    def total_bytes(self) -> int:
        """Total size of all fingerprinted files."""
        return sum(item.size_bytes for item in self.files.values())


def fingerprint_file(
    path: Path,
    *,
    relative_to: Path,
    max_content_bytes: int = CONTENT_DIGEST_MAX_BYTES,
) -> FileFingerprint:
    """Fingerprint one file, hashing content only when that is affordable."""
    stat = path.stat()
    relative = path.relative_to(relative_to).as_posix()
    if stat.st_size > max_content_bytes:
        return FileFingerprint(
            path=relative,
            size_bytes=stat.st_size,
            digest=f"{stat.st_size}:{stat.st_mtime_ns}",
            method=DigestMethod.SIZE_MTIME,
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
    return FileFingerprint(
        path=relative,
        size_bytes=stat.st_size,
        digest=digest.hexdigest(),
        method=DigestMethod.SHA256,
    )


def fingerprint_workspace(
    root: Path,
    *,
    ignored_directories: Iterable[str] = IGNORED_DIRECTORIES,
    max_content_bytes: int = CONTENT_DIGEST_MAX_BYTES,
) -> WorkspaceFingerprint:
    """Fingerprint every regular file under ``root``.

    Symlinks are recorded as unreadable rather than followed. Following them
    would let a link planted inside the workspace pull a file from outside it
    into the observed state, which is the boundary this layer exists to hold.
    """
    resolved = root.resolve()
    skip = frozenset(ignored_directories)
    files: dict[str, FileFingerprint] = {}
    unreadable: dict[str, str] = {}
    if not resolved.is_dir():
        return WorkspaceFingerprint(root=str(resolved))
    for path in sorted(resolved.rglob("*")):
        if any(part in skip for part in path.relative_to(resolved).parts):
            continue
        relative = path.relative_to(resolved).as_posix()
        try:
            if path.is_symlink():
                unreadable[relative] = "symlink not followed"
                continue
            if not path.is_file():
                continue
            files[relative] = fingerprint_file(
                path,
                relative_to=resolved,
                max_content_bytes=max_content_bytes,
            )
        except OSError as error:
            unreadable[relative] = f"{type(error).__name__}: {error}"
    return WorkspaceFingerprint(
        root=str(resolved),
        files=files,
        unreadable=unreadable,
    )


def diff_workspaces(
    before: WorkspaceFingerprint,
    after: WorkspaceFingerprint,
) -> KeyDelta:
    """Report which workspace files an execution created, removed, or rewrote.

    A path lands in ``unproven`` whenever either side's identity came from the
    size-and-mtime fallback, whichever way the comparison came out.  That is not
    a hedge about the direction of the finding: it says the finding rests on a
    proxy, so a large ``.h5ad`` rewritten in place with the same size inside one
    filesystem timestamp tick could look untouched.
    """
    shared = sorted(before.paths & after.paths)
    changed = [path for path in shared if before.files[path].digest != after.files[path].digest]
    unproven = [
        path
        for path in shared
        if not (before.files[path].is_proof and after.files[path].is_proof)
    ]
    return KeyDelta(
        added=sorted(after.paths - before.paths),
        removed=sorted(before.paths - after.paths),
        changed=changed,
        unproven=unproven,
    )


__all__ = [
    "CONTENT_DIGEST_MAX_BYTES",
    "IGNORED_DIRECTORIES",
    "DigestMethod",
    "FileFingerprint",
    "WorkspaceFingerprint",
    "diff_workspaces",
    "fingerprint_file",
    "fingerprint_workspace",
]
