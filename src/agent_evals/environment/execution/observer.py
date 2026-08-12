"""Take a dataset fingerprint from whatever holds the dataset.

Under free execution the harness does not hold an ``AnnData`` object -- the agent
does, inside its own process, and all the harness has afterwards is a file.  That
asymmetry is why observation is a separate injectable capability rather than
something the executor does inline: the workspace tier reads an ``.h5ad`` off
disk, the typed tier already has the object in memory, and the port lets both
feed the same diff.

The return type is deliberately ``DatasetFingerprint | None``.  ``None`` means
*this observer could not look*, which is a different statement from an empty
fingerprint, and the delta downstream treats it as such.  An observer that
guessed instead of returning ``None`` would manufacture evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from agent_evals.environment.execution.dataset import (
    ARRAY_DIGEST_MAX_BYTES,
    DatasetFingerprint,
    fingerprint_dataset,
)

#: Datasets at or below this size are read into memory so ``X`` contributes to
#: the fingerprint.  Above it, the file is opened backed and the matrix is left
#: unobserved rather than streamed twice per step.
DATASET_READ_MAX_BYTES = 512 * 1024 * 1024


@runtime_checkable
class DatasetObserver(Protocol):
    """Port for taking a dataset fingerprint at a point in time."""

    def snapshot(self) -> DatasetFingerprint | None:
        """Fingerprint the dataset now, or return ``None`` if it cannot look."""
        ...


class H5adDatasetObserver:
    """Fingerprint the agent-visible dataset by reading it off disk.

    The path is the *sanitized* dataset materialized into the workspace inputs
    directory, never the evaluator-side reference store.  Nothing here enforces
    that -- the filesystem boundary from Stage 0 does -- but pointing this at the
    reference store would pull hidden values into an observation, so the caller
    that constructs it owns that choice.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_digest_bytes: int = ARRAY_DIGEST_MAX_BYTES,
        max_read_bytes: int = DATASET_READ_MAX_BYTES,
    ) -> None:
        self.path = path
        self.max_digest_bytes = max_digest_bytes
        self.max_read_bytes = max_read_bytes

    def snapshot(self) -> DatasetFingerprint | None:
        """Read and fingerprint the dataset, returning ``None`` on any failure.

        A missing file is an ordinary outcome, not an error: before the agent's
        first step the dataset it will write may not exist yet, and a step that
        deleted it is information the diff should carry rather than crash on.
        """
        import anndata  # optional 'science' extra; absent from a core install

        if not self.path.is_file():
            return None
        try:
            size = self.path.stat().st_size
            read_matrix = size <= self.max_read_bytes
            adata = (
                anndata.read_h5ad(self.path)
                if read_matrix
                else anndata.read_h5ad(self.path, backed="r")
            )
        except Exception:  # unreadable, truncated, or not an .h5ad at all
            return None
        try:
            fingerprint = fingerprint_dataset(
                adata,
                max_digest_bytes=self.max_digest_bytes,
                read_matrix=read_matrix,
            )
        finally:
            handle = getattr(adata, "file", None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:  # nothing useful to do about a failed close
                    pass
        if not read_matrix:
            fingerprint.limitations.append(
                f"dataset is {size} bytes, above the {self.max_read_bytes}-byte "
                "read budget, so it was opened backed"
            )
        return fingerprint


class InMemoryDatasetObserver:
    """Fingerprint an ``AnnData`` object the harness already holds.

    This is the typed-action tier's observer.  It exists so the obs-column
    provenance that ``ScanpyExecutor`` used to compute by set-differencing comes
    from the same mechanism as the free-execution tier, rather than the two
    tiers establishing provenance in ways that could disagree.
    """

    def __init__(
        self,
        adata: object,
        *,
        max_digest_bytes: int = ARRAY_DIGEST_MAX_BYTES,
    ) -> None:
        self.adata = adata
        self.max_digest_bytes = max_digest_bytes

    def snapshot(self) -> DatasetFingerprint | None:
        """Fingerprint the held object, returning ``None`` if it is not one."""
        try:
            return fingerprint_dataset(self.adata, max_digest_bytes=self.max_digest_bytes)
        except AttributeError:  # not an AnnData-shaped object
            return None


__all__ = [
    "DATASET_READ_MAX_BYTES",
    "DatasetObserver",
    "H5adDatasetObserver",
    "InMemoryDatasetObserver",
]
