"""Content fingerprints for an annotated single-cell dataset.

The workspace fingerprint next door answers "which files changed".  This module
answers the question that actually carries the science: *what did the step do to
the data*.  A step that adds ``obs["leiden"]`` and one that overwrites ``X`` both
rewrite the same ``.h5ad``, and a file-level digest cannot tell them apart.

Two constraints shape everything here.

**It runs twice per step, so it has to be cheap.**  A fingerprint is taken
before and after every execution, so cost lands directly on top of the science
rather than beside it.  Columns are small and are always digested completely.
Matrices are not: above a byte budget they are digested from a deterministic
stride sample plus shape, dtype and non-zero count, and the result is labelled
:attr:`DigestScope.SAMPLED` so no consumer mistakes it for proof.  This is the
same bargain :mod:`~agent_evals.environment.execution.fingerprint` strikes for
large files, and it is recorded for the same reason: a benchmark that quietly
downgrades its own evidence is worse than one that never had it.

**It must never raise.**  Fingerprinting is observation, not science.  A column
with an exotic dtype or a matrix type nothing here anticipated is recorded as a
limitation on the fingerprint, because a traceback out of this module would
surface as a harness failure blamed on the agent's code.

Under-reporting change is the dangerous direction: a missed change becomes an
agent claim nobody contradicted.  Over-reporting is merely noisy.  Where the two
trade off -- byte-level float comparison, per-value digests over a filtered cell
set -- this module chooses to over-report.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import Field

from agent_evals.environment.models import KeyDelta, RuntimeModel, StateDelta

#: Per-array byte budget for a complete digest.  Above it, an array is sampled.
#: Sized so a fingerprint of one dataset with a matrix and a few embeddings stays
#: well under a tenth of a second on commodity hardware, twice per step.
ARRAY_DIGEST_MAX_BYTES = 64 * 1024 * 1024

#: Buffers that together determine a sparse matrix's contents.  Covers the CSR
#: and CSC layouts (``indices``/``indptr``) and COO (``row``/``col``); absent
#: names are skipped, so an unfamiliar layout degrades to whatever it exposes.
_SPARSE_BUFFERS = ("data", "indices", "indptr", "row", "col")

_VALUE_SEPARATOR = "\x00"

#: The namespaces a dataset fingerprint can speak to, drawn from
#: :data:`~agent_evals.environment.models.STATE_NAMESPACES`.  ``files`` is
#: deliberately absent: this module can see a dataset but never a directory tree,
#: so it is never in a position to report the file namespace as observed.
DATASET_NAMESPACES = ("obs", "var", "obsm", "layers", "matrix")


class DigestScope(StrEnum):
    """How much of a value contributed to its digest."""

    #: Every value contributed. Digest equality proves the contents match.
    COMPLETE = "complete"
    #: A deterministic subset contributed, alongside shape and dtype. Digest
    #: equality is evidence the contents match, not proof: a change confined to
    #: unsampled entries is invisible.
    SAMPLED = "sampled"


class ColumnFingerprint(RuntimeModel):
    """Identity of one ``obs`` or ``var`` column at one point in time."""

    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    n_missing: int = Field(default=0, ge=0)
    #: Distinct non-null values. Carried because cardinality is the difference
    #: between a clustering that found structure and one that found one blob.
    n_unique: int | None = Field(default=None, ge=0)


class ArrayFingerprint(RuntimeModel):
    """Identity of a matrix, embedding, or layer at one point in time."""

    key: str = Field(min_length=1)
    shape: list[int] = Field(default_factory=list)
    dtype: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    scope: DigestScope = DigestScope.COMPLETE
    #: Stored non-zeros, when the array is sparse and reports them.
    stored_values: int | None = Field(default=None, ge=0)

    @property
    def is_proof(self) -> bool:
        """Whether comparing this digest can prove the contents are identical."""
        return self.scope is DigestScope.COMPLETE


class DatasetFingerprint(RuntimeModel):
    """Everything observation can cheaply say about a dataset's current state."""

    n_obs: int = Field(default=0, ge=0)
    n_vars: int = Field(default=0, ge=0)
    #: Digests of the row and column indexes. These are what make a reordering
    #: or a substitution of cells visible; a count comparison alone would miss
    #: both, and both break the barcode join that scoring depends on.
    obs_names_digest: str = ""
    var_names_digest: str = ""
    obs: dict[str, ColumnFingerprint] = Field(default_factory=dict)
    var: dict[str, ColumnFingerprint] = Field(default_factory=dict)
    obsm: dict[str, ArrayFingerprint] = Field(default_factory=dict)
    layers: dict[str, ArrayFingerprint] = Field(default_factory=dict)
    matrix: ArrayFingerprint | None = None
    #: What this fingerprint could not establish, and why.
    limitations: list[str] = Field(default_factory=list)


def _digest_bytes(payload: bytes) -> str:
    """Return the SHA-256 of a byte payload."""
    return hashlib.sha256(payload).hexdigest()


def _value_bytes(values: Any) -> bytes:
    """Return canonical bytes for a numpy array of any dtype.

    Numeric arrays are hashed as raw memory, which distinguishes ``-0.0`` from
    ``0.0`` and one NaN payload from another.  That over-reports change in cases
    no real pipeline produces, and over-reporting is the safe direction.
    """
    import numpy as np

    array = np.asarray(values)
    if array.dtype.kind in "biufc":
        return np.ascontiguousarray(array).tobytes()
    if array.dtype.kind in "mM":
        return np.ascontiguousarray(array.view("i8")).tobytes()
    return _VALUE_SEPARATOR.join(str(item) for item in array.ravel().tolist()).encode("utf-8")


def _digest_values(
    values: Any,
    *,
    max_bytes: int,
) -> tuple[str, DigestScope]:
    """Digest an array completely, or from a stride sample when it is large."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(values)).ravel()
    header = f"{array.shape}|{array.dtype}".encode()
    if array.nbytes <= max_bytes:
        return _digest_bytes(header + _value_bytes(array)), DigestScope.COMPLETE
    stride = array.nbytes // max_bytes + 1
    sample = array[::stride]
    payload = header + f"|stride={stride}".encode() + _value_bytes(sample)
    return _digest_bytes(payload), DigestScope.SAMPLED


def _is_sparse(matrix: Any) -> bool:
    """Return whether a matrix exposes the sparse buffer protocol."""
    return hasattr(matrix, "nnz") and hasattr(matrix, "data")


def fingerprint_array(
    key: str,
    matrix: Any,
    *,
    max_bytes: int = ARRAY_DIGEST_MAX_BYTES,
) -> ArrayFingerprint:
    """Fingerprint one dense or sparse array."""
    shape = [int(dimension) for dimension in getattr(matrix, "shape", ()) or ()]
    dtype = str(getattr(matrix, "dtype", type(matrix).__name__))
    if not _is_sparse(matrix):
        digest, scope = _digest_values(matrix, max_bytes=max_bytes)
        return ArrayFingerprint(key=key, shape=shape, dtype=dtype, digest=digest, scope=scope)
    parts = [f"{shape}|{dtype}|nnz={int(matrix.nnz)}".encode()]
    scope = DigestScope.COMPLETE
    for name in _SPARSE_BUFFERS:
        buffer = getattr(matrix, name, None)
        if buffer is None:
            continue
        digest, buffer_scope = _digest_values(buffer, max_bytes=max_bytes)
        parts.append(f"|{name}={digest}".encode())
        if buffer_scope is DigestScope.SAMPLED:
            scope = DigestScope.SAMPLED
    return ArrayFingerprint(
        key=key,
        shape=shape,
        dtype=dtype,
        digest=_digest_bytes(b"".join(parts)),
        scope=scope,
        stored_values=int(matrix.nnz),
    )


def fingerprint_column(name: str, series: Any) -> ColumnFingerprint:
    """Fingerprint one ``obs`` or ``var`` column.

    Columns are always digested completely.  They are one value per cell or per
    gene, which is small next to the expression matrix, and they are where the
    agent's own annotations land -- exactly the values whose provenance the
    benchmark has to be able to prove rather than merely suggest.
    """
    values = series.to_numpy()
    digest, _ = _digest_values(values, max_bytes=values.nbytes + 1)
    try:
        n_unique: int | None = int(series.nunique(dropna=True))
    except (TypeError, ValueError):  # unhashable values; cardinality is optional
        n_unique = None
    return ColumnFingerprint(
        name=name,
        dtype=str(series.dtype),
        digest=digest,
        n_missing=int(series.isna().sum()),
        n_unique=n_unique,
    )


def _fingerprint_frame(
    frame: Any,
    *,
    label: str,
    limitations: list[str],
) -> dict[str, ColumnFingerprint]:
    """Fingerprint every column of ``obs`` or ``var``, tolerating bad columns."""
    fingerprints: dict[str, ColumnFingerprint] = {}
    for position, raw_name in enumerate(frame.columns):
        name = str(raw_name)
        if name in fingerprints:
            limitations.append(f"{label} column '{name}' is duplicated; only the first was read")
            continue
        try:
            fingerprints[name] = fingerprint_column(name, frame.iloc[:, position])
        except Exception as error:  # observation must not fail the science
            limitations.append(f"{label} column '{name}' unreadable: {type(error).__name__}")
    return fingerprints


def _fingerprint_mapping(
    mapping: Any,
    *,
    label: str,
    max_bytes: int,
    limitations: list[str],
) -> dict[str, ArrayFingerprint]:
    """Fingerprint every array in ``obsm`` or ``layers``, tolerating bad ones."""
    fingerprints: dict[str, ArrayFingerprint] = {}
    for raw_key in list(mapping.keys()):
        key = str(raw_key)
        try:
            fingerprints[key] = fingerprint_array(key, mapping[raw_key], max_bytes=max_bytes)
        except Exception as error:  # observation must not fail the science
            limitations.append(f"{label}['{key}'] unreadable: {type(error).__name__}")
    return fingerprints


def _digest_index(index: Any) -> str:
    """Digest a row or column index by its values."""
    digest, _ = _digest_values(index.to_numpy(), max_bytes=1 << 62)
    return digest


def fingerprint_dataset(
    adata: Any,
    *,
    max_digest_bytes: int = ARRAY_DIGEST_MAX_BYTES,
    read_matrix: bool = True,
) -> DatasetFingerprint:
    """Fingerprint an ``AnnData`` object without ever raising.

    Every failure to read part of the object becomes a limitation on the result
    rather than an exception, because this runs inside the step loop and a crash
    here would be reported as the agent's execution failing.

    ``read_matrix=False`` is for a dataset opened in backed mode, where reading
    ``X`` means streaming it off disk.  It leaves :attr:`DatasetFingerprint.matrix`
    as ``None`` rather than digesting a placeholder, so the matrix reads as
    unobserved instead of as unchanged.
    """
    limitations: list[str] = []
    obs = _fingerprint_frame(adata.obs, label="obs", limitations=limitations)
    var = _fingerprint_frame(adata.var, label="var", limitations=limitations)
    obsm = _fingerprint_mapping(
        adata.obsm,
        label="obsm",
        max_bytes=max_digest_bytes,
        limitations=limitations,
    )
    layers = _fingerprint_mapping(
        adata.layers,
        label="layers",
        max_bytes=max_digest_bytes,
        limitations=limitations,
    )
    matrix: ArrayFingerprint | None = None
    if not read_matrix:
        limitations.append("X was not read, so expression changes are unobserved")
    elif getattr(adata, "X", None) is not None:
        try:
            matrix = fingerprint_array("X", adata.X, max_bytes=max_digest_bytes)
        except Exception as error:  # observation must not fail the science
            limitations.append(f"X unreadable: {type(error).__name__}")
    else:
        limitations.append("X is absent; expression changes cannot be observed")
    if matrix is not None and matrix.scope is DigestScope.SAMPLED:
        limitations.append(
            "X was digested from a stride sample; an unchanged verdict is "
            "evidence rather than proof"
        )
    return DatasetFingerprint(
        n_obs=int(adata.n_obs),
        n_vars=int(adata.n_vars),
        obs_names_digest=_digest_index(adata.obs_names),
        var_names_digest=_digest_index(adata.var_names),
        obs=obs,
        var=var,
        obsm=obsm,
        layers=layers,
        matrix=matrix,
        limitations=limitations,
    )


def _diff_columns(
    before: dict[str, ColumnFingerprint],
    after: dict[str, ColumnFingerprint],
) -> KeyDelta:
    """Diff two column namespaces. Column digests are always complete."""
    shared = sorted(set(before) & set(after))
    return KeyDelta(
        added=sorted(set(after) - set(before)),
        removed=sorted(set(before) - set(after)),
        changed=[name for name in shared if before[name].digest != after[name].digest],
    )


def _diff_arrays(
    before: dict[str, ArrayFingerprint],
    after: dict[str, ArrayFingerprint],
) -> KeyDelta:
    """Diff two array namespaces, flagging verdicts that rest on a sample."""
    shared = sorted(set(before) & set(after))
    return KeyDelta(
        added=sorted(set(after) - set(before)),
        removed=sorted(set(before) - set(after)),
        changed=[key for key in shared if before[key].digest != after[key].digest],
        unproven=[key for key in shared if not (before[key].is_proof and after[key].is_proof)],
    )


def _diff_matrix(
    before: ArrayFingerprint | None,
    after: ArrayFingerprint | None,
) -> bool | None:
    """Report whether the expression matrix changed, or that it is unknown."""
    if before is None or after is None:
        return None
    if before.digest != after.digest:
        return True
    if before.is_proof and after.is_proof:
        return False
    return None


def diff_datasets(
    before: DatasetFingerprint,
    after: DatasetFingerprint,
    *,
    files: KeyDelta | None = None,
) -> StateDelta:
    """Report what observation says happened to the dataset between two points.

    When the cell set changes, every ``obs`` column's values change with it, so
    ``obs.changed`` fills up as a consequence of the filter rather than as
    evidence of separate edits.  That is recorded as a limitation instead of
    being papered over, because ``added`` and ``removed`` -- the fields that
    carry provenance -- stay meaningful either way.

    ``files`` is accepted here so a caller that observed both the workspace and
    the dataset produces one delta rather than two that a consumer has to know
    to merge.  Omitting it marks the file namespace unobserved, because this
    module can see a dataset but never a directory tree.
    """
    limitations = sorted({*before.limitations, *after.limitations})
    unobserved = [] if files is not None else ["files"]
    if before.matrix is None or after.matrix is None:
        unobserved.append("matrix")
    obs_names_changed = before.obs_names_digest != after.obs_names_digest
    var_names_changed = before.var_names_digest != after.var_names_digest
    if obs_names_changed:
        limitations.append(
            "the cell set changed, so every obs column digest differs as a "
            "consequence; read obs.added and obs.removed rather than obs.changed"
        )
    if var_names_changed:
        limitations.append(
            "the gene set changed, so every var column digest differs as a "
            "consequence; read var.added and var.removed rather than var.changed"
        )
    return StateDelta(
        n_obs_before=before.n_obs,
        n_obs_after=after.n_obs,
        n_vars_before=before.n_vars,
        n_vars_after=after.n_vars,
        obs=_diff_columns(before.obs, after.obs),
        var=_diff_columns(before.var, after.var),
        obsm=_diff_arrays(before.obsm, after.obsm),
        layers=_diff_arrays(before.layers, after.layers),
        files=files if files is not None else KeyDelta(),
        matrix_changed=_diff_matrix(before.matrix, after.matrix),
        obs_names_changed=obs_names_changed,
        var_names_changed=var_names_changed,
        unobserved=sorted(unobserved),
        limitations=limitations,
    )


def dataset_delta(
    before: DatasetFingerprint | None,
    after: DatasetFingerprint | None,
    *,
    files: KeyDelta | None = None,
) -> StateDelta:
    """Diff two dataset fingerprints, tolerating either one being absent.

    A missing fingerprint becomes an unobserved namespace rather than an empty
    diff.  That distinction is the point of the whole layer: an empty ``obs``
    delta from a dataset that *was* read means the step added no columns, while
    the same empty delta from a dataset nobody could read means nothing at all,
    and a consumer that conflated them would either manufacture a discrepancy or
    hide one.
    """
    if before is not None and after is not None:
        return diff_datasets(before, after, files=files)
    unobserved = set(DATASET_NAMESPACES)
    if files is None:
        unobserved.add("files")
    return StateDelta(
        files=files if files is not None else KeyDelta(),
        unobserved=sorted(unobserved),
        limitations=["the dataset could not be read, so its changes are unobserved"],
    )


def written_obs_columns(delta: StateDelta) -> list[str]:
    """Return the ``obs`` columns an observed step wrote.

    This is the provenance rule both execution tiers share, and it exists as one
    function because the tiers must not be able to disagree about it: a column
    only counts as the agent's own work if the step that ran wrote it, and
    scoring uses that answer to decide what it is allowed to score at all.

    ``added`` is always attribution.  ``changed`` is attribution only while the
    cell set held still.  Once cells are filtered every surviving column's values
    change along with them, so ``changed`` names the whole frame rather than the
    columns the step touched -- and attributing the whole frame would hand the
    agent credit for the reference labels it is being scored against.  When the
    barcodes moved, or when nobody could tell whether they moved, the narrower
    reading is the only safe one.

    Whether the step's work is *the agent's* work is the caller's judgement, not
    this function's: under free execution the agent ran the code itself, and
    under typed execution the harness ran it on the agent's instruction.
    """
    if not delta.is_observed("obs"):
        return []
    if delta.obs_names_changed is not False:
        return sorted(delta.obs.added)
    return sorted({*delta.obs.added, *delta.obs.changed})


__all__ = [
    "ARRAY_DIGEST_MAX_BYTES",
    "DATASET_NAMESPACES",
    "ArrayFingerprint",
    "ColumnFingerprint",
    "DatasetFingerprint",
    "DigestScope",
    "dataset_delta",
    "diff_datasets",
    "fingerprint_array",
    "fingerprint_column",
    "fingerprint_dataset",
    "written_obs_columns",
]
