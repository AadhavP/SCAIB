"""Physical separation of held-out reference biology from agent-visible data.

Reference cell-type labels ship inside ``adata.obs``.  Until now the only
defence was scoring-side: record which observation columns an agent wrote and
refuse to score any other one.  That defence assumes SCAIB executes every
operation itself, so it cannot survive an agent that runs its own code -- such
an agent copies ``obs["bulk_labels"]`` into a new column and scores perfectly
while obeying the rule.

This module makes the boundary a property of the data instead.  Reference
columns are physically removed from the object the agent can reach and kept in
an evaluator-only store keyed by cell barcode.  Scoring rejoins them on that
key, which also survives the agent filtering cells away.

Stripping makes the copy attack impossible; it cannot make copying
*undetectable*, because a reference column could still reach the agent by some
route this module does not control.  So every reference column is also
fingerprinted, and candidate columns are compared against those fingerprints at
scoring time.  See :func:`detect_reference_leakage` for what counts as proof
versus what is merely evidence.

Nothing here imports numpy, pandas, or anndata at module scope: the datasets
package stays importable without the ``science`` extra.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.core.reference_columns import RESERVED_REFERENCE_COLUMNS

#: Every observation column treated as held-out reference biology. Aliases the
#: list that already governed which columns an agent may not write, so the data
#: plane and the scoring rule cannot disagree about what counts as the answer key.
REFERENCE_COLUMNS: frozenset[str] = RESERVED_REFERENCE_COLUMNS

#: Store layout. The manifest is separate from the values so an audit can read
#: the fingerprints without loading the answer key.
REFERENCE_VALUES_FILENAME = "reference.csv.gz"
REFERENCE_MANIFEST_FILENAME = "manifest.json"
STORE_FORMAT_VERSION = 1

#: Column name for the join key in the persisted store.
BARCODE_FIELD = "barcode"

#: Disagreement budget below which a candidate is reported as a suspected copy,
#: as a fraction of the cells compared.
#:
#: Expressed as a count rather than a ratio because that is what is actually
#: suspicious.  An evader only has to perturb *one* cell to defeat both digests,
#: so a pure ratio threshold cannot fire on a small dataset -- one changed cell
#: in 210 is a purity of 0.995, indistinguishable from an honest strong run.
#: The rule is therefore "disagrees on at most 0.1% of cells, but always allow
#: at least one", which fires on a single-cell perturbation at any scale while
#: leaving a genuinely excellent prediction (a few percent wrong) untouched.
SUSPECTED_COPY_MISMATCH_FRACTION = 0.001


class ReferencePartitionError(RuntimeError):
    """Raised when data cannot be partitioned soundly."""


class LeakageSeverity(StrEnum):
    """How strongly a candidate column implicates the reference.

    ``CONFIRMED`` and ``RELABELED`` are proof: no analysis reproduces the answer
    key exactly, cell for cell.  ``SUSPECTED`` is evidence only -- an excellent
    prediction and a perturbed copy are not distinguishable by agreement alone,
    so a suspicion must never be allowed to zero a score on its own.
    """

    CONFIRMED = "confirmed"
    RELABELED = "relabeled"
    SUSPECTED = "suspected"


class ColumnFingerprint(BaseModel):
    """Identity of one reference column, recorded without its values."""

    model_config = ConfigDict(extra="forbid")

    column: str
    n_values: int
    n_distinct: int
    #: Digest of the values in observation order. Catches a verbatim copy.
    ordered_digest: str
    #: Digest of the induced partition, invariant to renaming the classes.
    #: Catches ``obs["pred"] = obs["bulk_labels"].map({"B": "0", "T": "1"})``.
    partition_digest: str


class LeakageFinding(BaseModel):
    """One candidate column implicating one reference column."""

    model_config = ConfigDict(extra="forbid")

    candidate_column: str
    reference_column: str
    severity: LeakageSeverity
    evidence: str
    purity: float | None = None
    n_cells_compared: int = 0

    @property
    def is_proof(self) -> bool:
        """True when the finding establishes copying rather than suggesting it."""
        return self.severity in (LeakageSeverity.CONFIRMED, LeakageSeverity.RELABELED)


class ReferenceStoreManifest(BaseModel):
    """Auditable description of a reference store, safe to publish."""

    model_config = ConfigDict(extra="forbid")

    format_version: int = STORE_FORMAT_VERSION
    columns: list[str] = Field(default_factory=list)
    n_cells: int = 0
    #: Digest of the barcode order, so a store can be matched to its dataset.
    barcode_digest: str = ""
    fingerprints: dict[str, ColumnFingerprint] = Field(default_factory=dict)
    removed_uns_keys: list[str] = Field(default_factory=list)
    removed_obsm_keys: list[str] = Field(default_factory=list)


def _ordered_digest(values: Sequence[str]) -> str:
    """Digest values in position order."""
    digest = hashlib.sha256()
    digest.update(f"{len(values)}\n".encode())
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _partition_digest(values: Sequence[str]) -> str:
    """Digest the partition the values induce, ignoring the class names."""
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        groups.setdefault(value, []).append(index)
    ordered = sorted(groups.values(), key=lambda members: members[0])
    digest = hashlib.sha256()
    digest.update(f"{len(values)}:{len(ordered)}\n".encode())
    for members in ordered:
        digest.update(",".join(str(member) for member in members).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def fingerprint_column(column: str, values: Sequence[str]) -> ColumnFingerprint:
    """Fingerprint one column of already-stringified values."""
    return ColumnFingerprint(
        column=column,
        n_values=len(values),
        n_distinct=len(set(values)),
        ordered_digest=_ordered_digest(values),
        partition_digest=_partition_digest(values),
    )


def _as_strings(series: Any) -> list[str]:
    """Stringify an observation column without importing pandas at module scope."""
    return [str(value) for value in series.tolist()]


@dataclass(frozen=True)
class AlignedReference:
    """Reference values restricted to the cells a candidate actually retained.

    The agent is free to filter cells, so scoring must never assume the
    candidate and the reference have the same length or order.  Alignment is by
    barcode and reports its own coverage rather than silently shrinking the
    population a score is computed over.
    """

    barcodes: tuple[str, ...]
    columns: Mapping[str, tuple[str, ...]]
    #: Barcodes requested that the store does not know about. Non-empty means
    #: the candidate invented cells, which is a finding in its own right.
    unknown_barcodes: tuple[str, ...] = ()
    #: Cells the store holds that the candidate dropped.
    n_dropped: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of stored cells the candidate still accounts for."""
        total = len(self.barcodes) + self.n_dropped
        return len(self.barcodes) / total if total else 0.0

    def fingerprints(self) -> dict[str, ColumnFingerprint]:
        """Fingerprint the aligned subset, which is what leakage compares against."""
        return {
            name: fingerprint_column(name, values)
            for name, values in self.columns.items()
        }


@dataclass
class ReferencePartition:
    """A dataset split into what the agent may see and what it may not."""

    #: The agent-visible object. Reference columns are gone, not hidden.
    visible: Any
    #: Barcodes in the order they appeared in the source object.
    barcodes: tuple[str, ...]
    #: Reference column name -> values, in barcode order.
    reference: dict[str, tuple[str, ...]]
    removed_obs_columns: list[str] = field(default_factory=list)
    removed_uns_keys: list[str] = field(default_factory=list)
    removed_obsm_keys: list[str] = field(default_factory=list)

    def fingerprints(self) -> dict[str, ColumnFingerprint]:
        """Fingerprint every removed column over the full cell population."""
        return {
            name: fingerprint_column(name, values)
            for name, values in self.reference.items()
        }

    def manifest(self) -> ReferenceStoreManifest:
        """Describe this partition without disclosing any reference value."""
        return ReferenceStoreManifest(
            columns=sorted(self.reference),
            n_cells=len(self.barcodes),
            barcode_digest=_ordered_digest(self.barcodes),
            fingerprints=self.fingerprints(),
            removed_uns_keys=list(self.removed_uns_keys),
            removed_obsm_keys=list(self.removed_obsm_keys),
        )

    def align(self, barcodes: Iterable[str]) -> AlignedReference:
        """Restrict the reference to ``barcodes``, joining on the cell barcode."""
        return _align(self.barcodes, self.reference, barcodes)


def _align(
    stored_barcodes: Sequence[str],
    columns: Mapping[str, Sequence[str]],
    requested: Iterable[str],
) -> AlignedReference:
    """Join reference columns onto a requested barcode order."""
    positions = {barcode: index for index, barcode in enumerate(stored_barcodes)}
    wanted = [str(barcode) for barcode in requested]
    known = [barcode for barcode in wanted if barcode in positions]
    unknown = tuple(barcode for barcode in wanted if barcode not in positions)
    indices = [positions[barcode] for barcode in known]
    return AlignedReference(
        barcodes=tuple(known),
        columns={
            name: tuple(values[index] for index in indices)
            for name, values in columns.items()
        },
        unknown_barcodes=unknown,
        n_dropped=len(stored_barcodes) - len(known),
    )


def _grouped_by_columns(entry: Any) -> set[str]:
    """Read which observation columns an ``uns`` entry says it was computed from.

    ``params.groupby`` is scanpy's own convention for recording the grouping
    behind a derived result: ``rank_genes_groups`` and ``dendrogram_*`` both set
    it.  It is the only link back to the reference for an entry whose *name*
    discloses nothing.
    """
    params = entry.get("params") if isinstance(entry, Mapping) else None
    if not isinstance(params, Mapping):
        return set()
    grouping = params.get("groupby")
    if grouping is None:
        return set()
    if isinstance(grouping, str):
        return {grouping}
    listed = grouping.tolist() if hasattr(grouping, "tolist") else grouping
    if isinstance(listed, str):
        return {listed}
    if isinstance(listed, list | tuple | set):
        return {str(value) for value in listed}
    return {str(listed)}


def _reference_uns_keys(adata: Any, columns: Iterable[str]) -> list[str]:
    """Find ``uns`` entries that disclose a reference column.

    Three routes, because a reference column reaches ``uns`` three ways and only
    the first is visible in a key name.

    Scanpy writes ``{column}_colors`` alongside every categorical, which
    discloses the reference class names and their cardinality even after the
    column itself is dropped.  A key that merely *contains* a reference column
    name goes too; that is deliberately broad, because over-removing costs the
    agent information it was never entitled to while under-removing publishes
    the answer key, and the manifest records exactly what went.

    The third route is the one a name-based rule cannot see.
    ``uns["rank_genes_groups"]`` names no column at all, yet on the pbmc68k
    fixture it is computed with ``groupby="bulk_labels"`` -- so its field names
    *are* the withheld label vocabulary and its values are the genes that best
    separate the withheld classes.  Shipping it hands the agent both the exact
    spellings it is scored against and a near-perfect marker panel, while the
    ``obs`` check that is supposed to prove redaction worked passes cleanly.
    """
    wanted = {str(column) for column in columns}
    suffixes = ("_colors", "_categories", "_order", "_sizes")
    named = {f"{column}{suffix}" for column in wanted for suffix in suffixes}
    uns = getattr(adata, "uns", {}) or {}
    return sorted(
        str(key)
        for key in uns
        if str(key) in named
        or any(column in str(key) for column in wanted)
        or _grouped_by_columns(uns[key]) & wanted
    )


def partition_reference_columns(
    adata: Any,
    *,
    columns: Iterable[str] | None = None,
    extra_uns_keys: Iterable[str] = (),
    extra_obsm_keys: Iterable[str] = (),
    copy: bool = True,
) -> ReferencePartition:
    """Split held-out reference biology out of an AnnData object.

    Returns the agent-visible object with every reference column physically
    removed, plus the removed values keyed by barcode.  ``copy=False`` strips in
    place, which the loader may use to avoid duplicating a large matrix; the
    default does not mutate its input.
    """
    names = [str(name) for name in adata.obs_names]
    if len(set(names)) != len(names):
        raise ReferencePartitionError(
            "cell barcodes are not unique, so the reference cannot be rejoined "
            f"by barcode ({len(names) - len(set(names))} duplicate(s)). Call "
            "obs_names_make_unique() before partitioning."
        )
    wanted = frozenset(columns) if columns is not None else REFERENCE_COLUMNS
    present = [str(name) for name in adata.obs.columns if str(name) in wanted]
    reference = {name: tuple(_as_strings(adata.obs[name])) for name in present}
    uns_keys = sorted(
        {*_reference_uns_keys(adata, present), *(str(key) for key in extra_uns_keys)}
        & set(getattr(adata, "uns", {}))
    )
    obsm_keys = sorted(
        {str(key) for key in extra_obsm_keys} & set(getattr(adata, "obsm", {}))
    )
    visible = adata.copy() if copy else adata
    if present:
        visible.obs = visible.obs.drop(columns=present)
    for key in uns_keys:
        del visible.uns[key]
    for key in obsm_keys:
        del visible.obsm[key]
    return ReferencePartition(
        visible=visible,
        barcodes=tuple(names),
        reference=reference,
        removed_obs_columns=present,
        removed_uns_keys=uns_keys,
        removed_obsm_keys=obsm_keys,
    )


@dataclass(frozen=True)
class ReferenceStore:
    """A reference partition read back from the evaluator-only directory."""

    manifest: ReferenceStoreManifest
    barcodes: tuple[str, ...]
    columns: Mapping[str, tuple[str, ...]]

    def align(self, barcodes: Iterable[str]) -> AlignedReference:
        """Restrict the reference to ``barcodes``, joining on the cell barcode."""
        return _align(self.barcodes, self.columns, barcodes)


def write_reference_store(partition: ReferencePartition, directory: Path) -> Path:
    """Persist the reference outside the agent-visible workspace.

    Gzipped CSV rather than parquet on purpose: neither pyarrow nor fastparquet
    is a dependency of this project, and the store must be writable in a bare
    ``science`` install.
    """
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / REFERENCE_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(partition.manifest().model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    names = sorted(partition.reference)
    values_path = directory / REFERENCE_VALUES_FILENAME
    with gzip.open(values_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([BARCODE_FIELD, *names])
        for index, barcode in enumerate(partition.barcodes):
            writer.writerow(
                [barcode, *(partition.reference[name][index] for name in names)]
            )
    return directory


def read_reference_store(directory: Path) -> ReferenceStore:
    """Load a reference store written by :func:`write_reference_store`."""
    manifest_path = directory / REFERENCE_MANIFEST_FILENAME
    values_path = directory / REFERENCE_VALUES_FILENAME
    if not manifest_path.is_file() or not values_path.is_file():
        raise ReferencePartitionError(
            f"reference store at '{directory}' is incomplete; expected "
            f"{REFERENCE_MANIFEST_FILENAME} and {REFERENCE_VALUES_FILENAME}"
        )
    manifest = ReferenceStoreManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    barcodes: list[str] = []
    collected: dict[str, list[str]] = {name: [] for name in manifest.columns}
    with gzip.open(values_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            barcodes.append(row[BARCODE_FIELD])
            for name in manifest.columns:
                collected[name].append(row[name])
    store = ReferenceStore(
        manifest=manifest,
        barcodes=tuple(barcodes),
        columns={name: tuple(values) for name, values in collected.items()},
    )
    _verify_store(store)
    return store


def _verify_store(store: ReferenceStore) -> None:
    """Re-derive the fingerprints instead of trusting the recorded ones."""
    if _ordered_digest(store.barcodes) != store.manifest.barcode_digest:
        raise ReferencePartitionError(
            "reference store barcodes do not match the manifest digest; the "
            "store has been modified or was written for another dataset"
        )
    for name, expected in store.manifest.fingerprints.items():
        observed = fingerprint_column(name, store.columns.get(name, ()))
        if observed.ordered_digest != expected.ordered_digest:
            raise ReferencePartitionError(
                f"reference column '{name}' does not match its recorded fingerprint"
            )


def _mismatches(candidate: Sequence[str], reference: Sequence[str]) -> int:
    """Cells falling outside their candidate class's dominant reference class.

    The complement of this count is *purity*, an upper bound on the agreement any
    relabeling could achieve -- not an accuracy, since splitting a class raises
    purity while lowering accuracy.  Using the upper bound means this
    over-reports rather than under-reports, which is the right direction for a
    signal that never penalizes on its own.
    """
    tallies: dict[str, dict[str, int]] = {}
    for candidate_value, reference_value in zip(candidate, reference, strict=True):
        counts = tallies.setdefault(candidate_value, {})
        counts[reference_value] = counts.get(reference_value, 0) + 1
    matched = sum(max(counts.values()) for counts in tallies.values())
    return len(candidate) - matched


def _mismatch_budget(n_cells: int) -> int:
    """Largest disagreement count still treated as suspicious at this scale."""
    return max(1, int(n_cells * SUSPECTED_COPY_MISMATCH_FRACTION))


def detect_reference_leakage(
    candidates: Mapping[str, Sequence[str]],
    reference: AlignedReference,
) -> list[LeakageFinding]:
    """Compare agent-produced columns against the aligned reference.

    Both sides must already be restricted to the same cells in the same order,
    which is what :meth:`ReferenceStore.align` produces -- comparing a filtered
    candidate against the full reference would miss every copy.

    Exact digest equality is reported as proof.  Near-perfect agreement is
    reported as suspicion and deliberately nothing stronger: a genuinely
    excellent annotation and a lightly perturbed copy of the answer key are
    indistinguishable by agreement, so treating agreement as proof would
    penalize the best agents hardest.
    """
    reference_prints = reference.fingerprints()
    findings: list[LeakageFinding] = []
    for candidate_name, raw_values in candidates.items():
        values = [str(value) for value in raw_values]
        if len(values) != len(reference.barcodes):
            continue
        candidate_print = fingerprint_column(candidate_name, values)
        for reference_name, reference_print in reference_prints.items():
            finding = _compare(
                candidate_print,
                values,
                reference_print,
                reference.columns[reference_name],
            )
            if finding is not None:
                findings.append(finding)
    return findings


def _compare(
    candidate_print: ColumnFingerprint,
    candidate_values: Sequence[str],
    reference_print: ColumnFingerprint,
    reference_values: Sequence[str],
) -> LeakageFinding | None:
    """Classify one candidate/reference column pair."""

    def finding(
        severity: LeakageSeverity, evidence: str, purity: float
    ) -> LeakageFinding:
        return LeakageFinding(
            candidate_column=candidate_print.column,
            reference_column=reference_print.column,
            n_cells_compared=len(candidate_values),
            severity=severity,
            evidence=evidence,
            purity=purity,
        )

    if candidate_print.ordered_digest == reference_print.ordered_digest:
        return finding(
            LeakageSeverity.CONFIRMED,
            "candidate values are identical to the reference, cell for cell",
            1.0,
        )
    if candidate_print.partition_digest == reference_print.partition_digest:
        return finding(
            LeakageSeverity.RELABELED,
            "candidate induces exactly the reference partition under a renaming "
            "of the classes",
            1.0,
        )
    # Purity is trivially high when a candidate splits the reference into many
    # small groups, so over-clustering must not read as copying.
    if candidate_print.n_distinct > reference_print.n_distinct:
        return None
    if not candidate_values:
        return None
    mismatches = _mismatches(candidate_values, reference_values)
    if mismatches > _mismatch_budget(len(candidate_values)):
        return None
    return finding(
        LeakageSeverity.SUSPECTED,
        f"candidate disagrees with the reference on {mismatches} of "
        f"{len(candidate_values)} cells using {candidate_print.n_distinct} "
        f"classes against {reference_print.n_distinct}; this is evidence, not "
        "proof, and must not reduce a score on its own",
        1.0 - mismatches / len(candidate_values),
    )


__all__ = [
    "BARCODE_FIELD",
    "REFERENCE_COLUMNS",
    "REFERENCE_MANIFEST_FILENAME",
    "REFERENCE_VALUES_FILENAME",
    "STORE_FORMAT_VERSION",
    "SUSPECTED_COPY_MISMATCH_FRACTION",
    "AlignedReference",
    "ColumnFingerprint",
    "LeakageFinding",
    "LeakageSeverity",
    "ReferencePartition",
    "ReferencePartitionError",
    "ReferenceStore",
    "ReferenceStoreManifest",
    "detect_reference_leakage",
    "fingerprint_column",
    "partition_reference_columns",
    "read_reference_store",
    "write_reference_store",
]
