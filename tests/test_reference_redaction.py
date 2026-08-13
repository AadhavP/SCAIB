"""Reference biology is removed from the data plane, not merely left unscored.

`test_reference_label_isolation.py` covers the scoring-side rule: only score a
column the agent wrote.  That rule assumes SCAIB runs every operation.  These
tests cover the data-plane guarantee that has to hold once the agent runs its
own code, plus the detection half -- stripping stops the copy attack, and the
fingerprints catch a copy that arrives by some other route.
"""

from pathlib import Path

import pytest

pytest.importorskip("anndata")

import anndata
import numpy as np
import pandas as pd

from agent_evals.datasets.redaction import (
    LeakageSeverity,
    ReferencePartitionError,
    detect_reference_leakage,
    partition_reference_columns,
    read_reference_store,
    write_reference_store,
)

REFERENCE = ["B", "T", "NK"]


def _adata(cells: int = 210) -> anndata.AnnData:
    """Build an object shaped like the real datasets: reference labels in obs."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "bulk_labels": pd.Categorical([REFERENCE[i % 3] for i in range(cells)]),
            "louvain": pd.Categorical([str(i % 5) for i in range(cells)]),
            "batch": pd.Categorical(["a" if i % 2 else "b" for i in range(cells)]),
        },
        index=[f"cell-{i}" for i in range(cells)],
    )
    adata = anndata.AnnData(
        X=rng.random((cells, 4), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["CD3D", "MS4A1", "GNLY", "LYZ"]),
    )
    # Scanpy writes these next to a categorical; they disclose the class names.
    adata.uns["bulk_labels_colors"] = np.array(["#1f77b4", "#ff7f0e", "#2ca02c"])
    adata.uns["louvain_colors"] = np.array(["#111111"] * 5)
    return adata


def test_reference_columns_are_physically_absent_from_the_visible_object() -> None:
    partition = partition_reference_columns(_adata())

    assert "bulk_labels" not in partition.visible.obs.columns
    assert partition.removed_obs_columns == ["bulk_labels"]
    # A column the agent legitimately needs must survive.
    assert "batch" in partition.visible.obs.columns
    assert "louvain" in partition.visible.obs.columns


def test_reference_class_names_do_not_survive_in_uns() -> None:
    """Dropping the column is not enough: `{column}_colors` leaks the vocabulary."""
    partition = partition_reference_columns(_adata())

    assert "bulk_labels_colors" not in partition.visible.uns
    assert partition.removed_uns_keys == ["bulk_labels_colors"]
    # Unrelated colour keys are not collateral damage.
    assert "louvain_colors" in partition.visible.uns


def test_partitioning_does_not_mutate_the_source_object() -> None:
    adata = _adata()
    partition_reference_columns(adata)

    assert "bulk_labels" in adata.obs.columns
    assert "bulk_labels_colors" in adata.uns


def test_duplicate_barcodes_are_refused_rather_than_joined_wrongly() -> None:
    adata = _adata(cells=6)
    adata.obs_names = ["a", "a", "b", "c", "d", "e"]

    with pytest.raises(ReferencePartitionError, match="not unique"):
        partition_reference_columns(adata)


def test_evaluator_can_still_score_after_the_agent_filters_cells() -> None:
    """The join is by barcode because QC legitimately removes cells."""
    partition = partition_reference_columns(_adata())
    retained = [f"cell-{i}" for i in range(0, 210, 2)]

    aligned = partition.align(retained)

    assert aligned.barcodes == tuple(retained)
    assert len(aligned.columns["bulk_labels"]) == len(retained)
    assert aligned.coverage == pytest.approx(0.5)
    assert aligned.n_dropped == 105
    # Values follow the barcode, not the position.
    assert aligned.columns["bulk_labels"][1] == REFERENCE[2 % 3]


def test_cells_the_store_never_saw_are_reported_not_silently_dropped() -> None:
    partition = partition_reference_columns(_adata(cells=9))

    aligned = partition.align(["cell-0", "invented-1", "cell-3"])

    assert aligned.barcodes == ("cell-0", "cell-3")
    assert aligned.unknown_barcodes == ("invented-1",)


def test_store_round_trips_outside_the_workspace(tmp_path: Path) -> None:
    partition = partition_reference_columns(_adata())
    directory = write_reference_store(partition, tmp_path / "evaluator")

    store = read_reference_store(directory)

    assert store.barcodes == partition.barcodes
    assert store.columns["bulk_labels"] == partition.reference["bulk_labels"]
    assert store.manifest.columns == ["bulk_labels"]
    assert store.manifest.n_cells == 210


def test_manifest_records_fingerprints_without_disclosing_values(
    tmp_path: Path,
) -> None:
    """The manifest is publishable; only the values file is the answer key."""
    partition = partition_reference_columns(_adata())
    directory = write_reference_store(partition, tmp_path / "evaluator")

    manifest = (directory / "manifest.json").read_text(encoding="utf-8")

    assert "bulk_labels" in manifest
    for label in REFERENCE:
        assert f'"{label}"' not in manifest


def test_a_tampered_store_is_refused_rather_than_trusted(tmp_path: Path) -> None:
    import gzip

    partition = partition_reference_columns(_adata(cells=9))
    directory = write_reference_store(partition, tmp_path / "evaluator")
    path = directory / "reference.csv.gz"
    original = gzip.open(path, "rt", encoding="utf-8").read()
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(original.replace("NK", "B"))

    with pytest.raises(ReferencePartitionError, match="fingerprint"):
        read_reference_store(directory)


def test_a_verbatim_copy_of_the_reference_is_proven(tmp_path: Path) -> None:
    partition = partition_reference_columns(_adata())
    aligned = partition.align(partition.barcodes)
    copied = list(aligned.columns["bulk_labels"])

    findings = detect_reference_leakage({"predicted_labels": copied}, aligned)

    assert [finding.severity for finding in findings] == [LeakageSeverity.CONFIRMED]
    assert findings[0].is_proof
    assert findings[0].reference_column == "bulk_labels"


def test_a_renamed_copy_of_the_reference_is_proven() -> None:
    """Renaming the classes changes every value but not the partition."""
    partition = partition_reference_columns(_adata())
    aligned = partition.align(partition.barcodes)
    renamed = {"B": "cluster_0", "T": "cluster_1", "NK": "cluster_2"}
    disguised = [renamed[value] for value in aligned.columns["bulk_labels"]]

    findings = detect_reference_leakage({"predicted_labels": disguised}, aligned)

    assert [finding.severity for finding in findings] == [LeakageSeverity.RELABELED]
    assert findings[0].is_proof


def test_a_copy_survives_the_agent_filtering_cells() -> None:
    """Comparing a filtered candidate against the full reference would miss this."""
    partition = partition_reference_columns(_adata())
    retained = [f"cell-{i}" for i in range(0, 210, 3)]
    aligned = partition.align(retained)
    copied = list(aligned.columns["bulk_labels"])

    findings = detect_reference_leakage({"predicted_labels": copied}, aligned)

    assert [finding.severity for finding in findings] == [LeakageSeverity.CONFIRMED]
    assert findings[0].n_cells_compared == len(retained)


def test_a_genuinely_good_prediction_is_not_flagged() -> None:
    """The false-positive half is the one that matters: a strong agent must pass."""
    partition = partition_reference_columns(_adata())
    aligned = partition.align(partition.barcodes)
    truth = list(aligned.columns["bulk_labels"])
    # 90% correct: excellent by any benchmark standard, and not a copy.
    prediction = [
        value if index % 10 else REFERENCE[(REFERENCE.index(value) + 1) % 3]
        for index, value in enumerate(truth)
    ]

    findings = detect_reference_leakage({"predicted_labels": prediction}, aligned)

    assert findings == []


def test_over_clustering_is_not_mistaken_for_copying() -> None:
    """Many small clusters have high purity by construction, not by leakage."""
    partition = partition_reference_columns(_adata())
    aligned = partition.align(partition.barcodes)
    fine = [f"c{index % 30}" for index in range(len(aligned.barcodes))]

    findings = detect_reference_leakage({"leiden": fine}, aligned)

    assert findings == []


def test_a_near_perfect_copy_is_suspected_but_never_proven() -> None:
    """One perturbed cell defeats both digests, so only agreement is left."""
    partition = partition_reference_columns(_adata())
    aligned = partition.align(partition.barcodes)
    perturbed = list(aligned.columns["bulk_labels"])
    perturbed[0] = "NK" if perturbed[0] != "NK" else "B"

    findings = detect_reference_leakage({"predicted_labels": perturbed}, aligned)

    assert [finding.severity for finding in findings] == [LeakageSeverity.SUSPECTED]
    assert not findings[0].is_proof
    assert findings[0].evidence.startswith(
        "candidate disagrees with the reference on 1"
    )
    assert "not proof" in findings[0].evidence


def test_an_excellent_prediction_just_short_of_a_copy_is_not_flagged() -> None:
    """The boundary case: 97% correct is a great agent, not an evasive one."""
    partition = partition_reference_columns(_adata())
    aligned = partition.align(partition.barcodes)
    truth = list(aligned.columns["bulk_labels"])
    prediction = [
        value if index % 32 else REFERENCE[(REFERENCE.index(value) + 1) % 3]
        for index, value in enumerate(truth)
    ]

    findings = detect_reference_leakage({"predicted_labels": prediction}, aligned)

    assert findings == []
