"""Tests for observing state, diffing it, and checking agent claims against it.

Once the agent runs its own code, the harness cannot learn what a step did by
reading the step. It can only compare what it saw before with what it sees
after. Four properties here carry more weight than the rest, because each is a
place the implementation could look finished while being wrong:

1. **Absent is never unchanged.** An empty delta is the honest answer for a step
   that changed nothing *and* for a namespace nobody could read. Every test that
   drives an unreadable namespace asserts on ``unobserved`` rather than on
   emptiness, because conflating the two either manufactures a discrepancy or
   hides one.
2. **Verification can only cost, never pay.** A truthful claim, an empty claim,
   and a false claim must all leave the measurement identical. Silence is not a
   discrepancy, and consistency is not credit.
3. **Filtering cells must not hand the agent the answer key.** A QC step changes
   every surviving column's values, so ``changed`` names the whole frame. If
   provenance read ``changed`` there, the reference labels would be attributed to
   the agent and scored as its own work.
4. **Readable-but-absent is a failure, not a gap.** If an artifact loads and the
   named column is missing, the rule fails. Reporting it as uncheckable would
   reward an agent for producing less.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("anndata")
pytest.importorskip("pandas")

import anndata
import numpy as np
import pandas as pd

from agent_evals.agents.decisions.verification import (
    # These two tables map the claim keys and namespace labels the verifier
    # understands. A name in either that is not in STATE_NAMESPACES fails
    # silently, so the agreement is asserted below -- which means reaching past
    # the module's public surface to read them.
    _ADDED_KEYS,
    _NAMESPACE_LABELS,
    DecisionVerification,
    DiscrepancyFlag,
    verify_state_claim,
)
from agent_evals.core.artifact_rules import (
    UnparseableValidationRule,
    parse_validation_rule,
)
from agent_evals.core.reference_columns import REFERENCE_LABEL_COLUMNS
from agent_evals.environment.execution.dataset import (
    DATASET_NAMESPACES,
    dataset_delta,
    diff_datasets,
    fingerprint_dataset,
    written_obs_columns,
)
from agent_evals.environment.execution.fingerprint import (
    DigestMethod,
    diff_workspaces,
    fingerprint_workspace,
)
from agent_evals.environment.execution.observer import (
    DatasetObserver,
    H5adDatasetObserver,
    InMemoryDatasetObserver,
)
from agent_evals.environment.models import (
    STATE_NAMESPACES,
    ArtifactRecord,
    KeyDelta,
    RuleOutcome,
    StateDelta,
)
from agent_evals.scientific.artifacts.validation import (
    ArtifactRuleValidator,
    artifact_path,
    verify_checksum,
)

_CELLS = 24
_GENES = 6


def _adata(*, cells: int = _CELLS) -> anndata.AnnData:
    """Build a small labelled dataset with a reference column in ``obs``."""
    rng = np.random.default_rng(11)
    obs = pd.DataFrame(
        {
            # The answer key, present in obs exactly as a raw dataset ships it.
            "cell_type": pd.Categorical([f"type-{i % 3}" for i in range(cells)]),
            "total_counts": np.arange(cells, dtype=np.float32) + 100.0,
        },
        index=[f"cell-{i}" for i in range(cells)],
    )
    return anndata.AnnData(
        X=rng.random((cells, _GENES), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"gene-{i}" for i in range(_GENES)]),
    )


# --------------------------------------------------------------------------- #
# Shared vocabularies. These agreements fail silently, so they are asserted.
# --------------------------------------------------------------------------- #


def test_every_verifier_namespace_is_a_real_state_namespace() -> None:
    """A namespace the verifier names but StateDelta does not is unreachable.

    ``getattr(observed, namespace)`` would raise, and ``is_observed`` would
    always answer True for a misspelling -- reading "we could not look" as
    "nothing happened".
    """
    assert set(_ADDED_KEYS.values()) <= STATE_NAMESPACES
    assert set(_NAMESPACE_LABELS) <= STATE_NAMESPACES


def test_dataset_namespaces_are_a_subset_of_state_namespaces() -> None:
    """``dataset_delta`` writes these into ``unobserved``, which is that vocabulary."""
    assert set(DATASET_NAMESPACES) <= STATE_NAMESPACES


def test_every_added_key_namespace_has_a_human_label() -> None:
    """A missing label degrades a finding to the bare namespace name."""
    assert set(_ADDED_KEYS.values()) <= set(_NAMESPACE_LABELS)


# --------------------------------------------------------------------------- #
# Workspace observation.
# --------------------------------------------------------------------------- #


def test_workspace_diff_separates_created_removed_and_rewritten(tmp_path: Path) -> None:
    kept = tmp_path / "kept.txt"
    doomed = tmp_path / "doomed.txt"
    edited = tmp_path / "edited.txt"
    kept.write_text("same", encoding="utf-8")
    doomed.write_text("bye", encoding="utf-8")
    edited.write_text("before", encoding="utf-8")

    before = fingerprint_workspace(tmp_path)
    doomed.unlink()
    edited.write_text("after", encoding="utf-8")
    (tmp_path / "fresh.txt").write_text("new", encoding="utf-8")
    delta = diff_workspaces(before, fingerprint_workspace(tmp_path))

    assert delta.added == ["fresh.txt"]
    assert delta.removed == ["doomed.txt"]
    assert delta.changed == ["edited.txt"]
    assert "kept.txt" not in delta.touched


def test_a_content_hashed_file_is_proof_and_a_size_mtime_file_is_not(
    tmp_path: Path,
) -> None:
    """An unchanged verdict from the cheap digest is evidence, not proof."""
    small = tmp_path / "small.txt"
    small.write_text("x" * 32, encoding="utf-8")
    large = tmp_path / "large.bin"
    large.write_bytes(b"y" * 4096)

    # A threshold below the large file forces it onto the size-and-mtime path.
    printed = fingerprint_workspace(tmp_path, max_content_bytes=1024)

    assert printed.files["small.txt"].method is DigestMethod.SHA256
    assert printed.files["small.txt"].is_proof
    assert printed.files["large.bin"].method is DigestMethod.SIZE_MTIME
    assert not printed.files["large.bin"].is_proof
    # The proxy-backed path is named as unproven even when it looks unchanged.
    assert diff_workspaces(printed, printed).unproven == ["large.bin"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_symlink_is_recorded_unreadable_rather_than_followed(tmp_path: Path) -> None:
    """Following one would pull a file from outside the workspace into its state."""
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "link.txt").symlink_to(outside)

    printed = fingerprint_workspace(root)

    assert printed.files == {}
    assert printed.unreadable == {"link.txt": "symlink not followed"}


# --------------------------------------------------------------------------- #
# Dataset observation.
# --------------------------------------------------------------------------- #


def test_a_written_obs_column_appears_in_the_observed_delta() -> None:
    before = fingerprint_dataset(_adata())
    after_adata = _adata()
    after_adata.obs["predicted_label"] = pd.Categorical(
        [f"guess-{i % 2}" for i in range(_CELLS)]
    )
    delta = diff_datasets(before, fingerprint_dataset(after_adata))

    assert delta.obs.added == ["predicted_label"]
    assert delta.obs_names_changed is False
    assert written_obs_columns(delta) == ["predicted_label"]


def test_a_written_embedding_appears_in_the_obsm_delta() -> None:
    before = fingerprint_dataset(_adata())
    after_adata = _adata()
    after_adata.obsm["X_pca"] = np.zeros((_CELLS, 3), dtype=np.float32)
    delta = diff_datasets(before, fingerprint_dataset(after_adata))

    assert delta.obsm.added == ["X_pca"]
    assert delta.obs.is_empty


def test_filtering_cells_does_not_attribute_reference_columns_to_the_agent() -> None:
    """The provenance rule that keeps a QC step from earning the answer key.

    Dropping cells changes every surviving column's values, so ``obs.changed``
    names the whole frame -- including ``cell_type``. Attributing ``changed``
    while the barcodes moved would hand the agent credit for the reference labels
    it is being scored against.
    """
    before = fingerprint_dataset(_adata())
    filtered = _adata()[: _CELLS // 2].copy()
    delta = diff_datasets(before, fingerprint_dataset(filtered))

    assert delta.n_obs_before == _CELLS
    assert delta.n_obs_after == _CELLS // 2
    assert delta.cells_removed == _CELLS // 2
    assert delta.obs_names_changed is True
    # The reference column is reported as changed -- that part is true ...
    assert "cell_type" in delta.obs.changed
    # ... but provenance must not read it as the agent's work.
    assert written_obs_columns(delta) == []
    for column in REFERENCE_LABEL_COLUMNS:
        assert column not in written_obs_columns(delta)
    assert any("obs.added" in note for note in delta.limitations)


def test_a_filter_that_also_writes_a_column_attributes_only_that_column() -> None:
    """The narrow reading still credits genuinely new work."""
    before = fingerprint_dataset(_adata())
    filtered = _adata()[: _CELLS // 2].copy()
    filtered.obs["qc_pass"] = np.ones(_CELLS // 2, dtype=bool)
    delta = diff_datasets(before, fingerprint_dataset(filtered))

    assert written_obs_columns(delta) == ["qc_pass"]


def test_an_unreadable_dataset_is_unobserved_rather_than_unchanged() -> None:
    delta = dataset_delta(None, fingerprint_dataset(_adata()))

    for namespace in DATASET_NAMESPACES:
        assert not delta.is_observed(namespace)
    assert not delta.is_observed("files")
    assert delta.limitations
    # Emptiness must not be readable as "nothing happened" here.
    assert written_obs_columns(delta) == []


def test_an_observed_dataset_with_observed_files_leaves_nothing_unobserved() -> None:
    printed = fingerprint_dataset(_adata())
    delta = dataset_delta(printed, printed, files=KeyDelta(added=["out.csv"]))

    assert delta.unobserved == []
    assert delta.files.added == ["out.csv"]
    assert delta.obs.is_empty


# --------------------------------------------------------------------------- #
# Observing the dataset from wherever it lives. Under free execution the harness
# holds a file, not an object, so both observers must feed the same diff.
# --------------------------------------------------------------------------- #


def test_both_observers_agree_about_the_same_dataset(tmp_path: Path) -> None:
    """The two tiers must not be able to establish provenance differently."""
    adata = _adata()
    target = tmp_path / "dataset.h5ad"
    adata.write_h5ad(target)

    from_disk = H5adDatasetObserver(target).snapshot()
    from_memory = InMemoryDatasetObserver(adata).snapshot()

    assert from_disk is not None
    assert from_memory is not None
    assert from_disk.n_obs == from_memory.n_obs == _CELLS
    assert from_disk.obs_names_digest == from_memory.obs_names_digest
    assert set(from_disk.obs) == set(from_memory.obs)
    # A round trip through disk must not change what a column's values digest to,
    # or the two tiers would report spurious changes for identical data.
    assert from_disk.obs["cell_type"].digest == from_memory.obs["cell_type"].digest


def test_an_h5ad_observer_diffs_a_column_the_agent_wrote(tmp_path: Path) -> None:
    """The free-execution shape: observe, let code run, observe again."""
    target = tmp_path / "dataset.h5ad"
    _adata().write_h5ad(target)
    observer = H5adDatasetObserver(target)
    before = observer.snapshot()

    # Stands in for the agent's own code rewriting the file in place.
    written = _adata()
    written.obs["predicted_label"] = pd.Categorical(["a"] * _CELLS)
    written.write_h5ad(target)
    delta = dataset_delta(before, observer.snapshot())

    assert delta.obs.added == ["predicted_label"]
    assert written_obs_columns(delta) == ["predicted_label"]


def test_an_absent_dataset_observes_as_none_rather_than_empty(tmp_path: Path) -> None:
    """Before the agent's first step the file it will write does not exist."""
    assert H5adDatasetObserver(tmp_path / "not-yet.h5ad").snapshot() is None


def test_an_unreadable_file_observes_as_none_rather_than_raising(tmp_path: Path) -> None:
    """A crash here would be recorded as the agent's execution failing."""
    target = tmp_path / "corrupt.h5ad"
    target.write_bytes(b"this is not an HDF5 file")

    assert H5adDatasetObserver(target).snapshot() is None


def test_a_dataset_above_the_read_budget_leaves_the_matrix_unobserved(
    tmp_path: Path,
) -> None:
    """Backed reads must report the matrix as unobserved, not as unchanged."""
    target = tmp_path / "dataset.h5ad"
    _adata().write_h5ad(target)

    printed = H5adDatasetObserver(target, max_read_bytes=1).snapshot()

    assert printed is not None
    assert printed.matrix is None
    assert any("opened backed" in note for note in printed.limitations)
    # Diffing two such fingerprints must name the matrix unobserved.
    assert not dataset_delta(printed, printed).is_observed("matrix")


def test_an_object_that_is_not_a_dataset_observes_as_none() -> None:
    assert InMemoryDatasetObserver(object()).snapshot() is None


def test_both_observers_satisfy_the_port(tmp_path: Path) -> None:
    assert isinstance(H5adDatasetObserver(tmp_path / "x.h5ad"), DatasetObserver)
    assert isinstance(InMemoryDatasetObserver(_adata()), DatasetObserver)


# --------------------------------------------------------------------------- #
# Claimed versus observed. The plan's acceptance check lives here.
# --------------------------------------------------------------------------- #


def _filter_delta() -> StateDelta:
    """An observed delta for a step that dropped half the cells."""
    before = fingerprint_dataset(_adata())
    return diff_datasets(before, fingerprint_dataset(_adata()[: _CELLS // 2].copy()))


def test_a_contradicted_state_claim_is_flagged() -> None:
    """The plan's acceptance check: a claim that contradicts observation is caught."""
    observed = _filter_delta()
    verification = verify_state_claim({"n_obs": _CELLS}, observed)

    assert DiscrepancyFlag.CONTRADICTED_CLAIM in verification.discrepancy_flags
    assert not verification.is_consistent
    assert verification.checked_claims == 1
    assert any(str(_CELLS // 2) in finding for finding in verification.findings)


def test_a_false_claim_does_not_alter_the_observation() -> None:
    """Verification can only cost, never pay.

    The measurement is the evidence a claim is checked against, so it must not be
    derived from the claim. A truthful, an absent, and a false claim all leave it
    byte-identical -- there is no channel by which saying the right thing improves
    what was measured.
    """
    observed = _filter_delta()
    truthful = verify_state_claim({"n_obs": _CELLS // 2}, observed)
    silent = verify_state_claim({}, observed)
    false = verify_state_claim({"n_obs": 1}, observed)

    assert truthful.observed_state_delta == observed
    assert silent.observed_state_delta == observed
    assert false.observed_state_delta == observed
    # Nor is there a score on the verdict that a claim could move.
    assert "score" not in DecisionVerification.model_fields


def test_silence_is_not_a_discrepancy() -> None:
    """An agent that claimed nothing is trivially consistent, and unmeasured."""
    verification = verify_state_claim({}, _filter_delta())

    assert verification.discrepancy_flags == []
    assert verification.is_consistent
    assert verification.checked_claims == 0


def test_a_truthful_claim_is_consistent_and_counted() -> None:
    observed = _filter_delta()
    verification = verify_state_claim(
        {"n_obs": _CELLS // 2, "n_vars": _GENES}, observed
    )

    assert verification.discrepancy_flags == []
    assert verification.checked_claims == 2


def test_a_claim_about_an_unobserved_namespace_is_unverifiable_not_contradicted() -> None:
    """Not knowing is neither exoneration nor accusation."""
    verification = verify_state_claim(
        {"obs_columns_added": ["predicted_label"]},
        dataset_delta(None, None),
    )

    assert verification.discrepancy_flags == [DiscrepancyFlag.UNVERIFIABLE]
    assert DiscrepancyFlag.UNSUPPORTED_CLAIM not in verification.discrepancy_flags


def test_no_observation_at_all_leaves_claims_unverifiable() -> None:
    verification = verify_state_claim({"n_obs": 10}, None)

    assert verification.discrepancy_flags == [DiscrepancyFlag.UNVERIFIABLE]
    assert verification.observed_state_delta is None
    # The claim survives in the record so a reader can audit the verdict.
    assert verification.claimed_state_delta == {"n_obs": 10}


def test_an_unreadable_claim_shape_is_malformed_not_ignored() -> None:
    """Silently dropping it would record the agent as having claimed nothing."""
    observed = _filter_delta()

    assert verify_state_claim({"n_obs": "many"}, observed).discrepancy_flags == [
        DiscrepancyFlag.MALFORMED_CLAIM
    ]
    assert verify_state_claim({"obs_columns_added": 7}, observed).discrepancy_flags == [
        DiscrepancyFlag.MALFORMED_CLAIM
    ]


def test_a_claim_naming_a_column_that_never_appeared_is_unsupported() -> None:
    before = fingerprint_dataset(_adata())
    after_adata = _adata()
    after_adata.obs["real"] = np.zeros(_CELLS, dtype=np.int8)
    observed = diff_datasets(before, fingerprint_dataset(after_adata))

    verification = verify_state_claim({"obs_columns_added": ["real", "invented"]}, observed)

    assert verification.discrepancy_flags == [DiscrepancyFlag.UNSUPPORTED_CLAIM]
    assert any("invented" in finding for finding in verification.findings)


def test_an_incomplete_claim_is_flagged_only_because_a_claim_was_made() -> None:
    before = fingerprint_dataset(_adata())
    after_adata = _adata()
    after_adata.obs["disclosed"] = np.zeros(_CELLS, dtype=np.int8)
    after_adata.obs["quiet"] = np.ones(_CELLS, dtype=np.int8)
    observed = diff_datasets(before, fingerprint_dataset(after_adata))

    disclosed = verify_state_claim({"obs_columns_added": ["disclosed"]}, observed)
    assert disclosed.discrepancy_flags == [DiscrepancyFlag.UNDISCLOSED_CHANGE]
    assert any("quiet" in finding for finding in disclosed.findings)
    # Claiming nothing of that kind raises nothing: silence is not concealment.
    assert verify_state_claim({}, observed).discrepancy_flags == []


def test_an_unrecognised_claim_key_is_preserved_and_left_unchecked() -> None:
    """It may be a convention the harness has not learned, not a violation."""
    verification = verify_state_claim({"invented_key": "whatever"}, _filter_delta())

    assert verification.discrepancy_flags == []
    assert verification.claimed_state_delta == {"invented_key": "whatever"}
    assert verification.checked_claims == 0


def test_a_state_claim_that_is_not_a_mapping_is_malformed() -> None:
    verification = verify_state_claim(["not", "a", "mapping"], _filter_delta())  # type: ignore[arg-type]

    assert DiscrepancyFlag.MALFORMED_CLAIM in verification.discrepancy_flags
    assert verification.claimed_state_delta == {}


# --------------------------------------------------------------------------- #
# Artifact validation.
# --------------------------------------------------------------------------- #


def _record(path: Path, *, checksum: str | None = None, uri: str | None = None) -> ArtifactRecord:
    """Build an artifact record pointing at ``path``."""
    return ArtifactRecord(
        artifact_id="probe",
        kind="table",
        format="csv",
        uri=uri if uri is not None else str(path),
        checksum=checksum,
    )


def _rule(rule: str, *, name: str = "probe-rule") -> object:
    """Build the declared validation rule a benchmark would carry."""
    from agent_evals.benchmarks.schema import ValidationRule

    return ValidationRule(name=name, description=name, rule=rule)


def test_artifact_path_reads_both_execution_tiers_spellings(tmp_path: Path) -> None:
    """The typed tier records ``str(path)``; the free tier records ``as_uri()``.

    A plain Windows path begins with a drive letter, which ``urlparse`` reads as a
    scheme. Resolving that to ``None`` made every typed-tier artifact look
    unreadable -- and because unreadable is deliberately non-blocking, the whole
    layer sat inert without failing anything.
    """
    target = tmp_path / "artifact.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")

    assert artifact_path(str(target)) == target
    assert artifact_path(target.as_uri()) == target
    assert artifact_path(None) is None
    assert artifact_path("   ") is None
    # A genuine remote locator is not a local artifact.
    assert artifact_path("s3://bucket/key.csv") is None


def test_a_recorded_checksum_is_recomputed_rather_than_trusted(tmp_path: Path) -> None:
    target = tmp_path / "artifact.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    assert verify_checksum(target, digest) is True
    # Both spellings the two tiers use must compare equal.
    assert verify_checksum(target, f"sha256:{digest}") is True
    target.write_text("a,b\n9,9\n", encoding="utf-8")
    assert verify_checksum(target, digest) is False
    # Nothing to compare against is not a failure.
    assert verify_checksum(target, None) is None


async def test_a_readable_artifact_missing_the_named_column_fails(tmp_path: Path) -> None:
    """The anti-gaming case: producing less must not become a harness gap.

    Reporting this as uncheckable would let an agent omit a required column
    entirely and have the check that should catch it stop blocking validity.
    """
    target = tmp_path / "artifact.csv"
    target.write_text("cell_id,cluster\nc0,1\n", encoding="utf-8")
    validator = ArtifactRuleValidator()

    validation = await validator.validate(
        _record(target),
        [_rule("columns include cluster,pct_counts_mt")],  # type: ignore[list-item]
        {},
    )

    assert validation.exists
    assert [rule.outcome for rule in validation.rules] == [RuleOutcome.FAILED]
    assert not validation.is_valid
    assert "pct_counts_mt" in validation.rules[0].detail


async def test_a_satisfied_rule_passes_and_marks_the_artifact_valid(tmp_path: Path) -> None:
    target = tmp_path / "artifact.csv"
    target.write_text("total_counts,pct_counts_mt\n10,0.1\n", encoding="utf-8")
    validator = ArtifactRuleValidator()

    validation = await validator.validate(
        _record(target),
        [_rule("columns include total_counts,pct_counts_mt")],  # type: ignore[list-item]
        {},
    )

    assert [rule.outcome for rule in validation.rules] == [RuleOutcome.PASSED]
    assert validation.is_valid


async def test_a_missing_artifact_is_uncheckable_and_does_not_claim_invalidity(
    tmp_path: Path,
) -> None:
    """Not measurable means no score, not a zero."""
    validator = ArtifactRuleValidator()

    validation = await validator.validate(
        _record(tmp_path / "never-written.csv"),
        [_rule("columns include cluster")],  # type: ignore[list-item]
        {},
    )

    assert not validation.exists
    assert [rule.outcome for rule in validation.rules] == [RuleOutcome.UNCHECKABLE]
    assert validation.limitations
    # An uncheckable rule is not a failed rule, but a missing file is still not valid.
    assert validation.with_outcome(RuleOutcome.FAILED) == []
    assert not validation.is_valid


async def test_a_tampered_artifact_fails_its_checksum_and_is_not_valid(
    tmp_path: Path,
) -> None:
    """Computed once and trusted forever is exactly what this must not do."""
    import hashlib

    target = tmp_path / "artifact.csv"
    target.write_text("total_counts\n10\n", encoding="utf-8")
    recorded = hashlib.sha256(target.read_bytes()).hexdigest()
    target.write_text("total_counts\n999999\n", encoding="utf-8")
    validator = ArtifactRuleValidator()

    validation = await validator.validate(
        _record(target, checksum=recorded),
        [_rule("columns include total_counts")],  # type: ignore[list-item]
        {},
    )

    # The rule still passes -- the column is there -- but the file is not the
    # file that was recorded, so the artifact is not valid.
    assert [rule.outcome for rule in validation.rules] == [RuleOutcome.PASSED]
    assert validation.checksum_verified is False
    assert not validation.is_valid


async def test_a_vocabulary_rule_reads_the_declared_parameter(tmp_path: Path) -> None:
    target = tmp_path / "artifact.csv"
    target.write_text("predicted_label\nB cell\nT cell\n", encoding="utf-8")
    validator = ArtifactRuleValidator()
    rules = [_rule("predicted_label is non-null and belongs to label_vocabulary")]

    allowed = await validator.validate(
        _record(target), rules, {"label_vocabulary": ["B cell", "T cell"]}  # type: ignore[arg-type]
    )
    assert [rule.outcome for rule in allowed.rules] == [RuleOutcome.PASSED]

    disallowed = await validator.validate(
        _record(target), rules, {"label_vocabulary": ["B cell"]}  # type: ignore[arg-type]
    )
    assert [rule.outcome for rule in disallowed.rules] == [RuleOutcome.FAILED]
    assert "T cell" in disallowed.rules[0].detail


async def test_an_undeclared_vocabulary_is_uncheckable_not_failed(tmp_path: Path) -> None:
    """The benchmark did not say what is allowed, so nothing can be concluded."""
    target = tmp_path / "artifact.csv"
    target.write_text("predicted_label\nB cell\n", encoding="utf-8")
    validator = ArtifactRuleValidator()

    validation = await validator.validate(
        _record(target),
        [_rule("predicted_label is non-null and belongs to label_vocabulary")],  # type: ignore[list-item]
        {},
    )

    assert [rule.outcome for rule in validation.rules] == [RuleOutcome.UNCHECKABLE]
    assert validation.with_outcome(RuleOutcome.FAILED) == []


async def test_an_artifact_with_no_rules_is_valid_when_it_exists(tmp_path: Path) -> None:
    """A benchmark that declared no rules has not found the artifact wanting."""
    target = tmp_path / "artifact.csv"
    target.write_text("anything\n1\n", encoding="utf-8")

    validation = await ArtifactRuleValidator().validate(_record(target), [], {})

    assert validation.exists
    assert validation.rules == []
    assert validation.is_valid


async def test_an_artifact_with_no_uri_is_not_silently_valid() -> None:
    validation = await ArtifactRuleValidator().validate(
        ArtifactRecord(artifact_id="probe", kind="table", format="csv"), [], {}
    )

    assert not validation.exists
    assert not validation.is_valid
    assert validation.limitations


# --------------------------------------------------------------------------- #
# Rules are checked when the benchmark loads, not when the agent is scored.
# --------------------------------------------------------------------------- #


def test_every_rule_in_the_example_benchmarks_can_be_parsed() -> None:
    """An unparseable rule is the author's typo, and it must not reach a run."""
    from agent_evals.benchmarks.io import load_benchmark

    for path in sorted(Path("examples/benchmarks").glob("*.yaml")):
        specification = load_benchmark(path)
        for artifact in specification.artifacts:
            for rule in artifact.validation:
                parsed = parse_validation_rule(rule.rule)
                assert parsed is not None, f"{path.name}:{artifact.id}:{rule.name}"


def test_an_unreadable_rule_is_rejected_rather_than_evaluated() -> None:
    with pytest.raises(UnparseableValidationRule):
        parse_validation_rule("please ensure the output is basically fine")
