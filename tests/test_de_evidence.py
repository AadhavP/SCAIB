"""The differential-expression benchmark's evidence, end of the chain to start.

Nine ``differential_expression.*`` metrics existed with nothing producing the
marker set they score against, so the whole family was structurally ineligible and
the DE benchmark honestly reported its scientific outcome as unmeasured. Closing
that needed four things, and each one is a place the obvious implementation is
wrong in a way that fails silently:

* the reference ranking has to be **computed** evaluator-side, and its effect sizes
  came back entirely non-finite on the real dataset for a reason a green suite
  cannot see (:func:`test_the_reference_effect_sizes_are_finite_on_the_real_dataset`);
* the contrast has to be published to the agent **without naming a reference
  class**, and the channel that publishes it is verbatim
  (:func:`test_no_declared_observation_publishes_reference_vocabulary`);
* ``normalize`` has to *honour* the method the benchmark makes required, rather
  than accept the parameter and normalize the way it always did
  (:func:`test_the_two_normalization_methods_produce_different_data`);
* ``report`` has to read the agent's own table and nothing the evaluator holds.

The leakage assertions are first because they are the constraint the rest is built
under: the reference marker ranking is an answer key, and every one of these
channels is somewhere it could reach the agent.
"""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("anndata")
pytest.importorskip("scanpy")

import anndata
import numpy as np
import pandas as pd

from agent_evals.benchmarks.io import load_benchmark
from agent_evals.core.de_evidence import (
    DE_REFERENCE_IMPLEMENTATION,
    DE_REFERENCE_TARGET,
    DE_SCORED_GROUP,
    DE_TABLE_ARTIFACT,
    REFERENCE_EFFECT_SIZES,
    REFERENCE_MARKERS,
)
from agent_evals.core.reference_columns import RESERVED_REFERENCE_COLUMNS
from agent_evals.evaluation.candidates import (
    CANDIDATE_LABEL_FIELD,
    REFERENCE_LABEL_FIELD,
)
from agent_evals.evaluation.reference_de import (
    ReferenceContrast,
    compute_reference_markers,
    contrast_for,
    reference_de_metadata,
    resolve_candidate_group,
    scored_group_metadata,
)
from agent_evals.scientific.artifacts.storage import LocalArtifactStore
from agent_evals.scientific.context import ScientificContext
from agent_evals.scientific.observations.observations import (
    ScientificObservationBuilder,
)
from agent_evals.scientific.operations.de import DE_ARTIFACT_ID
from agent_evals.scientific.operations.normalize import (
    LIBRARY_SIZE_LOG1P,
    MEDIAN_COUNTS_LOG1P,
    normalize,
)
from agent_evals.scientific.operations.report import (
    DEFAULT_TOP_N,
    UNGROUPED_SECTION,
    report,
)

EXAMPLES = Path(__file__).parents[1] / "examples" / "benchmarks"
PBMC_CACHE = Path(".cache/datasets/pbmc68k_reduced.h5ad")

DE_BENCHMARK = "pbmc-differential-expression"

#: The reference population the DE benchmark's outcome is measured on. Duplicated
#: from ``_CONTRASTS`` deliberately: reading it from the module under test would
#: make the leakage assertions agree with any rename, including one that renamed it
#: to something the YAML does publish.
SCORED_POPULATION = "CD14+ Monocyte"


def _counts(cells: int = 60, genes: int = 40, seed: int = 0) -> np.ndarray:
    """Non-negative integer-ish counts with a median library size well off 10000."""
    generator = np.random.default_rng(seed)
    return generator.integers(0, 30, size=(cells, genes)).astype(np.float64)


def _adata(cells: int = 60, genes: int = 40, seed: int = 0) -> anndata.AnnData:
    matrix = _counts(cells, genes, seed)
    return anndata.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(cells)]),
        var=pd.DataFrame(index=[f"GENE{index}" for index in range(genes)]),
    )


def _context(adata: Any, root: Path) -> ScientificContext:
    return ScientificContext(
        adata=adata,
        dataset_metadata={"id": "synthetic"},
        artifact_store=LocalArtifactStore(root / "artifacts"),
        workspace=root,
    )


def _library_sizes(adata: Any) -> np.ndarray:
    """Undo ``log1p`` and total per cell, which is what the method choice moves."""
    return np.expm1(np.asarray(adata.X)).sum(axis=1)


# --------------------------------------------------------------------------- #
# The leakage boundary
# --------------------------------------------------------------------------- #


def _flatten(value: Any) -> list[str]:
    """Every string a declared value would publish, at any nesting depth."""
    if isinstance(value, dict):
        return [
            text
            for key, item in value.items()
            for text in (str(key), *_flatten(item))
        ]
    if isinstance(value, list | tuple | set):
        return [text for item in value for text in _flatten(item)]
    return [str(value)]


@pytest.mark.parametrize(
    "benchmark", sorted(path.name for path in EXAMPLES.glob("*.yaml"))
)
def test_no_declared_observation_publishes_reference_vocabulary(benchmark: str) -> None:
    """``schema:`` is served to the agent verbatim, so its contents are a disclosure.

    ``DeclaredObservationBuilder`` was added in Stage 7 to give ``schema:`` its
    first reader, and giving an inert YAML field a reader turned the DE benchmark's
    declaration into a live leak: it named the two reference populations to
    contrast. The field is now the contrast's *shape* only, and this is what
    notices if a population name goes back into it -- on any benchmark, since the
    channel is generic and the next one to use it will not remember why.
    """
    specification = load_benchmark(EXAMPLES / benchmark)

    published = {
        observation.id: _flatten(observation.schema_definition)
        for observation in specification.observations
        if observation.schema_definition
    }

    withheld = {*RESERVED_REFERENCE_COLUMNS, SCORED_POPULATION}
    offending = {
        observation_id: sorted(set(strings) & withheld)
        for observation_id, strings in published.items()
        if set(strings) & withheld
    }
    assert offending == {}


def test_the_scored_population_appears_nowhere_in_the_benchmark_definition() -> None:
    """Stronger than the schema check, and it has to be: prose publishes too.

    The task objective, an observation ``description``, and an action ``purpose``
    are all rendered into the generated system prompt. So the guarantee worth
    asserting is not "the structured field is clean" but "the string does not occur
    in the file", which is the only form that survives someone helpfully explaining
    the contrast in a description.
    """
    text = (EXAMPLES / f"{DE_BENCHMARK}.yaml").read_text(encoding="utf-8")

    # The YAML comment recording *why* the old contrast was dropped enumerates the
    # measured populations, which is the same disclosure if a comment were served.
    # Comments are not, so they are excluded here -- and that exclusion is exactly
    # what makes this assertion about the served text rather than about the file.
    served = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )

    assert SCORED_POPULATION not in served
    for column in sorted(RESERVED_REFERENCE_COLUMNS):
        assert column not in served, column


def test_the_reference_metadata_keys_are_evaluator_side_only() -> None:
    """The evidence the reference produces must not be a benchmark declaration.

    ``reference_markers`` is the answer key and ``de_reference_target`` is a
    withheld class name. Both travel as metric-context metadata, which no
    observation builder reads -- but a benchmark could declare either as an
    observation ``schema:`` value and publish it, and nothing else would object.
    """
    for path in sorted(EXAMPLES.glob("*.yaml")):
        specification = load_benchmark(path)
        for observation in specification.observations:
            keys = set(_flatten(observation.schema_definition))
            assert REFERENCE_MARKERS not in keys, path.name
            assert REFERENCE_EFFECT_SIZES not in keys, path.name
            assert DE_REFERENCE_TARGET not in keys, path.name


# --------------------------------------------------------------------------- #
# normalize: the method the benchmark makes required
# --------------------------------------------------------------------------- #


def test_the_two_normalization_methods_produce_different_data(tmp_path: Path) -> None:
    """The parameter is honoured, not merely recorded.

    The DE benchmark declares ``method`` *required*, so an agent naming one has
    made a decision the run is scored on. Before this, ``normalize`` read only
    ``target_sum`` -- so both choices delivered the fixed-target pipeline and the
    scored decision was one the run never took. Asserted on the data rather than on
    the metadata, because a metadata-only assertion passes against exactly that
    defect.
    """
    fixed = _context(_adata(), tmp_path / "fixed")
    median = _context(_adata(), tmp_path / "median")

    normalize(fixed, {"method": LIBRARY_SIZE_LOG1P, "target_sum": 10_000})
    normalize(median, {"method": MEDIAN_COUNTS_LOG1P})

    fixed_sizes = _library_sizes(fixed.adata)
    median_sizes = _library_sizes(median.adata)

    assert np.allclose(fixed_sizes, 10_000, rtol=1e-3)
    # The synthetic data's own median is nowhere near 10000, so "normalized to the
    # dataset's median" and "normalized to 10000" are separable outcomes here.
    assert median_sizes.max() < 1_000
    assert not np.allclose(fixed_sizes, median_sizes)


def test_an_unimplemented_method_is_refused_by_name(tmp_path: Path) -> None:
    """A choice no executor implements must fail loudly, naming what exists.

    This is the guard that makes removing ``scran`` from the DE catalog a real fix
    rather than a documentation change: an agent that names it is told so, instead
    of silently receiving a different normalization.
    """
    context = _context(_adata(), tmp_path)

    with pytest.raises(ValueError, match="does not implement method 'scran'"):
        normalize(context, {"method": "scran"})


def test_a_target_sum_under_median_normalization_is_refused(tmp_path: Path) -> None:
    """Refused rather than ignored, which is a scoring decision and not pedantry.

    Honouring it would silently deliver the fixed-target method under the median
    method's name. Ignoring it would record a parameter the agent chose and the run
    then never applied -- and the parameter is scored.
    """
    context = _context(_adata(), tmp_path)

    with pytest.raises(ValueError, match="accepts no target_sum"):
        normalize(context, {"method": MEDIAN_COUNTS_LOG1P, "target_sum": 10_000})


def test_a_caller_that_names_no_method_gets_the_behaviour_it_always_had(
    tmp_path: Path,
) -> None:
    """The compatibility half, which nine existing call sites depend on.

    Every caller written before ``method`` existed sends ``target_sum`` alone. If
    the default had become the median method, all of them would have silently
    changed pipeline.
    """
    named = _context(_adata(), tmp_path / "named")
    unnamed = _context(_adata(), tmp_path / "unnamed")

    named_artifact = normalize(named, {"method": LIBRARY_SIZE_LOG1P, "target_sum": 5_000})
    unnamed_artifact = normalize(unnamed, {"target_sum": 5_000})

    assert np.allclose(np.asarray(named.adata.X), np.asarray(unnamed.adata.X))
    assert unnamed_artifact.outputs["method"] == LIBRARY_SIZE_LOG1P
    assert named_artifact.artifacts[0].metadata["target_sum"] == 5_000


def test_the_recorded_method_is_the_one_that_ran(tmp_path: Path) -> None:
    """Provenance for the decision, on both the artifact and the outputs."""
    context = _context(_adata(), tmp_path)

    output = normalize(context, {"method": MEDIAN_COUNTS_LOG1P})

    assert output.outputs["method"] == MEDIAN_COUNTS_LOG1P
    # ``None`` is how scanpy spells "normalize to the median", so the record has to
    # be able to hold it -- a float-typed field would have forced a fabricated 0.
    assert output.outputs["target_sum"] is None
    assert output.artifacts[0].metadata["method"] == MEDIAN_COUNTS_LOG1P
    assert output.artifacts[0].metadata["target_sum"] is None


# --------------------------------------------------------------------------- #
# report: the agent's own table and nothing else
# --------------------------------------------------------------------------- #


def _de_table(**columns: Any) -> pd.DataFrame:
    base = {
        "gene": ["GENE1", "GENE2", "GENE3", "GENE4"],
        "effect_size": [3.0, 2.0, 1.0, float("nan")],
        "p_value": [1e-9, 1e-6, 1e-3, 0.4],
        "q_value": [1e-8, 1e-5, 1e-2, 0.5],
    }
    base.update(columns)
    return pd.DataFrame(base)


def _with_de_table(context: ScientificContext, table: pd.DataFrame) -> None:
    artifact = context.artifact_store.save_table(DE_ARTIFACT_ID, table)
    context.add_artifact(artifact)


def test_a_report_without_a_de_table_is_refused(tmp_path: Path) -> None:
    """The precondition, at the operation as well as in the observation."""
    context = _context(_adata(), tmp_path)

    with pytest.raises(ValueError, match=f"no '{DE_ARTIFACT_ID}' artifact exists"):
        report(context, {})


def test_a_top_n_below_one_is_refused(tmp_path: Path) -> None:
    context = _context(_adata(), tmp_path)
    _with_de_table(context, _de_table())

    with pytest.raises(ValueError, match="top_n of at least 1"):
        report(context, {"top_n": 0})


def test_a_table_naming_no_gene_column_is_refused(tmp_path: Path) -> None:
    """A table the report cannot read is a harness gap, not an empty document.

    Rendering an empty report instead would archive an artifact that validates and
    says nothing, which scores as a produced deliverable.
    """
    context = _context(_adata(), tmp_path)
    _with_de_table(context, _de_table().drop(columns=["gene"]))

    with pytest.raises(ValueError, match="names no gene column"):
        report(context, {})


def test_the_report_is_written_as_html_and_cites_the_bytes_it_read(
    tmp_path: Path,
) -> None:
    """The extension follows the format, and the checksum is recomputed.

    Stage 3's validator dispatches on the *file*, so a document written as ``.txt``
    while the benchmark declares ``html`` leaves its rules unevaluable against a
    file sitting right there. The checksum is re-derived from the source at read
    time rather than copied off the source record, because a report citing a digest
    it never verified claims provenance it does not have.
    """
    import hashlib

    context = _context(_adata(), tmp_path)
    _with_de_table(context, _de_table())
    source = Path(context.artifacts[DE_ARTIFACT_ID].path)

    output = report(context, {"top_n": 2})
    artifact = output.artifacts[0]

    assert Path(artifact.path).suffix == ".html"
    assert artifact.format == "html"
    assert artifact.metadata["source_checksum"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert artifact.metadata["source_artifact"] == DE_ARTIFACT_ID


def test_a_non_finite_effect_size_is_rendered_as_itself(tmp_path: Path) -> None:
    """A fold change the agent's test could not size is not a claim of no effect.

    Formatting NaN as ``0`` would turn "unmeasurable" into "measured, and zero",
    which is the same substitution the scoring layer refuses everywhere else.
    """
    context = _context(_adata(), tmp_path)
    _with_de_table(context, _de_table())

    report(context, {})
    document = Path(context.artifact_store.root / "differential_expression_report.html").read_text(
        encoding="utf-8"
    )

    assert "nan" in document.lower()
    assert ">0<" not in document


def test_the_report_preserves_the_agents_own_ranking_and_groups(
    tmp_path: Path,
) -> None:
    """Row order is the agent's output, and re-sorting would describe another run."""
    context = _context(_adata(), tmp_path)
    _with_de_table(
        context,
        _de_table(group=["beta", "beta", "alpha", "alpha"]),
    )

    output = report(context, {"top_n": 1})

    # Declaration order, not sorted: ``sort=False`` on the groupby is what keeps the
    # document describing the table the agent submitted.
    assert output.outputs["groups_reported"] == ["beta", "alpha"]
    assert output.outputs["genes_reported"] == 2


def test_an_ungrouped_table_is_reported_as_one_section(tmp_path: Path) -> None:
    context = _context(_adata(), tmp_path)
    _with_de_table(context, _de_table())

    output = report(context, {})

    assert output.outputs["groups_reported"] == [UNGROUPED_SECTION]
    assert output.outputs["top_n"] == DEFAULT_TOP_N


def test_the_report_action_is_withheld_until_ranked_results_exist() -> None:
    """The observation half of the same precondition.

    Advertising ``report`` earlier offers an action certain to fail, and the failure
    is recorded against the agent rather than against the harness that offered it.
    """
    supported = {"report", "differential-expression"}
    before = ScientificObservationBuilder._action_precondition(
        "report",
        supported=supported,
        allowed=True,
        pipeline_state={"differential_expression_complete": False},
        batch_information={},
    )
    after = ScientificObservationBuilder._action_precondition(
        "report",
        supported=supported,
        allowed=True,
        pipeline_state={"differential_expression_complete": True},
        batch_information={},
    )

    assert before["available"] is False
    assert "differential-expression" in before["reason"]
    assert after["available"] is True


# --------------------------------------------------------------------------- #
# The reference ranking
# --------------------------------------------------------------------------- #


def _labelled(cells: int = 90, genes: int = 40) -> tuple[anndata.AnnData, list[str]]:
    """A dataset shaped like the real one: ``.raw`` in counts, ``.X`` z-scored.

    That combination is what made the defect invisible. Wilcoxon is rank-based and
    per-gene scaling is monotone, so the *ranking* is identical either way and every
    rank metric scored normally -- while the fold changes, which are computed from
    group means, came back non-finite because ``log2`` of a negative mean is NaN.
    """
    generator = np.random.default_rng(7)
    counts = generator.integers(0, 20, size=(cells, genes)).astype(np.float64)
    labels = [SCORED_POPULATION] * 30 + ["other"] * (cells - 30)
    # A real signal in the scored population, so the ranking is not noise.
    counts[:30, :5] += 60.0
    adata = anndata.AnnData(
        X=np.log1p(counts),
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(cells)]),
        var=pd.DataFrame(index=[f"GENE{index}" for index in range(genes)]),
    )
    adata.raw = adata.copy()
    scaled = np.asarray(adata.X)
    adata.X = (scaled - scaled.mean(axis=0)) / (scaled.std(axis=0) + 1e-9)
    return adata, labels


def test_the_reference_effect_sizes_are_finite_when_a_raw_layer_exists() -> None:
    """The tripwire for the defect no test could see.

    ``use_raw=False`` was forced on the theory that ``.raw`` holds an unfiltered
    gene universe the agent's table could not contain. Measurement falsified the
    premise and the cost was invisible: the rank metrics all scored, so the suite
    stayed green while ``effect_size_correlation`` and ``direction_agreement`` were
    *permanently* unmeasurable. Asserted as "every returned gene has a finite
    effect size", because the failure produced an empty mapping rather than an error.
    """
    adata, labels = _labelled()
    contrast = ReferenceContrast(benchmark=DE_BENCHMARK, group=SCORED_POPULATION, top_k=20)

    markers = compute_reference_markers(adata, labels=labels, contrast=contrast)

    assert markers.reason is None
    assert markers.available
    assert len(markers.genes) == 20
    assert set(markers.effect_sizes) == set(markers.genes)
    assert all(np.isfinite(value) for value in markers.effect_sizes.values())
    assert markers.implementation["parameters"]["use_raw"] is True


def test_the_reference_effect_sizes_are_finite_on_the_real_dataset() -> None:
    """The same claim against the data the paper's runs use.

    The synthetic case above reproduces the mechanism; only the real dataset can
    show that this dataset's ``.raw`` actually carries what the fix assumes. It was
    measured to hold the same 765 genes as ``.X``, which is why reading it costs no
    coverage of the agent's gene universe.
    """
    if not PBMC_CACHE.exists():
        pytest.skip("real PBMC cache is not available")
    adata = anndata.read_h5ad(PBMC_CACHE)
    if adata.raw is None:
        pytest.skip("this cached dataset ships no raw layer")
    column = next(
        (name for name in ("bulk_labels", "cell_type") if name in adata.obs), None
    )
    if column is None:
        pytest.skip("this cached dataset ships no reference population labels")
    labels = [str(value) for value in adata.obs[column].tolist()]
    contrast = ReferenceContrast(benchmark=DE_BENCHMARK, group=SCORED_POPULATION)

    markers = compute_reference_markers(adata, labels=labels, contrast=contrast)

    assert markers.reason is None, markers.reason
    assert len(markers.genes) == contrast.top_k
    assert markers.effect_sizes, "every reference fold change came back non-finite"
    assert all(np.isfinite(value) for value in markers.effect_sizes.values())


def test_a_population_absent_from_the_dataset_is_diagnosed_not_merely_survived() -> None:
    """Gathering evidence is observation, and observation must never raise.

    The weak form of this -- assert *a* reason came back -- passes even with the
    population-size guard deleted, because the broad ``except`` catches whatever
    scanpy then raises and reports that instead. Mutation testing caught exactly
    that. So the assertion has to name the diagnosis: this reason is the text a
    reader sees in ``outcome_limitations`` to learn why a metric went unscored, and
    "the reference population holds 0 cells" and "a KeyError escaped somewhere"
    are not the same finding even though both are honest about being a gap.
    """
    adata, labels = _labelled()
    contrast = ReferenceContrast(benchmark=DE_BENCHMARK, group="Nonexistent Population")

    markers = compute_reference_markers(adata, labels=labels, contrast=contrast)

    assert not markers.available
    assert markers.reason is not None
    assert "Nonexistent Population" in markers.reason
    assert "holds 0 cell(s)" in markers.reason
    assert "unmeasured rather than zero" in markers.reason
    # The generic wording of the ``except`` fallback. Its absence is what says the
    # gap was diagnosed by a guard rather than discovered by crashing.
    assert "could not be run" not in markers.reason
    assert markers.metadata(contrast) == {}


def test_a_grouping_that_covers_the_wrong_number_of_cells_is_a_reason() -> None:
    adata, labels = _labelled()
    contrast = ReferenceContrast(benchmark=DE_BENCHMARK, group=SCORED_POPULATION)

    markers = compute_reference_markers(adata, labels=labels[:-1], contrast=contrast)

    assert not markers.available
    assert markers.reason is not None
    assert "cannot be joined" in markers.reason


def test_a_benchmark_declaring_no_contrast_reports_no_gap() -> None:
    """No evidence and no limitation: marker recovery is not what it measures.

    The distinction matters because a ``reason`` is rendered into
    ``outcome_limitations``, and reporting one for the annotation benchmark would
    describe a missing reference the benchmark never asked for.
    """
    assert contrast_for("pbmc-cell-annotation") is None
    assert contrast_for(DE_BENCHMARK) is not None

    metadata, reason = reference_de_metadata("pbmc-cell-annotation", None, {})

    assert metadata == {}
    assert reason is None


def test_a_de_benchmark_with_no_reference_labels_records_the_gap() -> None:
    metadata, reason = reference_de_metadata(DE_BENCHMARK, None, {})

    assert metadata == {}
    assert reason is not None
    assert "unmeasured rather than zero" in reason


def test_the_reference_grouping_is_never_written_to_the_scored_object() -> None:
    """The reference has to be joined to run the test, and the join is a copy.

    Writing the grouping onto the object the agent's pipeline holds would put the
    answer key back into the data plane Stage 0 removed it from, one layer below
    every guard that checks for it -- and below ``record_produced_columns``, which
    only refuses the *reserved* spellings.
    """
    adata, labels = _labelled()
    before = set(adata.obs.columns)
    contrast = ReferenceContrast(benchmark=DE_BENCHMARK, group=SCORED_POPULATION, top_k=5)

    compute_reference_markers(adata, labels=labels, contrast=contrast)

    assert set(adata.obs.columns) == before


def test_the_reference_metadata_carries_pinned_provenance() -> None:
    adata, labels = _labelled()
    contrast = ReferenceContrast(benchmark=DE_BENCHMARK, group=SCORED_POPULATION, top_k=5)

    markers = compute_reference_markers(adata, labels=labels, contrast=contrast)
    metadata = markers.metadata(contrast)

    assert metadata[REFERENCE_MARKERS] == list(markers.genes)
    assert metadata[DE_REFERENCE_TARGET] == SCORED_POPULATION
    implementation = metadata[DE_REFERENCE_IMPLEMENTATION]
    assert implementation["provider"] == "scanpy"
    assert implementation["metric"] == "rank_genes_groups"
    assert implementation["version"] != "unknown"
    assert implementation["parameters"]["method"] == "wilcoxon"


# --------------------------------------------------------------------------- #
# Resolving the agent's own group
# --------------------------------------------------------------------------- #


def test_jaccard_overlap_refuses_to_reward_one_giant_cluster() -> None:
    """Why Jaccard and not raw overlap, in the only fixture that separates them.

    Raw overlap is maximized by a cluster containing every cell, so it would pay an
    agent for under-clustering. The separating case needs the giant group to hold
    *more* target cells than the tight one while being far worse matched -- 6 of 10
    in a group of 96 (Jaccard 0.06) against 4 of 10 in a group of 4 (Jaccard 0.40).
    Raw overlap picks ``everything``; Jaccard picks ``tight``.

    Written down because the first version of this test did not discriminate: its
    giant group held no target cells at all, so it never entered the comparison and
    both rules picked the same group. Mutation testing caught it -- swapping the key
    for a raw count left the test passing.
    """
    reference = ["target"] * 10 + ["other"] * 90
    candidate = ["tight"] * 4 + ["everything"] * 96

    assert resolve_candidate_group(candidate, reference, "target") == "tight"


def test_a_sole_group_is_returned_rather_than_rejected() -> None:
    """One cluster is a bad clustering, not an unresolvable one.

    It is scored as the poor grouping it is; refusing to resolve would make the
    metrics ineligible and cost the agent nothing for having produced it.
    """
    reference = ["target"] * 10 + ["other"] * 90

    assert resolve_candidate_group(["everything"] * 100, reference, "target") == "everything"


def test_an_unresolvable_grouping_returns_none_rather_than_a_guess() -> None:
    reference = ["target"] * 5 + ["other"] * 5
    assert resolve_candidate_group([], reference, "target") is None
    assert resolve_candidate_group(["a"] * 9, reference, "target") is None
    assert resolve_candidate_group(["a"] * 10, reference, "absent") is None


def test_a_tie_resolves_the_same_way_whatever_order_the_cells_arrive_in() -> None:
    """What the ``name`` tie-break actually buys, which is not what it looks like.

    Calling the function twice in one process would agree no matter what the key
    said -- it is pure, so that assertion is vacuous. The reproducibility hazard is
    that ``max`` scans a dict in insertion order, and insertion order is the order
    the *cells* arrive in. Without a total ordering in the key, the same clustering
    presented with its rows permuted would resolve a different group and score a
    different ``O``.

    ``b`` rather than ``a`` because ``max`` takes the largest name on a full tie.
    Which name wins is arbitrary and not a claim this makes; it is pinned so that
    changing the rule has to be deliberate.
    """
    reference = ["target", "target", "other", "other"]
    candidate = ["a", "b", "a", "b"]
    order = [3, 0, 2, 1]

    as_given = resolve_candidate_group(candidate, reference, "target")
    permuted = resolve_candidate_group(
        [candidate[index] for index in order],
        [reference[index] for index in order],
        "target",
    )

    assert as_given == permuted == "b"


def test_a_group_absent_from_the_de_table_is_not_used_to_filter_it() -> None:
    """The subtle half of resolution, and the one that fails silently.

    A name resolved from the clustering would otherwise be used to filter a table
    grouped by annotation. The filter matches nothing, and an empty ranking scores
    exactly like a wrong one.
    """
    reference_frame = pd.DataFrame({REFERENCE_LABEL_FIELD: ["target"] * 5 + ["other"] * 5})
    candidate_artifacts: dict[str, Any] = {
        "cluster_labels": ["3"] * 5 + ["1"] * 5,
        DE_TABLE_ARTIFACT: pd.DataFrame(
            {"gene": ["GENE1", "GENE2"], "group": ["7", "7"]}
        ),
    }

    resolved = scored_group_metadata(
        candidate_artifacts,
        {"labels": reference_frame},
        {DE_REFERENCE_TARGET: "target"},
    )

    assert resolved == {}


def test_a_resolvable_group_present_in_the_table_is_recorded() -> None:
    reference_frame = pd.DataFrame({REFERENCE_LABEL_FIELD: ["target"] * 5 + ["other"] * 5})
    candidate_artifacts: dict[str, Any] = {
        "cluster_labels": ["3"] * 5 + ["1"] * 5,
        DE_TABLE_ARTIFACT: pd.DataFrame(
            {"gene": ["GENE1", "GENE2"], "group": ["3", "1"]}
        ),
    }

    resolved = scored_group_metadata(
        candidate_artifacts,
        {"labels": reference_frame},
        {DE_REFERENCE_TARGET: "target"},
    )

    assert resolved == {DE_SCORED_GROUP: "3"}


def test_the_annotation_grouping_is_tried_when_the_clustering_does_not_match() -> None:
    """The agent chooses what it grouped by, so both of its groupings are offered."""
    reference_frame = pd.DataFrame({REFERENCE_LABEL_FIELD: ["target"] * 5 + ["other"] * 5})
    candidate_artifacts: dict[str, Any] = {
        "cluster_labels": ["3"] * 5 + ["1"] * 5,
        "prediction": pd.DataFrame(
            {CANDIDATE_LABEL_FIELD: ["Monocyte"] * 5 + ["T cell"] * 5}
        ),
        DE_TABLE_ARTIFACT: pd.DataFrame(
            {"gene": ["GENE1"], "group": ["Monocyte"]}
        ),
    }

    resolved = scored_group_metadata(
        candidate_artifacts,
        {"labels": reference_frame},
        {DE_REFERENCE_TARGET: "target"},
    )

    assert resolved == {DE_SCORED_GROUP: "Monocyte"}


def test_a_table_with_no_group_column_is_read_whole() -> None:
    """A single-group table needs no filter, and inventing one would empty it."""
    reference_frame = pd.DataFrame({REFERENCE_LABEL_FIELD: ["target"] * 5 + ["other"] * 5})
    candidate_artifacts: dict[str, Any] = {
        "cluster_labels": ["3"] * 5 + ["1"] * 5,
        DE_TABLE_ARTIFACT: pd.DataFrame({"gene": ["GENE1", "GENE2"]}),
    }

    resolved = scored_group_metadata(
        candidate_artifacts,
        {"labels": reference_frame},
        {DE_REFERENCE_TARGET: "target"},
    )

    assert resolved == {DE_SCORED_GROUP: "3"}
