"""Evaluator-side reference differential expression for the DE benchmark family.

The nine ``differential_expression.*`` metrics all score one thing: how well a
ranked gene list recovers a held-out marker set.  Until now nothing in the repo
produced that marker set, so every one of them was structurally ineligible and
the DE benchmark honestly reported its scientific outcome as unmeasured.  This
module is the missing evidence.

**Where the reference comes from, declared rather than implied.** SCAIB runs a
pinned differential-expression pipeline itself, evaluator-side, on the held-out
reference grouping: :func:`scanpy.tl.rank_genes_groups` with a Wilcoxon rank-sum
test, one declared population against the rest, top-K by test statistic.  Two
alternatives were considered and rejected.  A literature-curated marker panel is
independent of any pipeline but hardcodes an answer key into source and yields no
per-gene effect sizes, so ``effect_size_correlation`` and ``direction_agreement``
would stay permanently unmeasurable.  The dataset's own shipped
``uns["rank_genes_groups"]`` cannot serve either: on pbmc68k it carries only
``names`` and ``scores`` -- no fold changes and no p-values -- and it is computed
with ``method="logreg"`` over *every* reference population at once.

The obvious objection to a harness-computed reference is that it scores "did you
reproduce our pipeline" rather than "did you find the biology".  The answer is in
what the metrics read: precision@K, AUROC, and RBO compare *rank agreement*
against a set, never value equality, so an agent using a t-test or a pseudobulk
negative binomial recovers the same top genes by a different route and scores the
same.  Only ``effect_size_correlation`` reads magnitudes, and it is optional in
the profile precisely because it is the one that couples to method choice.

**The ranking is taken from ``.raw`` when the dataset ships one, matching what
the agent's own DE operation reads by default.** An earlier version forced
``use_raw=False`` on the theory that ``.raw`` holds an unfiltered gene universe
the agent's table could not contain.  Measurement says otherwise for
pbmc68k_reduced: ``.raw`` carries the same 765 genes as ``.X``, the top-50 ranking
is *identical* either way -- Wilcoxon is rank-based and per-gene scaling is
monotone -- and the difference is entirely in the effect sizes.  ``.X`` is
z-scored, so ``log2`` of a negative group mean is NaN and all 50 fold changes come
back non-finite, which silently left ``effect_size_correlation`` and
``direction_agreement`` permanently unmeasurable.  Read from ``.raw`` all 50 are
finite and in the expected range.  Symmetry with the candidate is the reason to
prefer it, not the fold changes: ``operations/de.py`` takes scanpy's default and
so ranks over ``.raw`` too, and a reference on a different layer would be scoring
a layer choice SCAIB made, not the agent's biology.

**The contrast is evaluator-hidden, deliberately.** The population scored is
declared here and never published to the agent.  It names a reference class, so
publishing it would hand over one of the withheld label spellings; and the task
does not need it, because an agent that characterizes every population it finds
answers the question without being told which one is graded.  Naming the target
in the benchmark YAML was the first design and was dropped for that reason.

**The candidate side is resolved by overlap, not by name.** The agent groups by
its own clustering or annotation, whose class names are its own choice, so the
evaluator cannot look up "the agent's CD14 monocytes" by string.
:func:`resolve_candidate_group` picks the candidate group with the highest
Jaccard overlap with the reference population.  Jaccard rather than raw overlap
on purpose: raw overlap is maximized by one giant cluster containing everything,
and Jaccard penalizes exactly that.

Nothing here raises.  Gathering evidence is observation, and an observation that
throws would report a harness failure as an agent failure -- so every failure
becomes a ``reason`` and leaves the metrics ineligible rather than scored at zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_evals.core.de_evidence import (
    DE_REFERENCE_IMPLEMENTATION,
    DE_REFERENCE_TARGET,
    DE_SCORED_GROUP,
    DE_TABLE_ARTIFACT,
    DE_TABLE_GROUP_COLUMN,
    DE_TOP_K,
    DEFAULT_TOP_K,
    REFERENCE_EFFECT_SIZES,
    REFERENCE_MARKERS,
)
from agent_evals.evaluation.candidates import (
    CANDIDATE_LABEL_FIELD,
    REFERENCE_LABEL_FIELD,
)

#: Pinned reference test. Rank-sum rather than the dataset's shipped ``logreg``:
#: it is the scanpy default, it is non-parametric, and it produces the fold
#: changes and adjusted p-values the effect-size channel needs.
REFERENCE_METHOD = "wilcoxon"

#: Private column the reference grouping is written to on a throwaway copy. Named
#: with sentinels so it can never collide with an agent-produced column, and never
#: written to the live object -- ``ScientificContext.record_produced_columns``
#: would refuse a reserved name, but the guard is not the reason this is a copy.
_REFERENCE_GROUP_COLUMN = "__scaib_reference_group__"


@dataclass(frozen=True)
class ReferenceContrast:
    """The population a DE benchmark is scored on, and how it is contrasted."""

    benchmark: str
    #: Reference class name, in the reference vocabulary. Evaluator-only.
    group: str
    against: str = "rest"
    top_k: int = DEFAULT_TOP_K


#: Benchmark id -> the contrast its scientific outcome is measured on.
#:
#: ``CD14+ Monocyte`` is the pbmc68k population chosen because it is both large
#: enough to support a rank test (129 of 700 cells at ``--max-cells 2000``) and
#: not the largest, so an agent cannot reach it by under-clustering into one
#: group. The benchmark's original declaration contrasted ``CD14_Monocytes``
#: against ``FCGR3A_Monocytes``; measurement showed pbmc68k_reduced carries no
#: FCGR3A population at all and spells the CD14 one differently, so that contrast
#: was unrunnable regardless of where its reference came from.
_CONTRASTS: dict[str, ReferenceContrast] = {
    "pbmc-differential-expression": ReferenceContrast(
        benchmark="pbmc-differential-expression",
        group="CD14+ Monocyte",
    ),
}


def contrast_for(benchmark_id: str) -> ReferenceContrast | None:
    """Return the declared contrast for a benchmark, or ``None`` for the rest.

    ``None`` means this benchmark's scientific outcome does not rest on marker
    recovery, so no reference DE is computed and the DE metrics stay ineligible
    rather than being scored against a contrast nobody declared.
    """
    return _CONTRASTS.get(benchmark_id)


@dataclass(frozen=True)
class ReferenceMarkerSet:
    """The reference marker evidence, or a recorded reason there is none."""

    genes: tuple[str, ...] = ()
    effect_sizes: Mapping[str, float] = field(default_factory=dict)
    implementation: Mapping[str, Any] = field(default_factory=dict)
    #: Why the reference could not be computed. ``None`` when it was.
    reason: str | None = None

    @property
    def available(self) -> bool:
        """True when there is a marker set to score against."""
        return bool(self.genes)

    def metadata(self, contrast: ReferenceContrast) -> dict[str, Any]:
        """The metric-context metadata this evidence supplies.

        Empty when unavailable, which is what leaves every DE metric
        *structurally* ineligible.  Supplying an empty marker list instead would
        make them eligible with unusable evidence, and the engine scores that at
        the failure value -- the same manufactured zero by a longer route.
        """
        if not self.available:
            return {}
        return {
            REFERENCE_MARKERS: list(self.genes),
            REFERENCE_EFFECT_SIZES: dict(self.effect_sizes),
            DE_TOP_K: contrast.top_k,
            DE_REFERENCE_TARGET: contrast.group,
            DE_REFERENCE_IMPLEMENTATION: dict(self.implementation),
        }


def reference_de_metadata(
    benchmark_id: str,
    adata: Any,
    reference_artifacts: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Build the DE evidence for a benchmark, plus a reason when there is none.

    Called once per run, before the agent starts, for two reasons. The reference
    is not a function of anything the agent does, so recomputing it per step would
    pay for a Wilcoxon test six times over; and computing it once means the
    per-step ``dS`` and the final ``O`` are scored against byte-identical evidence,
    which is the same guarantee :mod:`agent_evals.evaluation.candidates` exists to
    give the candidate side.

    The consequence, recorded because it is a real design choice: the reference is
    ranked over the dataset **as issued**, so a gene the agent later filters away
    cannot be recovered. That is charged to its QC decision, which is where a gene
    it chose to discard belongs.

    Returns ``({}, None)`` for a benchmark that declares no contrast -- no
    evidence, and no gap either, because marker recovery is not what it measures.
    """
    contrast = contrast_for(benchmark_id)
    if contrast is None:
        return {}, None
    labels = _reference_labels(reference_artifacts)
    if labels is None:
        return {}, (
            "this dataset carries no reference population labels, so the "
            "differential-expression reference could not be computed and "
            "marker-recovery metrics are unmeasured rather than zero"
        )
    markers = compute_reference_markers(adata, labels=labels, contrast=contrast)
    return markers.metadata(contrast), markers.reason


def scored_group_metadata(
    candidate_artifacts: Mapping[str, Any],
    reference_artifacts: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve which of the agent's own groups the DE ranking is read from.

    Two candidate groupings are tried, because the agent chooses what it grouped
    by: its clustering, or its annotation.  A resolved name is accepted only if it
    appears in the DE table's own ``group`` column when that column exists --
    otherwise a name resolved from the clustering would be used to filter a table
    grouped by annotation, and the filter would match nothing, silently turning a
    real ranking into an empty one.

    Returns ``{}`` when nothing can be resolved, which leaves the table read whole.
    That is the right reading for a single-group table and a harsh but honest one
    for a multi-group table: it is still a ranking of genes the agent produced.
    """
    target = metadata.get(DE_REFERENCE_TARGET)
    reference = _reference_labels(reference_artifacts)
    if target is None or reference is None:
        return {}
    available = _table_groups(candidate_artifacts.get(DE_TABLE_ARTIFACT))
    for candidate in _candidate_groupings(candidate_artifacts):
        group = resolve_candidate_group(candidate, reference, str(target))
        if group is None:
            continue
        if available is None or group in available:
            return {DE_SCORED_GROUP: group}
    return {}


def _reference_labels(reference_artifacts: Mapping[str, Any]) -> list[str] | None:
    """Read the reference population per cell from the evaluator-side channel."""
    labels = reference_artifacts.get("labels")
    if labels is None:
        return None
    columns = getattr(labels, "columns", None)
    if columns is not None and REFERENCE_LABEL_FIELD in columns:
        return [str(value) for value in labels[REFERENCE_LABEL_FIELD].tolist()]
    if hasattr(labels, "tolist"):
        return [str(value) for value in labels.tolist()]
    return None


def _candidate_groupings(candidate_artifacts: Mapping[str, Any]) -> list[list[str]]:
    """The agent's own groupings, in the order they are tried."""
    groupings: list[list[str]] = []
    clusters = candidate_artifacts.get("cluster_labels")
    if clusters is not None:
        groupings.append([str(value) for value in clusters])
    prediction: Any = candidate_artifacts.get("prediction")
    columns = getattr(prediction, "columns", None)
    if columns is not None and CANDIDATE_LABEL_FIELD in columns:
        groupings.append([str(value) for value in prediction[CANDIDATE_LABEL_FIELD]])
    return groupings


def _table_groups(table: Any) -> set[str] | None:
    """The group names a candidate DE table carries, or ``None`` if it has none."""
    columns = getattr(table, "columns", None)
    if columns is None or DE_TABLE_GROUP_COLUMN not in columns:
        return None
    return {str(value) for value in table[DE_TABLE_GROUP_COLUMN].tolist()}


def compute_reference_markers(
    adata: Any,
    *,
    labels: Sequence[str],
    contrast: ReferenceContrast,
) -> ReferenceMarkerSet:
    """Rank reference markers for ``contrast`` from the held-out labels.

    ``labels`` are the reference classes in ``adata.obs_names`` order, read from
    the evaluator-side channel rather than from ``adata.obs`` -- the reference
    column may already have been stripped from the object the agent reached.
    """
    try:
        return _compute(adata, labels, contrast)
    # Deliberately broad, for the same reason ``StageAwareRewardEvaluator._record``
    # is: this is evidence gathering, and any backend exception must become a
    # recorded gap rather than a failed run.
    except Exception as error:
        return ReferenceMarkerSet(
            reason=(
                "the reference differential-expression pipeline could not be run, "
                "so marker-recovery metrics are unmeasured rather than zero: "
                f"{type(error).__name__}: {error}"
            )
        )


def _compute(
    adata: Any,
    labels: Sequence[str],
    contrast: ReferenceContrast,
) -> ReferenceMarkerSet:
    """Run the pinned reference pipeline, reporting its own gaps as reasons."""
    import pandas as pd
    import scanpy as sc

    values = [str(value) for value in labels]
    if len(values) != int(adata.n_obs):
        return ReferenceMarkerSet(
            reason=(
                f"the reference grouping covers {len(values)} cells but the scored "
                f"dataset holds {int(adata.n_obs)}, so the two cannot be joined and "
                "marker-recovery metrics are unmeasured rather than zero"
            )
        )
    in_group = sum(1 for value in values if value == contrast.group)
    if in_group < 2:
        return ReferenceMarkerSet(
            reason=(
                f"the reference population '{contrast.group}' holds {in_group} "
                "cell(s) in this dataset, which cannot support a rank test, so "
                "marker-recovery metrics are unmeasured rather than zero"
            )
        )
    if in_group == len(values):
        return ReferenceMarkerSet(
            reason=(
                f"every cell belongs to the reference population "
                f"'{contrast.group}', so there is nothing to contrast it against "
                "and marker-recovery metrics are unmeasured rather than zero"
            )
        )
    # A throwaway copy, never the scored object: writing the reference grouping
    # onto ``context.adata`` would put the answer key back into the data plane
    # Stage 0 removed it from, one layer below every guard that checks for it.
    work = adata.copy()
    work.obs[_REFERENCE_GROUP_COLUMN] = pd.Categorical(values)
    use_raw = getattr(adata, "raw", None) is not None
    sc.tl.rank_genes_groups(
        work,
        groupby=_REFERENCE_GROUP_COLUMN,
        groups=[contrast.group],
        reference=contrast.against,
        method=REFERENCE_METHOD,
        use_raw=use_raw,
    )
    frame = sc.get.rank_genes_groups_df(work, group=contrast.group)
    frame = frame.head(contrast.top_k)
    genes = tuple(str(value) for value in frame["names"].tolist())
    effect_sizes = _effect_sizes(frame, genes)
    return ReferenceMarkerSet(
        genes=genes,
        effect_sizes=effect_sizes,
        implementation={
            "provider": "scanpy",
            "metric": "rank_genes_groups",
            "version": _scanpy_version(),
            "parameters": {
                "method": REFERENCE_METHOD,
                "group": contrast.group,
                "reference": contrast.against,
                "top_k": contrast.top_k,
                "use_raw": use_raw,
                # The gene universe the reference was ranked over. A run whose
                # agent filtered genes away is scored against the universe it was
                # given, so a marker it removed is a consequence of its own QC.
                "n_genes": int(adata.n_vars),
                "n_cells_in_group": in_group,
                "n_cells_against": len(values) - in_group,
            },
        },
    )


def _scanpy_version() -> str:
    """Resolve scanpy's version from dist metadata rather than ``__version__``.

    ``scanpy.__version__`` is deprecated and warns; a provenance record that emits
    a ``FutureWarning`` every run would train whoever reads the logs to ignore them.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("scanpy")
    except PackageNotFoundError:
        return "unknown"


def _effect_sizes(frame: Any, genes: Sequence[str]) -> dict[str, float]:
    """Per-gene reference effect sizes, omitting any that are not finite.

    An omitted gene leaves ``effect_size_correlation`` with fewer paired points
    rather than a NaN that would silently poison Pearson's r for the whole run.
    """
    import math

    if "logfoldchanges" not in frame.columns:
        return {}
    values = frame["logfoldchanges"].tolist()
    return {
        gene: float(value)
        for gene, value in zip(genes, values, strict=False)
        if value is not None and math.isfinite(float(value))
    }


def resolve_candidate_group(
    candidate: Sequence[str],
    reference: Sequence[str],
    target: str,
) -> str | None:
    """Name the candidate group best matching ``target`` by Jaccard overlap.

    Jaccard rather than raw overlap count.  Raw overlap is maximized by a single
    cluster holding every cell, so it would reward under-clustering; Jaccard
    divides by the union and so penalizes a group that is much larger than the
    reference population as heavily as one that is much smaller.

    Returns ``None`` when the two sequences cannot be compared or no candidate
    group contains a single cell of the target -- which is a fact about the
    candidate and is left to be scored, not turned into an ineligibility.
    """
    if len(candidate) != len(reference) or not candidate:
        return None
    overlap: dict[str, int] = {}
    sizes: dict[str, int] = {}
    target_size = 0
    for candidate_value, reference_value in zip(candidate, reference, strict=True):
        name = str(candidate_value)
        sizes[name] = sizes.get(name, 0) + 1
        if str(reference_value) == target:
            target_size += 1
            overlap[name] = overlap.get(name, 0) + 1
    if not overlap or target_size == 0:
        return None
    return max(
        overlap,
        key=lambda name: (
            overlap[name] / (sizes[name] + target_size - overlap[name]),
            # Deterministic tie-break, so two runs on the same data resolve the
            # same group and the score is reproducible.
            -sizes[name],
            name,
        ),
    )


__all__ = [
    "REFERENCE_METHOD",
    "ReferenceContrast",
    "ReferenceMarkerSet",
    "compute_reference_markers",
    "contrast_for",
    "reference_de_metadata",
    "resolve_candidate_group",
    "scored_group_metadata",
]
