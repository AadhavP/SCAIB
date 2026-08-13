"""The names the differential-expression evidence channel is spelled with.

Four sites must agree on these exactly, in three subsystems that do not import
each other: the evaluator that *produces* the reference evidence
(``evaluation/reference_de.py``), the extractor that *reads* it
(``metrics/builtin/_helpers.py``), the metric definitions that declare it as a
structural requirement (``metrics/builtin/differential_expression.py``), and the
assembly point that supplies the candidate table (``evaluation/candidates.py``).

A disagreement here fails **silently**, which is the whole reason this module
exists. ``metadata.get("reference_markers")`` returning ``None`` because the
producer wrote ``reference_marker`` is indistinguishable from a benchmark that
declared no reference at all: the metric is excluded, the domain reports
unmeasured, and the run looks like an honest gap. Same rule that produced
:mod:`agent_evals.core.reference_columns`,
:mod:`agent_evals.core.intent_parameters`, and
:mod:`agent_evals.core.progress_keys`.

The counter-case, for contrast: ``_WEIGHT_SUM_TOLERANCE`` is deliberately
duplicated between ``benchmarks/schema.py`` and ``evaluation/global_score.py``,
because both layers reject a mis-declared weight set *loudly*. Silence is the
criterion, not duplication.
"""

from __future__ import annotations

#: Metric-context metadata key holding the reference marker gene list. The
#: structural requirement of every ranked DE metric: evaluator-owned evidence, so
#: its absence must *exclude* the metric rather than score it zero.
REFERENCE_MARKERS = "reference_markers"

#: Metric-context metadata key holding per-gene reference effect sizes. A separate
#: requirement from the marker list, because a benchmark can supply one without
#: the other and the two metrics that read this read nothing else.
REFERENCE_EFFECT_SIZES = "reference_effect_sizes"

#: Metric-context metadata key naming which group of a multi-group candidate DE
#: table holds the ranking to score. Candidate-side, so deliberately not a
#: structural requirement anywhere.
DE_SCORED_GROUP = "de_scored_group"

#: Metric-context metadata key carrying the reference pipeline's provenance
#: record, so a published DE number is attributable to a pinned implementation.
DE_REFERENCE_IMPLEMENTATION = "de_reference_implementation"

#: Metric-context metadata key naming the *reference* population being scored.
#: Written by the evaluator before the run and read only by the evaluator, to
#: resolve which of the agent's own groups to compare against. Never published to
#: the agent: it is a withheld class name, and the agent does not need it, because
#: characterizing every population it finds answers the task either way.
DE_REFERENCE_TARGET = "de_reference_target"

#: Metric-context metadata key holding the top-K cutoff. Spelled ``k`` because
#: ``precision_at_k`` and ``recall_at_k`` already read it under that name.
DE_TOP_K = "k"

#: Default top-K when a benchmark declares no cutoff. 50 is the conventional size
#: of a marker panel.
DEFAULT_TOP_K = 50

#: ``candidate_artifacts`` key holding the agent's ranked DE table.
DE_TABLE_ARTIFACT = "de_table"

#: Column of a candidate DE table naming which group each row was ranked for.
#: ``scanpy.get.rank_genes_groups_df(adata, group=None)`` emits it and stacks every
#: tested group into one frame, so a table carrying it is several rankings
#: concatenated rather than one.
DE_TABLE_GROUP_COLUMN = "group"

#: Gene-name columns of a candidate DE table, in preference order. ``gene`` is
#: what the benchmark's artifact contract requires; ``names`` is what scanpy emits
#: before the operation renames it, and is accepted so a table written by an
#: agent's own scanpy call is still readable.
DE_TABLE_GENE_COLUMNS = ("gene", "names")

#: Effect-size column of a candidate DE table.
DE_TABLE_EFFECT_COLUMN = "effect_size"


__all__ = [
    "DEFAULT_TOP_K",
    "DE_REFERENCE_IMPLEMENTATION",
    "DE_REFERENCE_TARGET",
    "DE_SCORED_GROUP",
    "DE_TABLE_ARTIFACT",
    "DE_TABLE_EFFECT_COLUMN",
    "DE_TABLE_GENE_COLUMNS",
    "DE_TABLE_GROUP_COLUMN",
    "DE_TOP_K",
    "REFERENCE_EFFECT_SIZES",
    "REFERENCE_MARKERS",
]
