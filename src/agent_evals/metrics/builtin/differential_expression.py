"""Ranked differential-expression metric definitions."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_evals.metrics.builtin._helpers import de_ranked, failed
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.models import (
    MetricApplicability,
    MetricCategory,
    MetricDefinition,
    MetricDirection,
    MetricRole,
    NormalizationSpec,
)
from agent_evals.metrics.registry import MetricComputation

#: Evaluator-owned reference evidence, named per metric because these two
#: requirements are not interchangeable and the difference decides whether an
#: unanswerable metric is *excluded* or scored zero.
#:
#: A ranked metric needs a marker set; the effect-size pair needs per-gene effect
#: sizes. Every definition here used to declare ``reference_markers``, including
#: the two that read ``reference_effect_sizes`` and nothing else -- so on a
#: benchmark supplying markers alone they passed the structural gate, found no
#: effect sizes, and returned ``failed(...)``, which lands at ``failure_score``.
#: That is a manufactured 0.0 for evidence the *evaluator* never supplied,
#: charged to the agent, and it would have collapsed the whole DE domain through
#: the geometric mean.
REFERENCE_MARKERS = "reference_markers"
REFERENCE_EFFECT_SIZES = "reference_effect_sizes"


def _definition(
    metric_id: str,
    name: str,
    role: MetricRole,
    *,
    structural_metadata: str = REFERENCE_MARKERS,
    native_min: float = 0,
    normalization: NormalizationSpec | None = None,
) -> MetricDefinition:
    return MetricDefinition(
        metric_id=metric_id,
        name=name,
        version="1.0",
        description=name,
        category=MetricCategory.DIFFERENTIAL_EXPRESSION,
        role=role,
        direction=MetricDirection.HIGHER_IS_BETTER,
        native_min=native_min,
        native_max=1,
        applicability=MetricApplicability(
            # Candidate-side on purpose: the DE table is the agent's own output,
            # so its absence is the agent's failure and *should* score zero.
            required_artifacts=["de_table"],
            structural_metadata=[structural_metadata],
        ),
        computation_backend="sklearn/scipy",
        normalization=normalization or NormalizationSpec(policy="bounded"),
    )


def _inputs(context: ScientificMetricContext) -> tuple[list[str], set[str]] | None:
    return de_ranked(context)


def precision_at_k(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("ranked DE table and reference markers unavailable")
    ranked, reference = values
    k = min(int(context.metadata.get("k", 50)), len(ranked))
    return MetricComputation(len(set(ranked[:k]) & reference) / max(k, 1), metadata={"k": k})


def recall_at_k(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("ranked DE table and reference markers unavailable")
    ranked, reference = values
    k = min(int(context.metadata.get("k", 50)), len(ranked))
    return MetricComputation(len(set(ranked[:k]) & reference) / max(len(reference), 1), metadata={"k": k})


def f1_at_k(context: ScientificMetricContext) -> MetricComputation:
    precision = precision_at_k(context)
    recall = recall_at_k(context)
    if not isinstance(precision.raw_value, (float, int)) or not isinstance(recall.raw_value, (float, int)):
        return failed("precision and recall unavailable")
    denominator = float(precision.raw_value) + float(recall.raw_value)
    return MetricComputation(2 * float(precision.raw_value) * float(recall.raw_value) / denominator if denominator else 0.0)


def auroc(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("ranked DE table and reference markers unavailable")
    ranked, reference = values
    from sklearn.metrics import roc_auc_score

    truth = [int(gene in reference) for gene in ranked]
    if len(set(truth)) < 2:
        return failed("ranked list has no positive and negative marker examples")
    return MetricComputation(float(roc_auc_score(truth, list(range(len(ranked), 0, -1)))))


def auprc(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("ranked DE table and reference markers unavailable")
    ranked, reference = values
    from sklearn.metrics import average_precision_score

    truth = [int(gene in reference) for gene in ranked]
    if len(set(truth)) < 2:
        return failed("ranked list has no positive and negative marker examples")
    return MetricComputation(float(average_precision_score(truth, list(range(len(ranked), 0, -1)))))


def rank_biased_overlap(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("ranked DE table and reference markers unavailable")
    ranked, reference = values
    persistence = float(context.metadata.get("rbo_p", 0.9))
    overlap = 0.0
    seen: set[str] = set()
    for depth, gene in enumerate(ranked, start=1):
        seen.add(gene)
        overlap += persistence ** (depth - 1) * len(seen & reference) / depth
    denominator = sum(persistence**index for index in range(len(ranked)))
    return MetricComputation(overlap / denominator if denominator else 0.0, metadata={"p": persistence})


def effect_size_correlation(context: ScientificMetricContext) -> MetricComputation:
    table = context.candidate_artifacts.get("de_table")
    reference = context.metadata.get("reference_effect_sizes")
    if table is None or reference is None or "effect_size" not in table:
        return failed("candidate and reference effect sizes are required")
    from scipy.stats import pearsonr  # type: ignore[import-untyped]

    common = [gene for gene in table["gene"] if gene in reference]
    if len(common) < 2:
        return failed("fewer than two common effect-size genes")
    candidate = [float(table.loc[table["gene"] == gene, "effect_size"].iloc[0]) for gene in common]
    value = float(pearsonr(candidate, [float(reference[gene]) for gene in common]).statistic)
    return MetricComputation(value, metadata={"genes": len(common)})


def direction_agreement(context: ScientificMetricContext) -> MetricComputation:
    table = context.candidate_artifacts.get("de_table")
    reference = context.metadata.get("reference_effect_sizes")
    if table is None or reference is None or "effect_size" not in table:
        return failed("candidate and reference effect sizes are required")
    common = [gene for gene in table["gene"] if gene in reference]
    if not common:
        return failed("no common effect-size genes")
    return MetricComputation(float(np.mean([np.sign(float(table.loc[table["gene"] == gene, "effect_size"].iloc[0])) == np.sign(float(reference[gene])) for gene in common])))


def de_definitions() -> list[tuple[MetricDefinition, Any]]:
    return [
        (_definition("differential_expression.precision_at_k", "Precision@K", MetricRole.PRIMARY), precision_at_k),
        (_definition("differential_expression.recall_at_k", "Recall@K", MetricRole.PRIMARY), recall_at_k),
        (_definition("differential_expression.f1_at_k", "F1@K", MetricRole.PRIMARY), f1_at_k),
        (_definition("differential_expression.auprc", "AUPRC", MetricRole.PRIMARY), auprc),
        (_definition("differential_expression.auroc", "AUROC", MetricRole.PRIMARY), auroc),
        (_definition("differential_expression.rbo", "Rank-biased overlap", MetricRole.SECONDARY), rank_biased_overlap),
        # Pearson's r is native to [-1, 1], and the ``symmetric`` policy is what
        # maps that honestly. Declared ``native_min=0`` with ``bounded``, an
        # anti-correlated result clamped to exactly 0.0 while a barely-positive
        # one normalized to 0.01 -- and in a geometric mean 0.0 annihilates the
        # domain while 0.01 does not, so the score fell off a cliff at r = 0
        # rather than declining through it.
        (
            _definition(
                "differential_expression.effect_size_correlation",
                "Effect-size correlation",
                MetricRole.SECONDARY,
                structural_metadata=REFERENCE_EFFECT_SIZES,
                native_min=-1,
                normalization=NormalizationSpec(policy="symmetric"),
            ),
            effect_size_correlation,
        ),
        (
            _definition(
                "differential_expression.direction_agreement",
                "Direction agreement",
                MetricRole.SECONDARY,
                structural_metadata=REFERENCE_EFFECT_SIZES,
            ),
            direction_agreement,
        ),
    ]


__all__ = ["de_definitions"]
