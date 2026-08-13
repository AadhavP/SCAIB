"""Ranked differential-expression metric definitions."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_evals.core.de_evidence import (
    DE_TABLE_ARTIFACT,
    DE_TOP_K,
    DEFAULT_TOP_K,
    REFERENCE_EFFECT_SIZES,
    REFERENCE_MARKERS,
)
from agent_evals.metrics.builtin._helpers import de_effect_sizes, de_ranked, failed
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
#:
#: Both names now live in :mod:`agent_evals.core.de_evidence`, because the producer
#: added in Stage 8 is a third site that has to spell them identically and a
#: misspelling there is silent. Imported rather than restated so this module still
#: reads as the place the requirements are declared.


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
            required_artifacts=[DE_TABLE_ARTIFACT],
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
    k = min(int(context.metadata.get(DE_TOP_K, DEFAULT_TOP_K)), len(ranked))
    return MetricComputation(len(set(ranked[:k]) & reference) / max(k, 1), metadata={"k": k})


def recall_at_k(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("ranked DE table and reference markers unavailable")
    ranked, reference = values
    k = min(int(context.metadata.get(DE_TOP_K, DEFAULT_TOP_K)), len(ranked))
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
    paired = de_effect_sizes(context)
    if paired is None:
        return failed("candidate and reference effect sizes are required")
    candidate, reference = paired
    if len(candidate) < 2:
        return failed("fewer than two common effect-size genes")
    from scipy.stats import pearsonr  # type: ignore[import-untyped]

    genes = list(candidate)
    observed = [candidate[gene] for gene in genes]
    expected = [reference[gene] for gene in genes]
    # Pearson's r is undefined when either side is constant, and scipy returns NaN
    # with a warning rather than raising. A NaN would normalize to a number here,
    # so the degenerate case has to be caught before it reaches the aggregator.
    if len(set(observed)) < 2 or len(set(expected)) < 2:
        return failed("effect sizes are constant on one side, so r is undefined")
    return MetricComputation(
        float(pearsonr(observed, expected).statistic), metadata={"genes": len(genes)}
    )


def direction_agreement(context: ScientificMetricContext) -> MetricComputation:
    paired = de_effect_sizes(context)
    if paired is None:
        return failed("candidate and reference effect sizes are required")
    candidate, reference = paired
    if not candidate:
        return failed("no common effect-size genes")
    agreement = [
        np.sign(value) == np.sign(reference[gene]) for gene, value in candidate.items()
    ]
    return MetricComputation(
        float(np.mean(agreement)), metadata={"genes": len(candidate)}
    )


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
