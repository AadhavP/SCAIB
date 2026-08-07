"""Cell-annotation metric definitions and computations."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from agent_evals.metrics.backends.sklearn import (
    balanced_accuracy,
    f1_macro,
    matthews_correlation,
)
from agent_evals.metrics.builtin._helpers import failed, labels
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


def _definition(
    metric_id: str,
    name: str,
    role: MetricRole,
    *,
    backend: str = "sklearn",
    requires_prediction: bool = True,
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
    native_min: float = 0,
    native_max: float = 1,
) -> MetricDefinition:
    return MetricDefinition(
        metric_id=metric_id,
        name=name,
        version="1.0",
        description=name,
        category=MetricCategory.ANNOTATION,
        role=role,
        direction=direction,
        native_min=native_min,
        native_max=native_max,
        applicability=MetricApplicability(
            required_artifacts=["prediction"] if requires_prediction else [],
            requires_reference_labels=True,
            requires_predictions=requires_prediction,
        ),
        computation_backend=backend,
        normalization=NormalizationSpec(policy="bounded"),
    )


def _inputs(context: ScientificMetricContext) -> tuple[Any, Any] | None:
    return labels(context)


def macro_f1(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    return failed("prediction or reference labels unavailable") if values is None else MetricComputation(f1_macro(*values))


def mcc(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    return failed("prediction or reference labels unavailable") if values is None else MetricComputation(matthews_correlation(*values))


def balanced(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    return failed("prediction or reference labels unavailable") if values is None else MetricComputation(balanced_accuracy(*values))


def rare_recall(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("prediction or reference labels unavailable")
    reference, predicted = values
    from sklearn.metrics import recall_score

    counts = np.unique(reference, return_counts=True)
    rare = set(counts[0][counts[1] <= max(2, int(np.percentile(counts[1], 25)))])
    recalls = recall_score(reference, predicted, labels=sorted(rare), average=None, zero_division=0)
    return MetricComputation(float(np.mean(recalls)), metadata={"rare_labels": sorted(rare)})


def exact_accuracy(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    return failed("prediction or reference labels unavailable") if values is None else MetricComputation(float(np.mean(values[0] == values[1])))


def brier_score(context: ScientificMetricContext) -> MetricComputation:
    table = context.candidate_artifacts.get("prediction")
    if table is None or "confidence" not in table:
        return failed("confidence is not present in prediction artifact")
    reference, predicted = labels(context) or (None, None)
    if reference is None:
        return failed("reference labels unavailable")
    confidence = np.asarray(table["confidence"], dtype=float)
    correct = np.asarray(reference) == np.asarray(predicted)
    return MetricComputation(float(np.mean((confidence - correct.astype(float)) ** 2)))


def expected_calibration_error(context: ScientificMetricContext) -> MetricComputation:
    table = context.candidate_artifacts.get("prediction")
    values = labels(context)
    if table is None or "confidence" not in table or values is None:
        return failed("confidence and labels are required")
    confidence = np.asarray(table["confidence"], dtype=float)
    correct = (np.asarray(values[0]) == np.asarray(values[1])).astype(float)
    bins = np.linspace(0, 1, 11)
    error = 0.0
    for low, high in pairwise(bins):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return MetricComputation(error)


def hierarchical_f1(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    hierarchy = context.metadata.get("label_hierarchy")
    if values is None or not isinstance(hierarchy, dict):
        return failed("label hierarchy is not provided")
    reference, predicted = values
    mapped_reference = [hierarchy.get(str(value), value) for value in reference]
    mapped_predicted = [hierarchy.get(str(value), value) for value in predicted]
    return MetricComputation(f1_macro(mapped_reference, mapped_predicted))


def per_class_precision(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("prediction or reference labels unavailable")
    from sklearn.metrics import precision_score

    return MetricComputation(
        precision_score(values[0], values[1], average=None, zero_division=0).tolist()
    )


def per_class_recall(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("prediction or reference labels unavailable")
    from sklearn.metrics import recall_score

    return MetricComputation(recall_score(values[0], values[1], average=None, zero_division=0).tolist())


def confusion_matrix_data(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("prediction or reference labels unavailable")
    from sklearn.metrics import confusion_matrix

    labels_order = sorted(set(values[0]) | set(values[1]))
    return MetricComputation(
        confusion_matrix(values[0], values[1], labels=labels_order).tolist(),
        metadata={"labels": labels_order},
    )


def coverage(context: ScientificMetricContext) -> MetricComputation:
    table = context.candidate_artifacts.get("prediction")
    if table is None:
        return failed("prediction artifact unavailable")
    values = table["predicted_label"]
    return MetricComputation(float(values.notna().mean()))


def annotation_definitions() -> list[tuple[MetricDefinition, Any]]:
    """Return the annotation catalog."""
    return [
        (_definition("cell_annotation.macro_f1", "Macro F1", MetricRole.PRIMARY), macro_f1),
        (_definition("cell_annotation.mcc", "Multiclass MCC", MetricRole.PRIMARY, native_min=-1), mcc),
        (_definition("cell_annotation.balanced_accuracy", "Balanced accuracy", MetricRole.PRIMARY), balanced),
        (_definition("cell_annotation.rare_recall", "Rare-class recall", MetricRole.PRIMARY), rare_recall),
        (_definition("cell_annotation.accuracy", "Exact accuracy", MetricRole.PRIMARY), exact_accuracy),
        (_definition("cell_annotation.brier", "Brier score", MetricRole.SECONDARY, backend="sklearn", direction=MetricDirection.LOWER_IS_BETTER), brier_score),
        (_definition("cell_annotation.ece", "Expected calibration error", MetricRole.SECONDARY, direction=MetricDirection.LOWER_IS_BETTER), expected_calibration_error),
        (_definition("cell_annotation.hierarchical_f1", "Hierarchical F1", MetricRole.SECONDARY), hierarchical_f1),
        (_definition("cell_annotation.per_class_precision", "Per-class precision", MetricRole.DIAGNOSTIC), per_class_precision),
        (_definition("cell_annotation.per_class_recall", "Per-class recall", MetricRole.DIAGNOSTIC), per_class_recall),
        (_definition("cell_annotation.confusion_matrix", "Confusion matrix", MetricRole.DIAGNOSTIC), confusion_matrix_data),
        (_definition("cell_annotation.coverage", "Coverage", MetricRole.DIAGNOSTIC), coverage),
    ]


__all__ = ["annotation_definitions"]
