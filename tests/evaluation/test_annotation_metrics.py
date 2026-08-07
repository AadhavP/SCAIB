"""Annotation metric catalog tests."""

from agent_evals.evaluation.metrics import metric_registry


def test_annotation_catalog_contains_primary_and_calibration_metrics() -> None:
    metrics = metric_registry.list(category="cell_annotation")
    names = {metric.name for metric in metrics}
    assert {"cell_annotation.macro_f1", "cell_annotation.mcc", "cell_annotation.rare_recall"} <= names
    assert "cell_annotation.ece" in names
