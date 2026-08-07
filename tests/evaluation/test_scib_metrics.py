"""Optional scIB-metrics boundary tests."""

from agent_evals.evaluation.metrics import metric_registry


def test_scib_backed_metric_catalog_is_available_without_forcing_optional_imports() -> None:
    names = {metric.name for metric in metric_registry.list(category="batch_integration")}
    assert "batch_integration.iLISI" in names
    assert "batch_integration.kBET" in names
