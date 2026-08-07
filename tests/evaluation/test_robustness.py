"""Multi-seed robustness tests."""

from agent_evals.evaluation.metrics.robustness import RobustnessEvaluator


def test_robustness_is_deterministic_and_pairwise() -> None:
    report = RobustnessEvaluator().evaluate(
        [
            {"seed": 1, "cluster_labels": [0, 0, 1], "predicted_labels": ["a", "a", "b"], "artifact_checksums": ["x"]},
            {"seed": 2, "cluster_labels": [0, 0, 1], "predicted_labels": ["a", "a", "b"], "artifact_checksums": ["x"]},
        ]
    )

    assert report.seeds == [1, 2]
    assert report.seed_stability == 1.0
    assert report.clustering_pairwise_ari == 1.0
