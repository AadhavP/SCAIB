"""Deterministic Scanpy-default baseline plan."""

from agent_evals.baselines.base import BaselineResult, BaselineRunner


class ScanpyDefaultBaseline(BaselineRunner):
    """Reference sequence matching the standard Scanpy PBMC workflow."""

    baseline_id = "scanpy_default"

    def run(self, context: dict[str, object] | None = None, seed: int = 0) -> BaselineResult:
        """Return the declared plan; execution remains owned by the environment."""
        del context
        return BaselineResult(
            baseline_id=self.baseline_id,
            status="planned",
            actions=["qc", "normalize", "select_hvg", "pca", "cluster", "annotate"],
            metadata={"seed": seed, "backend": "scanpy"},
        )


__all__ = ["ScanpyDefaultBaseline"]
