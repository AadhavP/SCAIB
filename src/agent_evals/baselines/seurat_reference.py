"""Optional Seurat reference baseline boundary."""

from agent_evals.baselines.base import BaselineResult, BaselineRunner


class SeuratReferenceBaseline(BaselineRunner):
    """Report availability without requiring an R runtime in core evaluation."""

    baseline_id = "seurat_reference"

    def run(self, context: dict[str, object] | None = None, seed: int = 0) -> BaselineResult:
        """Return an explicit unavailable result when Seurat is not configured."""
        del context
        return BaselineResult(
            baseline_id=self.baseline_id,
            status="unavailable",
            seed=seed,
            metadata={"reason": "Seurat/R adapter is optional"},
        )


__all__ = ["SeuratReferenceBaseline"]
