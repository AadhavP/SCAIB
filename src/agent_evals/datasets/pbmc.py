"""Public PBMC dataset loader used by the first scientific benchmark slice."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PBMCMetadata(BaseModel):
    """Observed metadata for the loaded AnnData object."""

    model_config = ConfigDict(extra="forbid")

    organism: str = "Homo sapiens"
    assay: str = "scRNA-seq"
    technology: str = "10x Genomics"
    cells: int
    genes: int
    metadata_columns: list[str] = Field(default_factory=list)
    source: str = "scanpy.datasets.pbmc68k_reduced"


class PBMCDataset:
    """Load, validate, and cache the public Scanpy PBMC reduced dataset.

    The reduced public object is intentionally used for the first vertical
    slice: it is a real PBMC AnnData dataset with public provenance while
    remaining practical for local benchmark smoke runs.
    """

    def __init__(
        self,
        cache_dir: Path | str = Path(".cache/datasets"),
        local_path: Path | str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_path = Path(local_path) if local_path is not None else self.cache_dir / "pbmc68k_reduced.h5ad"
        self._metadata: PBMCMetadata | None = None

    def load(self, *, max_cells: int | None = None) -> Any:
        """Load the cached/public AnnData object and optionally subset cells."""
        import anndata as ad
        import scanpy as sc

        if self.local_path.exists():
            adata = ad.read_h5ad(self.local_path)
        else:
            adata = sc.datasets.pbmc68k_reduced()
            adata.write_h5ad(self.local_path)
        self.validate_schema(adata)
        if max_cells is not None:
            if max_cells < 1:
                raise ValueError("max_cells must be positive")
            adata = adata[:max_cells].copy()
        self._metadata = PBMCMetadata(
            cells=int(adata.n_obs),
            genes=int(adata.n_vars),
            metadata_columns=[str(column) for column in adata.obs.columns],
        )
        return adata

    @staticmethod
    def validate_schema(adata: Any) -> None:
        """Validate the minimum AnnData contract required by the pipeline."""
        for attribute in ("n_obs", "n_vars", "obs", "var_names", "X"):
            if not hasattr(adata, attribute):
                raise ValueError(f"PBMC dataset is missing required AnnData attribute '{attribute}'")
        if int(adata.n_obs) == 0 or int(adata.n_vars) == 0:
            raise ValueError("PBMC dataset must contain at least one cell and one gene")
        if len(adata.obs.index) != int(adata.n_obs):
            raise ValueError("PBMC observation index does not match n_obs")
        if len(adata.var_names) != int(adata.n_vars):
            raise ValueError("PBMC gene index does not match n_vars")

    @property
    def metadata(self) -> PBMCMetadata:
        """Return metadata from the last successful load."""
        if self._metadata is None:
            raise RuntimeError("load() must be called before metadata is available")
        return self._metadata


__all__ = ["PBMCDataset", "PBMCMetadata"]
