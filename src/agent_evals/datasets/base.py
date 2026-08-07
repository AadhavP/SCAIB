"""Abstract dataset interface for single-cell biology datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Metadata describing a single-cell dataset."""

    id: str
    name: str
    organism: str
    cell_count: int | None = None
    gene_count: int | None = None
    file_format: str = "h5ad"
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseDataset(ABC):
    """Abstract Base Class representing a single-cell benchmark dataset."""

    def __init__(self, metadata: DatasetMetadata, local_path: Path) -> None:
        self.metadata = metadata
        self.local_path = local_path

    @abstractmethod
    async def download(self) -> None:
        """Download dataset files from remote source if not present locally."""
        pass

    @abstractmethod
    async def load(self) -> Any:
        """Load and return dataset object (e.g. AnnData or dictionary metadata)."""
        pass
