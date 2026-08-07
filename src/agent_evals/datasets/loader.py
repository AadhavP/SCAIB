"""Dataset loader & local caching manager."""

from pathlib import Path

from agent_evals.core.exceptions import DatasetNotFoundError
from agent_evals.datasets.base import BaseDataset


class DatasetLoader:
    """Dataset manager responsible for resolving dataset paths and caching."""

    def __init__(self, cache_dir: Path = Path("./.cache/datasets")) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, BaseDataset] = {}

    def register_dataset(self, dataset_id: str, dataset: BaseDataset) -> None:
        """Register a dataset provider."""
        self._registry[dataset_id] = dataset

    def get_dataset(self, dataset_id: str) -> BaseDataset:
        """Fetch dataset instance by ID."""
        if dataset_id not in self._registry:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' is not registered in dataset loader."
            )
        return self._registry[dataset_id]
