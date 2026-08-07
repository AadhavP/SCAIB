"""Registry for scientific execution backends."""

from __future__ import annotations

from collections.abc import Callable

from agent_evals.core.exceptions import RegistryError
from agent_evals.scientific.executor.base import ScientificExecutor


class ScientificExecutorRegistry:
    """Resolve an execution backend without coupling benchmarks to Scanpy."""

    def __init__(self) -> None:
        self._registry: dict[str, type[ScientificExecutor]] = {}

    def register(self, name: str) -> Callable[[type[ScientificExecutor]], type[ScientificExecutor]]:
        """Register an executor class under a stable name."""
        def decorator(cls: type[ScientificExecutor]) -> type[ScientificExecutor]:
            if name in self._registry:
                raise RegistryError(f"Scientific executor '{name}' is already registered.")
            self._registry[name] = cls
            return cls
        return decorator

    def create(self, name: str) -> ScientificExecutor:
        """Instantiate a registered executor."""
        try:
            return self._registry[name]()
        except KeyError as error:
            raise RegistryError(f"Scientific executor '{name}' is not registered.") from error

    def list_names(self) -> list[str]:
        """Return registered names."""
        return sorted(self._registry)


scientific_executor_registry = ScientificExecutorRegistry()

