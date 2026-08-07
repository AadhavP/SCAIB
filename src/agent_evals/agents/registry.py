"""Registry for discovering and managing agent adapters."""

from collections.abc import Callable

from agent_evals.agents.base import BaseAgentAdapter
from agent_evals.agents.harness import AgentAdapter
from agent_evals.core.exceptions import RegistryError


class AgentRegistry:
    """Registry to manage and resolve agent adapter implementations."""

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseAgentAdapter]] = {}

    def register(
        self, agent_type: str
    ) -> Callable[[type[BaseAgentAdapter]], type[BaseAgentAdapter]]:
        """Decorator to register an agent adapter class under a unique type key."""

        def decorator(cls: type[BaseAgentAdapter]) -> type[BaseAgentAdapter]:
            if agent_type in self._registry:
                raise RegistryError(
                    f"Agent adapter with type '{agent_type}' is already registered."
                )
            self._registry[agent_type] = cls
            return cls

        return decorator

    def get(self, agent_type: str) -> type[BaseAgentAdapter]:
        """Retrieve registered agent adapter class by type key."""
        if agent_type not in self._registry:
            raise RegistryError(
                f"Agent adapter '{agent_type}' not found in registry. "
                f"Available: {self.list_types()}"
            )
        return self._registry[agent_type]

    def list_types(self) -> list[str]:
        """Return list of registered agent adapter types."""
        return sorted(self._registry.keys())


# Global agent registry singleton instance
agent_registry = AgentRegistry()


class AgentAdapterRegistry:
    """Registry for framework-neutral harness adapters.

    Adapter classes are registered separately from the legacy raw-code agent
    registry so existing integrations remain compatible while new adapters
    return normalized :class:`AgentRun` objects.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[AgentAdapter]] = {}

    def register(
        self,
        agent_type: str,
    ) -> Callable[[type[AgentAdapter]], type[AgentAdapter]]:
        """Register an adapter class under a provider-neutral type."""

        def decorator(cls: type[AgentAdapter]) -> type[AgentAdapter]:
            if agent_type in self._registry:
                raise RegistryError(
                    f"Harness adapter with type '{agent_type}' is already registered."
                )
            self._registry[agent_type] = cls
            return cls

        return decorator

    def get(self, agent_type: str) -> type[AgentAdapter]:
        """Resolve an adapter class without importing optional backends."""
        if agent_type not in self._registry:
            raise RegistryError(
                f"Harness adapter '{agent_type}' not found in registry. "
                f"Available: {self.list_types()}"
            )
        return self._registry[agent_type]

    def create(self, agent_type: str) -> AgentAdapter:
        """Instantiate an adapter using its provider-neutral default constructor."""
        return self.get(agent_type)()

    def list_types(self) -> list[str]:
        """Return all registered harness adapter types."""
        return sorted(self._registry)

    def availability(self) -> dict[str, bool]:
        """Report optional backend availability without raising import errors."""
        result: dict[str, bool] = {}
        for agent_type, adapter_class in self._registry.items():
            try:
                adapter = adapter_class()
                result[agent_type] = bool(getattr(adapter, "available", True))
            except Exception:
                result[agent_type] = False
        return result


agent_adapter_registry = AgentAdapterRegistry()
