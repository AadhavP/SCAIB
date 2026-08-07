"""Discoverable registry for provider-neutral agent runtimes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_evals.agents.runtime.base import AgentRuntime
from agent_evals.agents.runtime.protocol import AgentManifest

RuntimeFactory = Callable[..., AgentRuntime]


class AgentRuntimeRegistry:
    """Register runtime factories without importing provider SDKs eagerly."""

    def __init__(self) -> None:
        self._factories: dict[str, RuntimeFactory] = {}
        self._manifests: dict[str, AgentManifest] = {}
        self._config_schemas: dict[str, Any | None] = {}

    def register(
        self,
        name: str,
        backend: RuntimeFactory | None = None,
        *,
        capabilities: list[str] | None = None,
        manifest: AgentManifest | None = None,
        config_schema: Any | None = None,
    ) -> Callable[[RuntimeFactory], RuntimeFactory] | RuntimeFactory:
        """Register a factory directly or return a decorator."""

        def decorator(factory: RuntimeFactory) -> RuntimeFactory:
            if name in self._factories:
                raise ValueError(f"agent runtime '{name}' is already registered")
            self._factories[name] = factory
            self._manifests[name] = manifest or AgentManifest(
                name=name,
                type=name,
                capabilities=capabilities or [],
            )
            self._config_schemas[name] = config_schema
            return factory

        if backend is None:
            return decorator
        return decorator(backend)

    def get(self, name: str) -> RuntimeFactory:
        """Return a registered runtime factory."""
        try:
            return self._factories[name]
        except KeyError as error:
            raise KeyError(f"agent runtime '{name}' is not registered") from error

    def create(self, name: str, **config: Any) -> AgentRuntime:
        """Instantiate a runtime with provider-specific configuration."""
        runtime = self.get(name)(**config)
        if not isinstance(runtime, AgentRuntime):
            raise TypeError(f"runtime factory '{name}' did not return an AgentRuntime")
        return runtime

    def manifest(self, name: str) -> AgentManifest:
        """Return public registration metadata."""
        return self._manifests[name].model_copy(deep=True)

    def config_schema(self, name: str) -> Any | None:
        """Return the optional provider configuration schema."""
        return self._config_schemas[name]

    def list(self) -> list[str]:
        """List runtimes in stable order."""
        return sorted(self._factories)

    def clear(self) -> None:
        """Clear registrations for isolated test suites."""
        self._factories.clear()
        self._manifests.clear()
        self._config_schemas.clear()


agent_runtime_registry = AgentRuntimeRegistry()

__all__ = ["AgentRuntimeRegistry", "RuntimeFactory", "agent_runtime_registry"]
