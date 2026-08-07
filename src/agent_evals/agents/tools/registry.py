"""Tool registry and execution boundary."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from agent_evals.agents.tools.base import ToolDefinition, ToolHandler


class ToolRegistry:
    """Discoverable tool definitions and handlers."""

    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(
        self,
        definition: ToolDefinition,
        handler: ToolHandler | None = None,
    ) -> ToolDefinition:
        """Register one definition and optional executor."""
        if definition.name in self._definitions:
            raise ValueError(f"tool '{definition.name}' is already registered")
        self._definitions[definition.name] = definition
        if handler is not None:
            self._handlers[definition.name] = handler
        return definition

    def decorator(
        self,
        name: str,
        *,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Register a handler while preserving its callable identity."""

        def register_handler(handler: ToolHandler) -> ToolHandler:
            self.register(
                ToolDefinition(name=name, description=description, parameters=parameters or {}),
                handler,
            )
            return handler

        return register_handler

    def get(self, name: str) -> ToolDefinition:
        """Return a public definition."""
        try:
            return self._definitions[name]
        except KeyError as error:
            raise KeyError(f"tool '{name}' is not registered") from error

    def handler(self, name: str) -> ToolHandler:
        """Return the executable handler."""
        try:
            return self._handlers[name]
        except KeyError as error:
            raise KeyError(f"tool '{name}' has no executor") from error

    def list(self) -> list[ToolDefinition]:
        """Return definitions in stable order."""
        return [self._definitions[name] for name in sorted(self._definitions)]

    def clear(self) -> None:
        """Clear registrations for isolated test suites."""
        self._definitions.clear()
        self._handlers.clear()


class ToolExecutor:
    """Execute only registered handlers and retain a complete call log."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.call_log: list[dict[str, Any]] = []

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        context: Any = None,
    ) -> Any:
        """Execute one registered tool and append a replayable log entry."""
        definition = self.registry.get(name)
        del definition
        handler = self.registry.handler(name)
        args = arguments or {}
        try:
            value = handler(args, context)
            if inspect.isawaitable(value):
                value = await value
        except Exception as error:
            record = {"tool": name, "arguments": args, "error": str(error)}
            self.call_log.append(record)
            raise
        self.call_log.append({"tool": name, "arguments": args, "result": value})
        return value


tool_registry = ToolRegistry()

__all__ = ["ToolExecutor", "ToolRegistry", "tool_registry"]
