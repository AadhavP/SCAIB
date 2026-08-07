"""Schema-version migration seam for future benchmark definitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_evals.benchmarks.schema import CURRENT_SCHEMA_VERSION

Migration = Callable[[dict[str, Any]], dict[str, Any]]


class SchemaMigrationRegistry:
    """Registry of pure payload migrations between schema versions.

    No migration is currently required for schema ``1.0.0``.  Keeping the
    migration seam at the serialization boundary means future schema changes
    can be handled without putting compatibility logic into scientific task
    definitions or execution code.
    """

    def __init__(self) -> None:
        self._migrations: dict[tuple[str, str], Migration] = {}

    def register(self, from_version: str, to_version: str) -> Callable[[Migration], Migration]:
        """Register a pure migration function for one version transition."""

        def decorator(function: Migration) -> Migration:
            key = (from_version, to_version)
            if key in self._migrations:
                raise ValueError(f"migration {from_version} -> {to_version} already exists")
            self._migrations[key] = function
            return function

        return decorator

    def migrate(
        self,
        payload: dict[str, Any],
        *,
        target_version: str = CURRENT_SCHEMA_VERSION,
    ) -> dict[str, Any]:
        """Apply registered one-step migrations until the target is reached."""
        current = str(payload.get("schema_version", "1.0.0"))
        migrated = dict(payload)
        if current == target_version:
            return migrated
        seen: set[str] = set()
        while current != target_version:
            if current in seen:
                raise ValueError(f"schema migration cycle detected at {current}")
            seen.add(current)
            candidates = [
                (to_version, function)
                for (from_version, to_version), function in self._migrations.items()
                if from_version == current
            ]
            if not candidates:
                raise ValueError(
                    f"no schema migration registered from {current} to {target_version}"
                )
            if len(candidates) > 1:
                raise ValueError(f"ambiguous schema migrations from {current}")
            next_version, function = candidates[0]
            migrated = function(dict(migrated))
            migrated["schema_version"] = next_version
            current = next_version
        return migrated


schema_migrations = SchemaMigrationRegistry()
"""Process-wide migration registry used by loaders that opt into migration."""

