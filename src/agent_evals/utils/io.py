"""Filesystem I/O and YAML/JSON helpers."""

import json
from pathlib import Path
from typing import Any, cast

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Safely load YAML file into dictionary."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_yaml(data: dict[str, Any], path: Path) -> None:
    """Save dictionary into formatted YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file into dictionary."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return cast(dict[str, Any], data) if isinstance(data, dict) else {}


def save_json(data: dict[str, Any], path: Path, indent: int = 2) -> None:
    """Save dictionary into formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
