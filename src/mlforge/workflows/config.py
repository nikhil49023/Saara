from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_workflow_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        value = _load_yaml(text)
    else:
        raise ValueError("Workflow config must be .json, .yaml, or .yml")
    if not isinstance(value, dict):
        raise ValueError("Workflow config must be an object")
    return value


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML or use JSON workflow configs") from exc
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("YAML workflow config must be an object")
    return value
