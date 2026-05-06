from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Line {line_number} is not a JSON object")
                rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict) and isinstance(value.get("examples"), list):
            return [row for row in value["examples"] if isinstance(row, dict)]
        raise ValueError("JSON dataset must be a list or an object with an examples list")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported input format for {path}")
