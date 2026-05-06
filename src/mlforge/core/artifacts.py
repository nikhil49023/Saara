from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Any

from .hashing import short_hash
from .schemas import WorkflowRun, utc_now


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "workflow"


class ArtifactStore:
    def __init__(self, root: Path | str = ".mlforge") -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.datasets_dir = self.root / "datasets"
        self.cache_dir = self.root / "cache"
        self.reports_dir = self.root / "reports"

    def init(self) -> None:
        for path in (self.runs_dir, self.datasets_dir, self.cache_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)

    def create_run(self, name: str, config: Mapping[str, Any]) -> WorkflowRun:
        self.init()
        run_id = f"{slugify(name)}-{short_hash({'name': name, 'config': config, 'time': utc_now()})}"
        run_dir = self.runs_dir / run_id
        for child in ("artifacts", "raw", "logs"):
            (run_dir / child).mkdir(parents=True, exist_ok=True)
        run = WorkflowRun(run_id=run_id, name=name, run_dir=str(run_dir), config=dict(config))
        self.write_json(run_dir / "run.json", run.to_dict())
        return run

    def update_run(self, run: WorkflowRun) -> None:
        run.updated_at = utc_now()
        self.write_json(Path(run.run_dir) / "run.json", run.to_dict())

    @staticmethod
    def write_json(path: Path | str, payload: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        return count

    @staticmethod
    def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid JSONL at line {line_number}: expected object")
                rows.append(value)
        return rows
