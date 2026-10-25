"""Dataset export engine: schema conversion + deterministic splits + sharding + compression + manifest.

Layout written under `out_dir`:
  <split>/<name>-00000-of-00002.<ext>[.gz]   data shards (one directory per split)
  export_manifest.json                        schema, format, counts, drops, sha256 per file
  README.md                                   Hugging Face dataset card (configs point at the shards)
  dataset_info.json                           LLaMA-Factory registration (sharegpt/alpaca only)
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from mlforge.core.schemas import utc_now
from mlforge.datasets.exporters import export_examples
from mlforge.datasets.schemas import SCHEMAS, SchemaError

STORAGE = {"jsonl": "jsonl", "json": "json", "csv": "csv", "tsv": "tsv", "parquet": "parquet", "arrow": "arrow", "hf": ""}
COMPRESSIBLE = {"jsonl", "json", "csv", "tsv"}


@dataclass(slots=True)
class ExportSpec:
    fmt: str = "jsonl"
    schema: str = "saara"
    splits: dict[str, float] = field(default_factory=lambda: {"train": 1.0})
    group_by: str | None = None
    shard_size: int | None = None
    compress: str | None = None
    name: str = "data"
    seed: int = 0
    strict: bool = False

    def __post_init__(self) -> None:
        if self.fmt not in STORAGE:
            raise ValueError(f"Unsupported format {self.fmt!r}; choose from {sorted(STORAGE)}")
        if self.schema not in SCHEMAS:
            raise ValueError(f"Unsupported schema {self.schema!r}; choose from {sorted(SCHEMAS)}")
        if self.compress not in (None, "gzip"):
            raise ValueError("compress must be 'gzip' or omitted")
        if self.compress and self.fmt not in COMPRESSIBLE:
            raise ValueError(f"compression applies to {sorted(COMPRESSIBLE)} only")
        total = sum(self.splits.values())
        if not self.splits or any(v < 0 for v in self.splits.values()) or abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must be non-negative and sum to 1 (got {self.splits})")
        if self.shard_size is not None and self.shard_size < 1:
            raise ValueError("shard_size must be >= 1")


def parse_splits(value: str) -> dict[str, float]:
    """'train=0.9,val=0.05,test=0.05' -> {'train': 0.9, ...}"""
    splits = {}
    for part in value.split(","):
        key, _, fraction = part.partition("=")
        if not key.strip() or not fraction:
            raise ValueError(f"bad split spec {part!r}; expected name=fraction")
        splits[key.strip()] = float(fraction)
    return splits


def assign_split(key: str, splits: Mapping[str, float], seed: int) -> str:
    """Deterministic: the same key always lands in the same split (stable across runs and machines)."""
    bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / float(16**12)
    edge = 0.0
    for name, fraction in splits.items():
        edge += fraction
        if bucket < edge:
            return name
    return list(splits)[-1]


def _row_key(row: Mapping[str, Any], group_by: str | None, index: int) -> str:
    if group_by:
        value: Any = row
        for part in group_by.split("."):
            value = value.get(part) if isinstance(value, Mapping) else None
        if value is not None:
            return str(value)
    return str(row.get("id") or index)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in ([path] if path.is_file() else path.rglob("*")) if p.is_file()):
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _write_shard(records: list[dict[str, Any]], path: Path, spec: ExportSpec) -> Path:
    if spec.fmt == "hf":
        return export_examples(records, path, "hf")
    export_examples(records, path, spec.fmt)
    if spec.compress == "gzip":
        gz_path = path.with_name(path.name + ".gz")
        with path.open("rb") as source, gzip.open(gz_path, "wb", compresslevel=6) as target:
            target.write(source.read())
        path.unlink()
        return gz_path
    return path


def export_dataset(rows: Iterable[Mapping[str, Any]], out_dir: Path | str, spec: ExportSpec) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    convert = SCHEMAS[spec.schema]
    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in spec.splits}
    drops: Counter[str] = Counter()
    groups: dict[str, set[str]] = {name: set() for name in spec.splits}
    total = 0
    for index, row in enumerate(rows):
        total += 1
        try:
            record = convert(dict(row))
        except (SchemaError, KeyError, TypeError) as exc:
            reason = str(exc) if isinstance(exc, SchemaError) else f"{type(exc).__name__}:{exc}"
            if spec.strict:
                raise ValueError(f"row {row.get('id', index)} cannot be exported as {spec.schema}: {reason}") from exc
            drops[reason] += 1
            continue
        key = _row_key(row, spec.group_by, index)
        split = assign_split(key, spec.splits, spec.seed)
        groups[split].add(key)
        by_split[split].append(record)

    leaked = sorted(set.intersection(*groups.values())) if len(groups) > 1 else []
    files: list[dict[str, Any]] = []
    ext = STORAGE[spec.fmt]
    for split, records in by_split.items():
        if not records:
            continue
        size = spec.shard_size or len(records)
        shards = [records[i : i + size] for i in range(0, len(records), size)]
        for number, shard in enumerate(shards):
            stem = f"{spec.name}-{number:05d}-of-{len(shards):05d}"
            path = out_dir / split / (f"{stem}.{ext}" if ext else stem)
            written = _write_shard(shard, path, spec)
            files.append({"split": split, "path": str(written.relative_to(out_dir)), "rows": len(shard),
                          "sha256": _sha256(written)})

    manifest = {
        "name": spec.name, "schema": spec.schema, "format": spec.fmt, "compress": spec.compress,
        "created_at": utc_now(), "seed": spec.seed, "group_by": spec.group_by, "splits": spec.splits,
        "rows_in": total, "rows_out": sum(f["rows"] for f in files),
        "counts": {split: len(records) for split, records in by_split.items()},
        "dropped": dict(drops), "group_leakage": len(leaked), "files": files,
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_card(out_dir, manifest)
    if spec.schema in ("sharegpt", "alpaca") and spec.fmt in ("jsonl", "json") and not spec.compress:
        _write_llamafactory_info(out_dir, manifest)
    return manifest


def _write_card(out_dir: Path, manifest: dict[str, Any]) -> None:
    if manifest["format"] == "hf":
        return
    configs = "\n".join(
        f"  - split: {split}\n    path: \"{split}/*\"" for split in manifest["counts"] if manifest["counts"][split]
    )
    counts = "\n".join(f"| {split} | {n} |" for split, n in manifest["counts"].items())
    drops = "\n".join(f"| {reason} | {n} |" for reason, n in manifest["dropped"].items()) or "| none | 0 |"
    card = f"""---
configs:
- config_name: default
  data_files:
{configs}
tags: [saara, {manifest['schema']}]
---

# {manifest['name']}

Exported by [Saara](https://github.com/nikhil49023/Saara) in the **{manifest['schema']}** schema as **{manifest['format']}**
({manifest['created_at']}). Splits are deterministic (seed {manifest['seed']}, grouped by `{manifest['group_by'] or 'id'}`;
groups shared across splits: {manifest['group_leakage']}).

| Split | Rows |
|---|---|
{counts}

Rows not exportable in this schema (reason → count):

| Reason | Rows |
|---|---|
{drops}

File checksums are in `export_manifest.json`.
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def _write_llamafactory_info(out_dir: Path, manifest: dict[str, Any]) -> None:
    info = {}
    for split in manifest["counts"]:
        paths = [f["path"] for f in manifest["files"] if f["split"] == split]
        if not paths:
            continue
        entry: dict[str, Any] = {"file_name": paths[0] if len(paths) == 1 else split, "formatting": manifest["schema"]}
        if manifest["schema"] == "sharegpt":
            entry["columns"] = {"messages": "conversations", "system": "system", "tools": "tools", "images": "images"}
        else:
            entry["columns"] = {"prompt": "instruction", "query": "input", "response": "output", "system": "system",
                                "history": "history"}
        info[f"{manifest['name']}_{split}"] = entry
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
