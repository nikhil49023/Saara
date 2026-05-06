from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def export_examples(
    rows: Iterable[Mapping[str, Any]],
    output_path: Path | str,
    fmt: str,
) -> Path:
    fmt = fmt.lower()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    if fmt == "jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for row in materialized:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return output_path
    if fmt == "json":
        output_path.write_text(
            json.dumps({"examples": materialized}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output_path
    if fmt in {"csv", "tsv"}:
        delimiter = "\t" if fmt == "tsv" else ","
        flat_rows = [_flatten(row) for row in materialized]
        fieldnames = sorted({key for row in flat_rows for key in row})
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(flat_rows)
        return output_path
    if fmt in {"parquet", "arrow"}:
        return _export_arrow_like(materialized, output_path, fmt)
    if fmt == "hf":
        return _export_hf(materialized, output_path)
    raise ValueError(f"Unsupported export format: {fmt}")


def _flatten(row: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
        else:
            flat[key] = json.dumps(value, ensure_ascii=False)
    return flat


def _export_arrow_like(rows: list[dict[str, Any]], output_path: Path, fmt: str) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install optional data dependencies: pip install 'saara[data]'") from exc
    table = pa.Table.from_pylist(rows)
    if fmt == "parquet":
        pq.write_table(table, output_path)
    else:
        with pa.OSFile(str(output_path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write(table)
    return output_path


def _export_hf(rows: list[dict[str, Any]], output_path: Path) -> Path:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Install optional data dependencies: pip install 'saara[data]'") from exc
    dataset = Dataset.from_list(rows)
    dataset.save_to_disk(str(output_path))
    return output_path
