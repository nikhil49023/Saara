from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from mlforge.core.hashing import content_hash
from mlforge.core.schemas import ValidationIssue, ValidationReport

from .io import load_rows


def validate_examples(rows: list[dict[str, Any]], dataset_path: str) -> ValidationReport:
    issues: list[ValidationIssue] = []
    seen: dict[str, str] = {}
    duplicate_count = 0
    labels: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        example_id = str(row.get("id") or f"row_{index}")
        if "labels" in row and isinstance(row["labels"], dict):
            for key, value in row["labels"].items():
                labels[f"{key}:{value}"] += 1
        if not row.get("id"):
            issues.append(ValidationIssue("missing_id", "Example is missing id", example_id))
        if not row.get("task_type"):
            issues.append(ValidationIssue("missing_task_type", "Example is missing task_type", example_id))
        has_messages = isinstance(row.get("messages"), list) and bool(row.get("messages"))
        has_output = bool(row.get("output")) or bool(row.get("labels"))
        if not has_messages and not has_output:
            issues.append(
                ValidationIssue(
                    "missing_payload",
                    "Example needs messages, output, or labels",
                    example_id,
                )
            )
        normalized = dict(row)
        normalized.pop("id", None)
        row_hash = content_hash(normalized)
        if row_hash in seen:
            duplicate_count += 1
            issues.append(
                ValidationIssue(
                    "duplicate",
                    f"Duplicate of {seen[row_hash]}",
                    example_id,
                    severity="warning",
                )
            )
        else:
            seen[row_hash] = example_id

    invalid_ids = {issue.example_id for issue in issues if issue.severity == "error"}
    invalid_count = len(invalid_ids)
    return ValidationReport(
        dataset_path=dataset_path,
        total_examples=len(rows),
        valid_examples=max(0, len(rows) - invalid_count),
        invalid_examples=invalid_count,
        duplicate_examples=duplicate_count,
        issues=issues,
        label_distribution=dict(labels),
    )


def validate_dataset(path: Path | str) -> ValidationReport:
    rows = load_rows(path)
    return validate_examples(rows, str(path))
