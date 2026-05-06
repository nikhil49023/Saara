from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlforge.core.hashing import short_hash
from mlforge.datasets.exporters import export_examples
from mlforge.datasets.io import load_rows
from mlforge.providers.base import ModelRequest


LABEL_SYSTEM_PROMPT = "You label dataset examples. Return only valid JSON."
LABEL_PROMPT_TEMPLATE = """Classify this dataset example.
Allowed labels: {labels}
Label field: {label_field}

Example:
{example}

Return JSON with string field label and numeric field confidence."""


@dataclass(slots=True)
class LabelDatasetWorkflow:
    input_path: Path
    output_path: Path
    provider: Any
    labels: list[str]
    label_field: str = "label"
    output_format: str = "jsonl"
    system_prompt: str | None = LABEL_SYSTEM_PROMPT
    prompt_template: str | None = LABEL_PROMPT_TEMPLATE
    temperature: float = 0.0
    max_tokens: int | None = None

    def run(self) -> dict[str, Any]:
        rows = load_rows(self.input_path)
        labeled = [self._label_row(row, index) for index, row in enumerate(rows, start=1)]
        export_examples(labeled, self.output_path, self.output_format)
        return {
            "input_path": str(self.input_path),
            "dataset_path": str(self.output_path),
            "format": self.output_format,
            "examples": len(labeled),
            "labels": self.labels,
            "label_field": self.label_field,
            "provider": getattr(self.provider, "name", "unknown"),
        }

    def _label_row(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        result = dict(row)
        label, confidence = self._predict_label(row, index)
        existing_labels = result.get("labels") if isinstance(result.get("labels"), dict) else {}
        result["labels"] = {**existing_labels, self.label_field: label}
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        result["metadata"] = {
            **metadata,
            "label_provider": getattr(self.provider, "name", "unknown"),
            "label_confidence": confidence,
        }
        result.setdefault("id", f"labeled_{index:06d}")
        result.setdefault("task_type", "classification")
        return result

    def _predict_label(self, row: dict[str, Any], index: int) -> tuple[str, float]:
        if not self.labels:
            return "unlabeled", 0.0
        if getattr(self.provider, "name", "") == "mock":
            position = int(short_hash({"row": row, "index": index}), 16) % len(self.labels)
            return self.labels[position], 0.75

        prompt = (self.prompt_template or LABEL_PROMPT_TEMPLATE).format(
            labels=", ".join(self.labels),
            label_field=self.label_field,
            example=json.dumps(row, ensure_ascii=False),
        )
        response = self.provider.generate_json(
            ModelRequest(
                prompt=prompt,
                system=self.system_prompt or LABEL_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                metadata={"index": index},
            ),
            schema={
                "type": "object",
                "required": ["label", "confidence"],
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        )
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError:
            return self.labels[0], 0.0
        label = str(payload.get("label") or self.labels[0])
        if label not in self.labels:
            label = self.labels[0]
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return label, max(0.0, min(1.0, confidence))
