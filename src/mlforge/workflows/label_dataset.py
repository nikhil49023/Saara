from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlforge.core.hashing import short_hash
from mlforge.datasets.exporters import export_examples
from mlforge.datasets.io import load_rows
from mlforge.providers.base import ModelRequest


@dataclass(slots=True)
class LabelDatasetWorkflow:
    input_path: Path
    output_path: Path
    provider: Any
    labels: list[str]
    label_field: str = "label"
    output_format: str = "jsonl"

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

        prompt = (
            "Classify this dataset example. Return JSON with string field label and numeric "
            f"field confidence. Allowed labels: {', '.join(self.labels)}.\n\n"
            f"Example:\n{json.dumps(row, ensure_ascii=False)}"
        )
        response = self.provider.generate_json(
            ModelRequest(prompt=prompt, temperature=0.0, metadata={"index": index}),
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
