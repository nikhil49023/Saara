from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlforge.datasets.exporters import export_examples
from mlforge.datasets.io import load_rows


@dataclass(slots=True)
class DistillWorkflow:
    input_path: Path
    output_path: Path
    provider: Any
    method: str = "sft"
    output_format: str = "jsonl"

    def run(self) -> dict[str, Any]:
        rows = load_rows(self.input_path)
        if self.method == "sft":
            distilled = [self._to_sft(row, index) for index, row in enumerate(rows, start=1)]
        elif self.method == "dpo":
            distilled = [self._to_dpo(row, index) for index, row in enumerate(rows, start=1)]
        else:
            raise ValueError(f"Unsupported distillation method: {self.method}")
        export_examples(distilled, self.output_path, self.output_format)
        return {
            "input_path": str(self.input_path),
            "dataset_path": str(self.output_path),
            "format": self.output_format,
            "examples": len(distilled),
            "method": self.method,
            "teacher_provider": getattr(self.provider, "name", "unknown"),
        }

    def _to_sft(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        messages = _messages(row)
        return {
            "id": row.get("id") or f"sft_{index:06d}",
            "task_type": "sft",
            "messages": messages,
            "sources": row.get("sources", []),
            "metadata": {
                **_metadata(row),
                "distillation_method": "sft",
                "teacher_provider": getattr(self.provider, "name", "unknown"),
            },
        }

    def _to_dpo(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        messages = _messages(row)
        prompt = _first_user(messages)
        chosen = _last_assistant(messages)
        rejected = "I do not have enough grounded information to answer this well."
        return {
            "id": row.get("id") or f"dpo_{index:06d}",
            "task_type": "preference_dpo",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "sources": row.get("sources", []),
            "metadata": {
                **_metadata(row),
                "distillation_method": "dpo",
                "teacher_provider": getattr(self.provider, "name", "unknown"),
            },
        }


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return [message for message in messages if isinstance(message, dict)]
    instruction = ""
    input_payload = row.get("input")
    if isinstance(input_payload, dict):
        instruction = str(input_payload.get("instruction") or input_payload.get("input") or "")
    output_payload = row.get("output")
    output = ""
    if isinstance(output_payload, dict):
        output = str(output_payload.get("output") or output_payload)
    return [
        {"role": "user", "content": instruction or "Complete the task."},
        {"role": "assistant", "content": output or ""},
    ]


def _first_user(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return str(messages[0].get("content") if messages else "")


def _last_assistant(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return str(messages[-1].get("content") if messages else "")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}
