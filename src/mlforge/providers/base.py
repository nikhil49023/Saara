from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ModelRequest:
    prompt: str
    system: str | None = None
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelResponse:
    text: str
    provider: str
    model: str
    latency_ms: int
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    valid: bool = True
    error: str | None = None


class ModelProvider(Protocol):
    name: str

    def generate_text(self, request: ModelRequest) -> ModelResponse:
        ...

    def generate_json(self, request: ModelRequest, schema: dict[str, Any]) -> ModelResponse:
        ...

    def health(self) -> dict[str, Any]:
        ...
