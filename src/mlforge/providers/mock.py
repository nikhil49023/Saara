from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from .base import ModelRequest, ModelResponse


class MockProvider:
    name = "mock"

    def __init__(self, model: str = "mock-local") -> None:
        self.model = model

    def generate_text(self, request: ModelRequest) -> ModelResponse:
        start = perf_counter()
        topic = request.metadata.get("topic", "the requested topic")
        index = request.metadata.get("index", 1)
        payload = {
            "instruction": f"Explain a core idea about {topic}.",
            "input": f"Focus on example {index}.",
            "output": f"{topic} example {index}: this is a deterministic development sample.",
        }
        text = json.dumps(payload)
        return ModelResponse(
            text=text,
            provider=self.name,
            model=request.model or self.model,
            latency_ms=int((perf_counter() - start) * 1000),
            raw={"mock": True},
        )

    def generate_json(self, request: ModelRequest, schema: dict[str, Any]) -> ModelResponse:
        return self.generate_text(request)

    def health(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "ok": True}
