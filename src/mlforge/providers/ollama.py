from __future__ import annotations

import json
from time import perf_counter
from typing import Any
from urllib import error, request as urlrequest

from .base import ModelRequest, ModelResponse


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str = "qwen", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate_text(self, request: ModelRequest) -> ModelResponse:
        start = perf_counter()
        model = request.model or self.model
        prompt = request.prompt
        if request.system:
            prompt = f"System:\n{request.system}\n\nUser:\n{request.prompt}"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens
        raw = self._post("/api/generate", payload)
        return ModelResponse(
            text=str(raw.get("response", "")),
            provider=self.name,
            model=model,
            latency_ms=int((perf_counter() - start) * 1000),
            raw=raw,
            finish_reason="done" if raw.get("done") else None,
        )

    def generate_json(self, request: ModelRequest, schema: dict[str, Any]) -> ModelResponse:
        json_request = ModelRequest(
            prompt=request.prompt,
            system=(request.system or "")
            + "\nReturn only a single valid JSON object matching the requested schema.",
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            metadata=request.metadata,
        )
        return self.generate_text(json_request)

    def health(self) -> dict[str, Any]:
        try:
            raw = self._get("/api/tags")
        except RuntimeError as exc:
            return {"provider": self.name, "ok": False, "error": str(exc)}
        models = [item.get("name") for item in raw.get("models", [])]
        return {"provider": self.name, "ok": True, "base_url": self.base_url, "models": models}

    def _get(self, path: str) -> dict[str, Any]:
        try:
            with urlrequest.urlopen(f"{self.base_url}{path}", timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
