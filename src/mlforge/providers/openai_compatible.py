from __future__ import annotations

import json
from time import perf_counter
from typing import Any
from urllib import error, request as urlrequest

from .base import ModelRequest, ModelResponse


class OpenAICompatibleProvider:
    name = "vllm"

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate_text(self, request: ModelRequest) -> ModelResponse:
        start = perf_counter()
        model = request.model or self.model
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        raw = self._post("/chat/completions", payload)
        choice = raw.get("choices", [{}])[0]
        message = choice.get("message", {})
        return ModelResponse(
            text=str(message.get("content", "")),
            provider=self.name,
            model=model,
            latency_ms=int((perf_counter() - start) * 1000),
            raw=raw,
            finish_reason=choice.get("finish_reason"),
        )

    def generate_json(self, request: ModelRequest, schema: dict[str, Any]) -> ModelResponse:
        json_request = ModelRequest(
            prompt=request.prompt,
            system=(request.system or "")
            + "\nReturn only one valid JSON object. Do not include markdown fences.",
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            metadata=request.metadata,
        )
        return self.generate_text(json_request)

    def health(self) -> dict[str, Any]:
        try:
            raw = self._get("/models")
        except RuntimeError as exc:
            return {"provider": self.name, "ok": False, "error": str(exc)}
        return {"provider": self.name, "ok": True, "base_url": self.base_url, "raw": raw}

    def _get(self, path: str) -> dict[str, Any]:
        req = urlrequest.Request(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with urlrequest.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
