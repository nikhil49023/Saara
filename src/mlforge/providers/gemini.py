from __future__ import annotations

import base64
import os
from time import perf_counter
from typing import Any

from .base import ModelRequest, ModelResponse

DEFAULT_MODEL = "gemma-4-26b-a4b-it"


class GeminiProvider:
    """Hosted Gemma 4 / Gemini models through the official `google-genai` SDK (Gemini API).

    API key: `--api-key`, or the GEMINI_API_KEY / GOOGLE_API_KEY environment variables.
    Images from `request.metadata["images"]` (base64) are sent as inline parts, so PDF vision OCR works.
    Set `request.metadata["thinking_level"]` to "high" or "minimal" to control Gemma 4 thinking.
    """

    name = "gemini"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, client: Any | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("Install the Gemini provider: pip install 'saara-ai[gemini]'") from exc
            if not self.api_key:
                raise RuntimeError("Set GEMINI_API_KEY (or pass --api-key) to use --provider gemini")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_text(self, request: ModelRequest) -> ModelResponse:
        return self._generate(request, json_mode=False)

    def generate_json(self, request: ModelRequest, schema: dict[str, Any]) -> ModelResponse:
        json_request = ModelRequest(
            prompt=request.prompt,
            system=(request.system or "") + "\nReturn only one valid JSON object. Do not include markdown fences.",
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            metadata=request.metadata,
        )
        return self._generate(json_request, json_mode=True)

    def health(self) -> dict[str, Any]:
        try:
            info = self.client.models.get(model=self.model)
        except Exception as exc:  # noqa: BLE001 - report any SDK/auth/network failure as unhealthy
            return {"provider": self.name, "ok": False, "model": self.model, "error": str(exc)}
        return {"provider": self.name, "ok": True, "model": self.model, "display_name": getattr(info, "display_name", None)}

    def _generate(self, request: ModelRequest, json_mode: bool) -> ModelResponse:
        from google.genai import types

        start = perf_counter()
        model = request.model or self.model
        parts: list[Any] = []
        images = request.metadata.get("images") or []
        mime = f"image/{request.metadata.get('image_format', 'png')}"
        for image in images:  # image before text, as recommended for Gemma 4
            data = base64.b64decode(image) if isinstance(image, str) else image
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        parts.append(request.prompt)
        config: dict[str, Any] = {"temperature": request.temperature}
        if request.system:
            config["system_instruction"] = request.system
        if request.max_tokens:
            config["max_output_tokens"] = request.max_tokens
        if json_mode and model.startswith("gemini"):
            config["response_mime_type"] = "application/json"
        level = request.metadata.get("thinking_level")
        if level:
            config["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        try:
            response = self.client.models.generate_content(
                model=model, contents=parts, config=types.GenerateContentConfig(**config)
            )
        except Exception as exc:  # noqa: BLE001 - surface API errors as an invalid response
            return ModelResponse(text="", provider=self.name, model=model, latency_ms=int((perf_counter() - start) * 1000),
                                 valid=False, error=str(exc))
        candidates = getattr(response, "candidates", None) or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        return ModelResponse(
            text=response.text or "",
            provider=self.name,
            model=model,
            latency_ms=int((perf_counter() - start) * 1000),
            finish_reason=str(finish) if finish is not None else None,
        )
