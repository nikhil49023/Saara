from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")

from mlforge.providers.base import ModelRequest  # noqa: E402
from mlforge.providers.factory import create_provider  # noqa: E402
from mlforge.providers.gemini import GeminiProvider  # noqa: E402


class FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text='{"ok": true}', candidates=[SimpleNamespace(finish_reason="STOP")])

    def get(self, model):
        return SimpleNamespace(display_name=f"Gemma ({model})")


def provider() -> tuple[GeminiProvider, FakeModels]:
    models = FakeModels()
    return GeminiProvider(model="gemma-4-26b-a4b-it", api_key="test", client=SimpleNamespace(models=models)), models


def test_text_generation_passes_system_tokens_and_thinking() -> None:
    gem, models = provider()
    response = gem.generate_text(ModelRequest(prompt="hi", system="be brief", max_tokens=64,
                                              metadata={"thinking_level": "high"}))
    assert response.valid and response.text == '{"ok": true}' and response.finish_reason == "STOP"
    config = models.calls[0]["config"]
    assert config.system_instruction == "be brief" and config.max_output_tokens == 64
    assert config.thinking_config.thinking_level.value.lower() == "high"


def test_images_are_sent_before_text_for_vision_ocr() -> None:
    gem, models = provider()
    image = base64.b64encode(b"\x89PNG fake").decode()
    gem.generate_text(ModelRequest(prompt="Transcribe this page.", metadata={"images": [image], "image_format": "png"}))
    contents = models.calls[0]["contents"]
    assert contents[-1] == "Transcribe this page."
    assert contents[0].inline_data.mime_type == "image/png" and contents[0].inline_data.data == b"\x89PNG fake"


def test_json_mode_and_health() -> None:
    gem, models = provider()
    gem.generate_json(ModelRequest(prompt="give json"), schema={"type": "object"})
    assert "valid JSON object" in models.calls[0]["config"].system_instruction
    assert gem.health()["ok"] is True


def test_api_errors_become_invalid_responses() -> None:
    class Broken(FakeModels):
        def generate_content(self, model, contents, config):
            raise RuntimeError("quota exceeded")

    gem = GeminiProvider(api_key="test", client=SimpleNamespace(models=Broken()))
    response = gem.generate_text(ModelRequest(prompt="hi"))
    assert response.valid is False and "quota exceeded" in response.error


def test_factory_and_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    gem = create_provider("gemini")
    assert isinstance(gem, GeminiProvider) and gem.model == "gemma-4-26b-a4b-it"
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        _ = gem.client
