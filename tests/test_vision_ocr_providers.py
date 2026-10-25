"""End-to-end: a scanned (text-less) PDF page reaches each real provider as an image."""
from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from mlforge.ingest.pdf import PDFIngestConfig, PDFIngestor
from mlforge.providers.ollama import OllamaProvider


def _scanned_pdf(path: Path) -> Path:
    doc = fitz.open()
    doc.new_page()  # no text layer, like a scan
    doc.save(str(path))
    doc.close()
    return path


def _ingest(provider, tmp_path: Path):
    config = PDFIngestConfig(min_text_chars=20, vision_fallback=True, use_docling=False)
    return PDFIngestor(provider=provider, config=config).ingest(_scanned_pdf(tmp_path / "scan.pdf"))


def test_ollama_receives_page_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict] = []

    def fake_post(self, path, payload):
        sent.append(payload)
        return {"response": "# Transcribed heading", "done": True}

    monkeypatch.setattr(OllamaProvider, "_post", fake_post)
    chunks = _ingest(OllamaProvider(model="gemma4:e4b"), tmp_path)
    assert chunks[0].metadata["extractor"] == "gemma_vision" and "Transcribed heading" in chunks[0].content
    assert sent[0]["model"] == "gemma4:e4b"
    assert base64.b64decode(sent[0]["images"][0]).startswith(b"\x89PNG")


def test_gemini_receives_page_image(tmp_path: Path) -> None:
    pytest.importorskip("google.genai")
    from mlforge.providers.gemini import GeminiProvider

    calls: list = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(contents)
            return SimpleNamespace(text="# Transcribed heading", candidates=[])

    chunks = _ingest(GeminiProvider(api_key="test", client=SimpleNamespace(models=Models())), tmp_path)
    assert chunks[0].metadata["extractor"] == "gemma_vision" and "Transcribed heading" in chunks[0].content
    assert calls[0][0].inline_data.mime_type == "image/png"
    assert calls[0][0].inline_data.data.startswith(b"\x89PNG")


def test_ollama_text_requests_have_no_images(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlforge.providers.base import ModelRequest

    sent: list[dict] = []
    monkeypatch.setattr(OllamaProvider, "_post", lambda self, path, payload: sent.append(payload) or {"response": "ok"})
    OllamaProvider().generate_text(ModelRequest(prompt="hi"))
    assert "images" not in sent[0]
