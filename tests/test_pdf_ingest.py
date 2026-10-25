from __future__ import annotations

import json
from pathlib import Path

import fitz  # PyMuPDF
import pytest

from mlforge.cli import main
from mlforge.ingest.pdf import PDFIngestConfig, PDFIngestor, ingest_pdf
from mlforge.providers.base import ModelProvider, ModelRequest, ModelResponse


class DummyVisionProvider:
    name = "dummy-vision"

    def __init__(self, transcript: str = "| Item | Cost |\n| --- | --- |\n| Cloud Run | $10 |"):
        self.transcript = transcript
        self.calls: list[ModelRequest] = []

    def generate_text(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        return ModelResponse(
            text=self.transcript,
            provider=self.name,
            model="dummy-gemma-vision",
            latency_ms=15,
            valid=True,
        )

    def generate_json(self, request: ModelRequest, schema: dict) -> ModelResponse:
        return ModelResponse(text="{}", provider=self.name, model="dummy", latency_ms=10)

    def health(self) -> dict:
        return {"status": "ok"}


def _create_sample_pdf(path: Path, pages_text: list[str]) -> Path:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def test_pdf_ingest_text_extraction(tmp_path: Path):
    pdf_path = _create_sample_pdf(
        tmp_path / "sample.pdf",
        [
            "Page 1: GST Invoice INV-2026-001 for Indian MSME Services. Total: INR 45,000.",
            "Page 2: Terms and Conditions of Supply and Bank Account details for NEFT transfer.",
        ],
    )

    chunks = ingest_pdf(pdf_path, min_text_chars=10)

    assert len(chunks) == 2
    assert chunks[0].source.page == 1
    assert "GST Invoice" in chunks[0].content
    assert chunks[1].source.page == 2
    assert "Terms and Conditions" in chunks[1].content
    assert chunks[0].metadata["extractor"] == "fitz_text"


def test_pdf_ingest_vision_fallback_on_sparse_page(tmp_path: Path):
    # Page 1 is text, Page 2 is empty/sparse (simulating a scanned invoice image)
    pdf_path = _create_sample_pdf(
        tmp_path / "scanned_invoice.pdf",
        [
            "Page 1 has plenty of direct digital text content exceeding the minimum character threshold.",
            "",  # empty/scanned page
        ],
    )

    vision_provider = DummyVisionProvider(
        transcript="| Invoice # | Amount |\n| --- | --- |\n| GST-99 | 15000 |"
    )
    ingestor = PDFIngestor(
        provider=vision_provider,
        config=PDFIngestConfig(min_text_chars=20, vision_fallback=True),
    )

    chunks = ingestor.ingest(pdf_path)

    assert len(chunks) == 2
    # Page 1 extracted via text
    assert chunks[0].metadata["extractor"] == "fitz_text"
    # Page 2 extracted via vision fallback
    assert chunks[1].metadata["extractor"] == "gemma_vision"
    assert "GST-99" in chunks[1].content
    assert len(vision_provider.calls) == 1
    assert "images" in vision_provider.calls[0].metadata


def test_cli_ingest_pdf(tmp_path: Path):
    pdf_path = _create_sample_pdf(
        tmp_path / "report.pdf",
        ["Financial summary for Q3 2026 with verified metric revenue figures."],
    )
    out_file = tmp_path / "output_chunks.jsonl"

    main([
        "ingest",
        "pdf",
        str(pdf_path),
        "--out",
        str(out_file),
        "--no-vision",
    ])

    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert "Financial summary" in data["content"]
    assert data["source"]["page"] == 1
