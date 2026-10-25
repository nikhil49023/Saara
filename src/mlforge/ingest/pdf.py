from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

from mlforge.core.hashing import short_hash
from mlforge.core.schemas import DocumentChunk, ResearchSource
from mlforge.providers.base import ModelProvider, ModelRequest

logger = logging.getLogger(__name__)

VISION_PROMPT = """You are an accurate OCR document parser.
Transcribe and extract the full content of this document page.
Preserve all text, structured data, headers, financial figures, tables, and notes.
Format tables as clean Markdown tables.
Do not add conversational commentary; output only the extracted content.
"""


@dataclass(slots=True)
class PDFIngestConfig:
    min_text_chars: int = 40
    vision_fallback: bool = True
    use_docling: bool = True
    dpi: int = 150
    chunk_overlap: int = 50


class PDFIngestor:
    """Extracts structured text, tables, and OCR content from PDF documents.
    
    Supports:
    1. Docling (if available) for advanced semantic layout/tables.
    2. Native PyMuPDF (fitz) for fast text & table bounding-box extraction.
    3. Gemma 4 Vision fallback for scanned documents, handwriting, and charts.
    """

    def __init__(
        self,
        provider: ModelProvider | None = None,
        config: PDFIngestConfig | None = None,
    ):
        self.provider = provider
        self.config = config or PDFIngestConfig()

    def ingest(self, pdf_path: str | Path) -> list[DocumentChunk]:
        path = Path(pdf_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        # Try docling first if configured and available
        if self.config.use_docling:
            chunks = self._try_docling(path)
            if chunks:
                return chunks

        # Fallback to PyMuPDF with optional vision transcription
        return self._ingest_with_fitz(path)

    def ingest_batch(self, pdf_paths: list[str | Path]) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []
        for path in pdf_paths:
            try:
                all_chunks.extend(self.ingest(path))
            except Exception as e:
                logger.error(f"Failed to ingest {path}: {e}")
        return all_chunks

    def _try_docling(self, path: Path) -> list[DocumentChunk] | None:
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(str(path))
            doc = result.document
            markdown_content = doc.export_to_markdown()

            if not markdown_content.strip():
                return None

            source = ResearchSource(
                source_id=f"docling-{path.stem}",
                title=path.name,
                path=str(path),
                content_hash=short_hash(markdown_content),
                metadata={"extractor": "docling"},
            )
            return [
                DocumentChunk(
                    chunk_id=f"{path.stem}-docling-0",
                    content=markdown_content,
                    source=source,
                    kind="markdown",
                    metadata={"extractor": "docling"},
                )
            ]
        except ImportError:
            logger.debug("Docling not installed; using PyMuPDF extractor.")
            return None
        except Exception as e:
            logger.warning(f"Docling conversion failed for {path.name}: {e}. Falling back to PyMuPDF.")
            return None

    def _ingest_with_fitz(self, path: Path) -> list[DocumentChunk]:
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required for PDF ingestion. Install with: pip install pymupdf")

        chunks: list[DocumentChunk] = []
        doc = fitz.open(str(path))

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1
                text = page.get_text("text").strip()
                extraction_method = "fitz_text"

                # Check if page is scanned or sparse (few extractable characters)
                if len(text) < self.config.min_text_chars and self.config.vision_fallback:
                    if self.provider is not None:
                        logger.info(f"Page {page_num} has sparse text ({len(text)} chars); triggering Vision OCR.")
                        pix = page.get_pixmap(dpi=self.config.dpi)
                        img_bytes = pix.tobytes("png")
                        vision_text = self._transcribe_with_vision(img_bytes, page_num)
                        if vision_text.strip():
                            text = vision_text
                            extraction_method = "gemma_vision"
                    else:
                        logger.debug(f"Page {page_num} is sparse, but no vision provider was supplied.")

                if not text:
                    continue

                source = ResearchSource(
                    source_id=f"pdf-{path.stem}-p{page_num}",
                    title=f"{path.name} (Page {page_num})",
                    path=str(path),
                    page=page_num,
                    content_hash=short_hash(text),
                    metadata={"extractor": extraction_method},
                )

                chunk = DocumentChunk(
                    chunk_id=f"{path.stem}-p{page_num}",
                    content=text,
                    source=source,
                    kind="text",
                    metadata={
                        "page": page_num,
                        "total_pages": len(doc),
                        "extractor": extraction_method,
                    },
                )
                chunks.append(chunk)
        finally:
            doc.close()

        return chunks

    def _transcribe_with_vision(self, image_bytes: bytes, page_num: int) -> str:
        if self.provider is None:
            return ""

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        request = ModelRequest(
            prompt=VISION_PROMPT,
            metadata={
                "images": [b64_image],
                "image_format": "png",
                "page": page_num,
            },
        )
        response = self.provider.generate_text(request)
        return response.text if response.valid else ""


def ingest_pdf(
    pdf_path: str | Path,
    provider: ModelProvider | None = None,
    min_text_chars: int = 40,
    vision_fallback: bool = True,
) -> list[DocumentChunk]:
    """Convenience helper to ingest a PDF into DocumentChunks."""
    ingestor = PDFIngestor(
        provider=provider,
        config=PDFIngestConfig(
            min_text_chars=min_text_chars,
            vision_fallback=vision_fallback,
        ),
    )
    return ingestor.ingest(pdf_path)
