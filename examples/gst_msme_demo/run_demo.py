from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from mlforge.ingest.pdf import PDFIngestConfig, PDFIngestor
from mlforge.providers.base import ModelProvider, ModelRequest, ModelResponse

try:
    from examples.gst_msme_demo.generate_sample_invoices import create_sample_gst_invoices
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_sample_invoices import create_sample_gst_invoices

console = Console()

GSTIN_REGEX = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")


class DemoGemmaVisionProvider:
    """Mock/offline vision provider for scanned invoice demo."""
    name = "gemma-4-vision-ocr"

    def generate_text(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            text="CASH MEMO # 891 — Cafe Chai Shai — GSTIN: 36AAUPC1010A1Z2 — Total: ₹450.00 (Food @ 5% GST)",
            provider=self.name,
            model="gemma-4-vision",
            latency_ms=28,
            valid=True,
        )

    def generate_json(self, request: ModelRequest, schema: dict) -> ModelResponse:
        return ModelResponse(text="{}", provider=self.name, model="gemma-4-vision", latency_ms=20)

    def health(self) -> dict:
        return {"status": "ok"}


def verify_gst_example(example: dict[str, Any], source_text: str) -> tuple[bool, str | None]:
    """Verifies GSTIN format, arithmetic balance, and claim grounding."""
    parsed = example.get("output", {})

    # 1. GSTIN format verification
    seller_gstin = parsed.get("seller_gstin", "")
    if seller_gstin and not GSTIN_REGEX.match(seller_gstin):
        return False, f"INVALID_GSTIN: '{seller_gstin}' does not match standard 15-character GST pattern"

    # 2. Arithmetic verification (Taxable + CGST + SGST + IGST == Total)
    try:
        taxable = float(parsed.get("taxable_amount", 0.0))
        cgst = float(parsed.get("cgst", 0.0))
        sgst = float(parsed.get("sgst", 0.0))
        igst = float(parsed.get("igst", 0.0))
        total = float(parsed.get("total_amount", 0.0))

        if total > 0:
            expected_total = taxable + cgst + sgst + igst
            if abs(expected_total - total) > 1.0:
                return False, f"ARITHMETIC_MISMATCH: Computed ₹{expected_total:.2f} != Total ₹{total:.2f}"
    except (ValueError, TypeError):
        return False, "TYPE_ERROR: Non-numeric financial amounts"

    # 3. Source grounding check: key entities must exist in source chunk
    invoice_no = parsed.get("invoice_no")
    if invoice_no and invoice_no not in source_text:
        return False, f"UNGROUNDED_CLAIM: Invoice number '{invoice_no}' not found in source text"

    return True, None


def _extract_amount(keyword: str, text: str, default: float = 0.0) -> float:
    pattern = rf"{keyword}[^₹\n\r:]*[:₹]\s*₹?\s*([0-9,]+(?:\.[0-9]+)?)"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return default


def run_gst_demo(output_dir: Path | None = None) -> dict[str, Any]:
    base_dir = output_dir or (Path(__file__).resolve().parent / "runs")
    data_dir = Path(__file__).resolve().parent / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       SAARA GST/MSME FINANCE INGESTION & AUDIT PIPELINE         [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]\n")

    # Step 1: Synthesize sample PDFs
    pdf_files = create_sample_gst_invoices(data_dir)
    console.print(f"• Synthesized [bold green]{len(pdf_files)} sample GST/MSME invoices[/bold green] in {data_dir.name}/")

    # Step 2: Ingest PDFs with Docling & Gemma Vision Fallback
    vision_provider = DemoGemmaVisionProvider()
    ingestor = PDFIngestor(
        provider=vision_provider,
        config=PDFIngestConfig(min_text_chars=30, vision_fallback=True),
    )
    chunks = ingestor.ingest_batch(pdf_files)
    console.print(f"• Ingested [bold green]{len(chunks)} document chunks[/bold green] via Docling/PyMuPDF + Gemma Vision OCR.")

    # Step 3: Extract structured financial training examples
    raw_examples: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, 1):
        text = chunk.content

        # Extract values accurately
        gstin_match = re.search(r"GSTIN:\s*([0-9A-Z]{15})", text)
        seller_gstin = gstin_match.group(1) if gstin_match else "36AABCT1234F1Z5"

        taxable_val = _extract_amount("Taxable", text, default=0.0)
        cgst_val = _extract_amount("CGST", text, default=0.0)
        sgst_val = _extract_amount("SGST", text, default=0.0)
        igst_val = _extract_amount("IGST", text, default=0.0)

        total_val = _extract_amount("(?:Grand Total|Total Invoice Value|Total Amount Payable|Total Value|Total Amount|Invoice Total|Total)", text, default=0.0)
        if total_val == 0.0:
            total_val = taxable_val + cgst_val + sgst_val + igst_val

        # If taxable was 0 (e.g. food bill of supply), deduce from total
        if taxable_val == 0.0 and (cgst_val + sgst_val + igst_val == 0.0):
            taxable_val = total_val

        inv_match = re.search(r"(?:Invoice\s*(?:No|Ref)?|Bill\s*No|LR\s*No|Lorry Receipt\s*\(LR\)\s*No):\s*([A-Za-z0-9\-\/]+)", text)
        inv_no = inv_match.group(1) if inv_match else f"INV-{idx}"

        # Deliberately inject 2 controlled anomalies into raw examples to test verification gate!
        if idx == 3:
            # Arithmetic defect injection: Total is off by ₹5,000
            total_val += 5000.0
        elif idx == 7:
            # Corrupted GSTIN injection (14 chars instead of 15)
            seller_gstin = "36INVALID123"

        example = {
            "id": f"saara-gst-{idx:03d}",
            "source_chunk_id": chunk.chunk_id,
            "instruction": "Extract structured GST financial data and verify tax calculation balance.",
            "input": {"raw_text": text},
            "output": {
                "invoice_no": inv_no,
                "seller_gstin": seller_gstin,
                "taxable_amount": taxable_val,
                "cgst": cgst_val,
                "sgst": sgst_val,
                "igst": igst_val,
                "total_amount": total_val,
            },
        }
        raw_examples.append(example)

    # Step 4: Verification & Drop-Rate Audit
    verified_examples: list[dict[str, Any]] = []
    dropped_examples: list[dict[str, Any]] = []
    drop_reasons: dict[str, int] = {}

    for ex in raw_examples:
        chunk = next((c for c in chunks if c.chunk_id == ex["source_chunk_id"]), None)
        source_text = chunk.content if chunk else ""
        is_valid, reason = verify_gst_example(ex, source_text)

        if is_valid:
            verified_examples.append(ex)
        else:
            dropped_examples.append({"example_id": ex["id"], "reason": reason, "example": ex})
            reason_category = reason.split(":")[0] if reason else "UNKNOWN"
            drop_reasons[reason_category] = drop_reasons.get(reason_category, 0) + 1

    total_count = len(raw_examples)
    verified_count = len(verified_examples)
    dropped_count = len(dropped_examples)
    drop_rate_pct = round((dropped_count / total_count) * 100, 1) if total_count > 0 else 0.0

    # Write output artifacts
    verified_file = base_dir / "verified_dataset.jsonl"
    drop_file = base_dir / "drop_log.jsonl"
    report_file = base_dir / "drop_rate_report.json"

    with verified_file.open("w", encoding="utf-8") as f:
        for ex in verified_examples:
            f.write(json.dumps(ex) + "\n")

    with drop_file.open("w", encoding="utf-8") as f:
        for ex in dropped_examples:
            f.write(json.dumps(ex) + "\n")

    report_data = {
        "total_documents": len(pdf_files),
        "total_chunks": len(chunks),
        "total_generated_examples": total_count,
        "verified_examples": verified_count,
        "dropped_examples": dropped_count,
        "drop_rate_percentage": drop_rate_pct,
        "drop_reasons": drop_reasons,
        "artifacts": {
            "verified_dataset": str(verified_file),
            "drop_log": str(drop_file),
        },
    }
    report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    # Display Rich summary table
    table = Table(title="Saara Verification & Drop-Rate Audit Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Count / Value", style="green")
    table.add_column("Significance")

    table.add_row("Input Documents", str(len(pdf_files)), "Indian B2B, B2C, SEZ, Freight, and Scanned Invoices")
    table.add_row("Ingested Chunks", str(len(chunks)), "Parsed via Docling/PyMuPDF + Gemma 4 Vision OCR")
    table.add_row("Raw Generated Rows", str(total_count), "Initial synthetic candidate dataset")
    table.add_row("Verified Surviving Rows", f"[bold green]{verified_count}[/bold green]", "100% mathematically balanced & GSTIN compliant")
    table.add_row("Dropped Defective Rows", f"[bold red]{dropped_count}[/bold red]", "Filtered by Saara Verification Gate")
    table.add_row("Drop Rate", f"[bold yellow]{drop_rate_pct}%[/bold yellow]", "Defects prevented from contaminating student model")

    console.print(table)

    if dropped_examples:
        drop_table = Table(title="Audit Log: Dropped Candidates with Grounding Evidence", border_style="red")
        drop_table.add_column("Example ID", style="dim")
        drop_table.add_column("Reason Code", style="bold red")
        drop_table.add_column("Audit Evidence / Diagnostic")

        for d in dropped_examples:
            drop_table.add_row(d["example_id"], d["reason"].split(":")[0], d["reason"])
        console.print(drop_table)

    console.print(f"\n[bold green]✓ Pipeline Complete:[/bold green] Clean verified dataset ready at [cyan]{verified_file}[/cyan]\n")
    return report_data


if __name__ == "__main__":
    run_gst_demo()
