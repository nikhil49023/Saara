from __future__ import annotations

import json
from pathlib import Path

from examples.gst_msme_demo.run_demo import run_gst_demo


def test_gst_msme_demo_pipeline(tmp_path: Path):
    output_dir = tmp_path / "gst_runs"
    report = run_gst_demo(output_dir=output_dir)

    assert report["total_documents"] == 10
    assert report["total_chunks"] >= 10
    assert report["total_generated_examples"] == 10
    assert report["verified_examples"] > 0
    assert report["dropped_examples"] > 0
    assert 0.0 < report["drop_rate_percentage"] < 100.0

    # Verify output files exist
    verified_file = Path(report["artifacts"]["verified_dataset"])
    drop_file = Path(report["artifacts"]["drop_log"])

    assert verified_file.exists()
    assert drop_file.exists()

    # Verify contents
    verified_lines = verified_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(verified_lines) == report["verified_examples"]
    row = json.loads(verified_lines[0])
    assert "output" in row
    assert "seller_gstin" in row["output"]

    drop_lines = drop_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(drop_lines) == report["dropped_examples"]
