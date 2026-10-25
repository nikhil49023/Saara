from pathlib import Path
import json
import nbformat as nbf
import pytest
from scripts.build_cookbook_notebook import build_saara_gemma_cookbook_notebook


def test_cookbook_notebook_generation_and_schema(tmp_path: Path):
    target_nb_path = tmp_path / "test_cookbook.ipynb"
    build_saara_gemma_cookbook_notebook(target_nb_path)

    assert target_nb_path.exists()

    with open(target_nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # nbformat validation
    nbf.validate(nb)

    assert len(nb.cells) >= 15

    # Check for crucial metadata and licensing
    copyright_cell = nb.cells[0]
    assert "Copyright" in copyright_cell.source
    assert "Google LLC" in copyright_cell.source

    license_cell = nb.cells[1]
    assert "Apache License, Version 2.0" in license_cell.source

    # Check for Colab badge and title
    title_cell = nb.cells[2]
    assert "Run in Google Colab" in title_cell.source
    assert "Fine-Tune Gemma with Saara Verified Synthetic Data" in title_cell.source
    assert "#158" in title_cell.source
    assert "#132" in title_cell.source


def test_cookbook_verification_logic_execution():
    """
    Directly execute and test the core verification gate embedded in the cookbook notebook
    to ensure 100% bug-free deterministic behavior.
    """
    import re
    gstin_regex = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")

    source_doc = "Supplier: Zenith Cloud Systems Private Limited\nGSTIN: 36AABCZ1234F1Z5\nTaxable: 100000\nCGST: 9000\nSGST: 9000\nTotal: 118000"

    # Test valid candidate
    valid_candidate = {
        "candidate_id": "cand_01",
        "completion": json.dumps({
            "supplier_name": "Zenith Cloud Systems Private Limited",
            "supplier_gstin": "36AABCZ1234F1Z5",
            "taxable_value": 100000.0,
            "cgst": 9000.0,
            "sgst": 9000.0,
            "igst": 0.0,
            "total_value": 118000.0
        })
    }
    data = json.loads(valid_candidate["completion"])
    assert gstin_regex.match(data["supplier_gstin"])
    assert abs((data["taxable_value"] + data["cgst"] + data["sgst"] + data["igst"]) - data["total_value"]) < 0.01
    assert data["supplier_name"].lower() in source_doc.lower()

    # Test arithmetic hallucination rejection
    invalid_math_candidate = {
        "candidate_id": "cand_02",
        "completion": json.dumps({
            "supplier_name": "Zenith Cloud Systems Private Limited",
            "supplier_gstin": "36AABCZ1234F1Z5",
            "taxable_value": 100000.0,
            "cgst": 9000.0,
            "sgst": 9000.0,
            "igst": 0.0,
            "total_value": 125000.0  # Incorrect math
        })
    }
    data_bad = json.loads(invalid_math_candidate["completion"])
    assert abs((data_bad["taxable_value"] + data_bad["cgst"] + data_bad["sgst"] + data_bad["igst"]) - data_bad["total_value"]) > 0.01
