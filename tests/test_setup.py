from __future__ import annotations

from mlforge.setup.doctor import doctor_checks
from mlforge.setup.installer import run_setup


def test_doctor_checks_include_core_dependencies() -> None:
    names = {check.name for check in doctor_checks("http://127.0.0.1:9")}

    assert {"python", "docker", "ollama", "firecrawl-local"}.issubset(names)


def test_setup_dry_run_manual_target() -> None:
    results = run_setup(["ollama"], dry_run=True)

    assert results[0]["target"] == "ollama"
    assert results[0]["status"] in {"already-installed", "manual-required"}
