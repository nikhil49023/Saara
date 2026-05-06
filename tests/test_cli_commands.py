from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mlforge", *args],
        input=input_text,
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=True,
    )


def test_no_arg_splash_and_help() -> None:
    result = run_cli()

    assert "Get started" in result.stdout
    assert "usage: saara" in result.stdout


def test_splash_command() -> None:
    result = run_cli("splash", "--no-animation")

    assert "saara wizard" in result.stdout


def test_wizard_can_exit() -> None:
    result = run_cli("wizard", input_text="6\n")

    assert "Saara Interactive" in result.stdout
    assert "1. Generate topic dataset" in result.stdout
    assert "4. Run workflow config" in result.stdout
    assert "5. Other" in result.stdout
    assert "Initialize workspace" not in result.stdout
    assert "Saara closed." in result.stdout


def test_core_command_flow(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    dataset = tmp_path / "dataset.jsonl"
    report = tmp_path / "report.json"
    exported = tmp_path / "dataset.json"
    labeled = tmp_path / "labeled.jsonl"
    distilled = tmp_path / "distilled.jsonl"
    config = tmp_path / "workflow.json"
    workflow_output = tmp_path / "workflow.jsonl"

    assert "Initialized" in run_cli("init", "--root", str(root)).stdout
    assert "python" in run_cli("doctor").stdout
    setup_output = run_cli("setup", "ollama", "--dry-run").stdout
    assert any(status in setup_output for status in ("manual-required", "dry-run", "already-installed"))
    assert '"ok": true' in run_cli("models", "health", "--provider", "mock").stdout
    assert "firecrawl-local" in run_cli(
        "tools",
        "firecrawl-health",
        "--base-url",
        "http://127.0.0.1:9",
    ).stdout

    generated = run_cli(
        "generate",
        "topic",
        "release smoke",
        "--samples",
        "2",
        "--provider",
        "mock",
        "--out",
        str(dataset),
        "--root",
        str(root),
    )
    assert '"examples": 2' in generated.stdout
    assert dataset.exists()

    validated = run_cli("validate", str(dataset), "--report", str(report))
    assert '"invalid_examples": 0' in validated.stdout
    assert report.exists()

    exported_result = run_cli("export", str(dataset), "--to", "json", "--out", str(exported))
    assert '"rows": 2' in exported_result.stdout
    assert exported.exists()

    labeled_result = run_cli(
        "label",
        str(dataset),
        "--labels",
        "good,bad",
        "--provider",
        "mock",
        "--out",
        str(labeled),
    )
    assert '"examples": 2' in labeled_result.stdout
    assert labeled.exists()

    distilled_result = run_cli(
        "distill",
        str(dataset),
        "--method",
        "dpo",
        "--provider",
        "mock",
        "--out",
        str(distilled),
    )
    assert '"method": "dpo"' in distilled_result.stdout
    assert distilled.exists()

    config.write_text(
        json.dumps(
            {
                "kind": "topic-dataset",
                "topic": "release config",
                "samples": 1,
                "provider": {"name": "mock"},
                "output": {"format": "jsonl", "path": str(workflow_output)},
            }
        ),
        encoding="utf-8",
    )
    run_result = run_cli("run", str(config), "--root", str(root))
    assert '"examples": 1' in run_result.stdout
    assert workflow_output.exists()
