#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlforge.core.artifacts import ArtifactStore  # noqa: E402
from mlforge.datasets.io import load_rows  # noqa: E402
from mlforge.providers.factory import create_provider  # noqa: E402
from mlforge.workflows.config import load_workflow_config  # noqa: E402
from mlforge.workflows.topic_dataset import TopicDatasetWorkflow  # noqa: E402


DEFAULT_CONFIG_DIR = ROOT / "experiments" / "configs"
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Saara's reproducible research artifact.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory containing experiment JSON/YAML configs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for generated datasets, reports, workspaces, and summaries.",
    )
    parser.add_argument(
        "--include-templates",
        action="store_true",
        help="Also run config files whose name includes 'template'.",
    )
    args = parser.parse_args()

    configs = sorted(args.config_dir.glob("*.json"))
    if not args.include_templates:
        configs = [path for path in configs if "template" not in path.name]
    if not configs:
        raise SystemExit(f"No experiment configs found in {args.config_dir}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "datasets").mkdir(exist_ok=True)
    (args.results_dir / "reports").mkdir(exist_ok=True)
    (args.results_dir / "workspaces").mkdir(exist_ok=True)

    summaries = []
    for config_path in configs:
        summaries.append(run_config(config_path, args.results_dir))

    summary_json = args.results_dir / "summary.json"
    summary_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown_summary(args.results_dir / "summary.md", summaries)

    print(f"Wrote {summary_json}")
    print(f"Wrote {args.results_dir / 'summary.md'}")


def run_config(config_path: Path, results_dir: Path) -> dict[str, Any]:
    config = load_workflow_config(config_path)
    kind = config.get("kind") or config.get("workflow")
    if kind != "topic-dataset":
        raise ValueError(f"Unsupported experiment workflow kind in {config_path}: {kind}")

    experiment_id = str(config.get("experiment_id") or config_path.stem)
    provider_config = as_dict(config.get("provider", {}), "provider")
    output_config = as_dict(config.get("output", {}), "output")
    output_path = ROOT / str(output_config.get("path") or f"experiments/results/datasets/{experiment_id}.jsonl")
    output_format = str(output_config.get("format", output_path.suffix.removeprefix(".") or "jsonl"))
    workspace_path = results_dir / "workspaces" / experiment_id
    if workspace_path.exists():
        shutil.rmtree(workspace_path)

    provider = create_provider(
        str(provider_config.get("name", "mock")),
        optional_str(provider_config.get("model")),
        optional_str(provider_config.get("base_url")),
        optional_str(provider_config.get("api_key")),
    )
    store = ArtifactStore(workspace_path)
    workflow = TopicDatasetWorkflow(
        topic=str(config["topic"]),
        samples=int(config.get("samples", 10)),
        provider=provider,
        output_format=output_format,
        output_path=output_path,
        research=str(config.get("research", "none")),
        firecrawl_url=str(config.get("firecrawl_url", "http://localhost:3002")),
        store=store,
        dataset_type=str(config.get("dataset_type", "finetuning")),
        system_prompt=optional_str(config.get("system_prompt")),
        prompt_template=optional_str(config.get("prompt_template")),
        temperature=float(config.get("temperature", 0.2)),
        max_tokens=int(config["max_tokens"]) if config.get("max_tokens") else None,
        include_reasoning=bool(config.get("include_reasoning", False)),
        include_tool_calls=bool(config.get("include_tool_calls", False)),
    )
    manifest = workflow.run()
    rows = load_rows(output_path)

    report_path = Path(manifest["validation_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    copied_report = results_dir / "reports" / f"{experiment_id}-validation.json"
    copied_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    source_examples = sum(1 for row in rows if row.get("sources"))
    provider_name = str(provider_config.get("name", "mock"))
    return {
        "experiment_id": experiment_id,
        "config": str(config_path.relative_to(ROOT)),
        "topic": str(config["topic"]),
        "dataset_type": str(config.get("dataset_type", "finetuning")),
        "provider": provider_name,
        "model": provider_config.get("model"),
        "research": str(config.get("research", "none")),
        "dataset_path": relative(output_path),
        "manifest_path": relative(Path(results_dir / "workspaces" / experiment_id)),
        "validation_report": relative(copied_report),
        "examples": len(rows),
        "valid_examples": report["valid_examples"],
        "invalid_examples": report["invalid_examples"],
        "duplicate_examples": report["duplicate_examples"],
        "source_coverage": round(source_examples / len(rows), 4) if rows else 0.0,
    }


def write_markdown_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Saara Experiment Summary",
        "",
        "| Experiment | Topic | Dataset type | Provider | Research | Examples | Valid | Invalid | Duplicates | Source coverage |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {experiment_id} | {topic} | {dataset_type} | {provider} | {research} | {examples} | "
            "{valid_examples} | {invalid_examples} | {duplicate_examples} | {source_coverage:.2%} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "These default runs use the deterministic mock provider. Add local-model and Firecrawl runs",
            "before making dataset-quality claims in a paper.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_dict(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
