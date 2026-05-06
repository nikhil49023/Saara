from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.json import JSON

from mlforge.core.artifacts import ArtifactStore
from mlforge.datasets.exporters import export_examples
from mlforge.datasets.io import load_rows
from mlforge.datasets.validators import validate_dataset
from mlforge.providers.factory import create_provider
from mlforge.setup.doctor import doctor_checks
from mlforge.setup.installer import run_setup, TASKS
from mlforge.tools.firecrawl import FirecrawlClient
from mlforge.ui.animations import spinner, success_pulse
from mlforge.ui.interactive import launch_interactive
from mlforge.ui.splash import BRAND_NAME, render_splash
from mlforge.ui.theme import console
from mlforge.workflows.config import load_workflow_config
from mlforge.workflows.distill import DistillWorkflow
from mlforge.workflows.label_dataset import LabelDatasetWorkflow
from mlforge.workflows.topic_dataset import DATASET_TYPES, TopicDatasetWorkflow


FORMATS = ["json", "jsonl", "csv", "tsv", "parquet", "arrow", "hf"]


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        render_splash(animated=False)
        console.print()
        build_parser().print_help()
        return
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        if hasattr(args, "verbose") and args.verbose:
            raise
        console.print(f"[error]Error:[/] {exc}")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saara",
        description=(
            "Saara is a local-first CLI for dataset generation, labeling, validation, "
            "research, and distillation workflows."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Show full stack traces on error")
    sub = parser.add_subparsers(dest="command", required=True)

    splash = sub.add_parser("splash", help="Show the Saara terminal splash screen")
    splash.add_argument("--no-animation", action="store_true", help="Render a static splash")
    splash.add_argument("--seconds", type=float, default=1.4, help="Animation duration")
    splash.set_defaults(func=cmd_splash)

    wizard = sub.add_parser("wizard", help="Open the interactive Saara wizard")
    wizard.set_defaults(func=cmd_interactive)

    interactive = sub.add_parser("interactive", help="Open the interactive Saara wizard")
    interactive.set_defaults(func=cmd_interactive)

    init_cmd = sub.add_parser("init", help="Initialize local Saara directories")
    init_cmd.add_argument("--root", default=".mlforge")
    init_cmd.set_defaults(func=cmd_init)

    doctor = sub.add_parser("doctor", help="Check Saara runtime requirements")
    doctor.add_argument("--firecrawl-url", default="http://localhost:3002")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    doctor.set_defaults(func=cmd_doctor)

    setup = sub.add_parser("setup", help="Step-wise dependency installer")
    setup.add_argument(
        "targets",
        nargs="*",
        default=["all"],
        choices=["all"] + sorted(TASKS.keys()),
    )
    setup.add_argument("--yes", action="store_true", help="Run proposed install commands without prompting")
    setup.add_argument("--dry-run", action="store_true", help="Show actions without running install commands")
    setup.set_defaults(func=cmd_setup)

    models = sub.add_parser("models", help="Inspect local model providers")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    health = models_sub.add_parser("health", help="Check provider health")
    add_provider_args(health)
    health.set_defaults(func=cmd_models_health)

    tools = sub.add_parser("tools", help="Inspect external tools")
    tools_sub = tools.add_subparsers(dest="tools_command", required=True)
    firecrawl = tools_sub.add_parser("firecrawl-health", help="Check Firecrawl-local")
    firecrawl.add_argument("--base-url", default="http://localhost:3002")
    firecrawl.set_defaults(func=cmd_firecrawl_health)

    generate = sub.add_parser("generate", help="Generate datasets")
    generate_sub = generate.add_subparsers(dest="generate_command", required=True)
    topic = generate_sub.add_parser("topic", help="Generate a dataset from a topic")
    topic.add_argument("topic")
    topic.add_argument("--samples", type=int, default=10)
    topic.add_argument("--format", default="jsonl", choices=FORMATS)
    topic.add_argument("--out", default=None)
    topic.add_argument("--output-dir", default=None, help="Directory for datasets, reports, and run artifacts")
    topic.add_argument("--dataset-type", default="finetuning", choices=sorted(DATASET_TYPES))
    topic.add_argument("--system-prompt", default=None)
    topic.add_argument("--system-prompt-file", default=None)
    topic.add_argument("--prompt-template", default=None)
    topic.add_argument("--prompt-template-file", default=None)
    topic.add_argument("--temperature", type=float, default=0.2)
    topic.add_argument("--max-tokens", type=int, default=None)
    topic.add_argument("--include-reasoning", action="store_true")
    topic.add_argument("--include-tool-calls", action="store_true")
    topic.add_argument("--research", default="none", choices=["none", "firecrawl"])
    topic.add_argument("--firecrawl-url", default="http://localhost:3002")
    topic.add_argument("--root", default=".mlforge")
    add_provider_args(topic)
    topic.set_defaults(func=cmd_generate_topic)

    label = sub.add_parser("label", help="Label an existing dataset")
    label.add_argument("input")
    label.add_argument("--labels", required=True, help="Comma-separated allowed labels")
    label.add_argument("--label-field", default="label")
    label.add_argument("--format", default="jsonl", choices=FORMATS)
    label.add_argument("--out", default=None)
    label.add_argument("--output-dir", default=None)
    label.add_argument("--system-prompt", default=None)
    label.add_argument("--system-prompt-file", default=None)
    label.add_argument("--prompt-template", default=None)
    label.add_argument("--prompt-template-file", default=None)
    label.add_argument("--temperature", type=float, default=0.0)
    label.add_argument("--max-tokens", type=int, default=None)
    add_provider_args(label)
    label.set_defaults(func=cmd_label)

    distill = sub.add_parser("distill", help="Create distilled SFT or DPO data from a dataset")
    distill.add_argument("input")
    distill.add_argument("--method", default="sft", choices=["sft", "dpo"])
    distill.add_argument("--format", default="jsonl", choices=FORMATS)
    distill.add_argument("--out", default=None)
    distill.add_argument("--output-dir", default=None)
    add_provider_args(distill)
    distill.set_defaults(func=cmd_distill)

    validate = sub.add_parser("validate", help="Validate a dataset")
    validate.add_argument("path")
    validate.add_argument("--report", default=None)
    validate.add_argument("--output-dir", default=None)
    validate.set_defaults(func=cmd_validate)

    export = sub.add_parser("export", help="Convert a dataset to another format")
    export.add_argument("input")
    export.add_argument("--to", required=True, choices=FORMATS)
    export.add_argument("--out", default=None)
    export.add_argument("--output-dir", default=None)
    export.set_defaults(func=cmd_export)

    run = sub.add_parser("run", help="Run a workflow config")
    run.add_argument("config")
    run.add_argument("--root", default=".mlforge")
    run.set_defaults(func=cmd_run)

    return parser


def add_provider_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Model Provider Settings")
    group.add_argument("--provider", default="mock", choices=["mock", "ollama", "vllm", "openai-compatible"])
    group.add_argument("--model", default=None)
    group.add_argument("--base-url", default=None)
    group.add_argument("--api-key", default=None)


def cmd_init(args: argparse.Namespace) -> None:
    store = ArtifactStore(args.root)
    with spinner("Initializing workspace"):
        store.init()
    success_pulse(f"Initialized {BRAND_NAME} workspace at {store.root}")


def cmd_splash(args: argparse.Namespace) -> None:
    render_splash(animated=not args.no_animation, seconds=max(0.1, args.seconds))


def cmd_interactive(args: argparse.Namespace) -> None:
    launch_interactive()


def cmd_doctor(args: argparse.Namespace) -> None:
    checks = doctor_checks(args.firecrawl_url)
    if args.json:
        console.print_json(data=[check.to_dict() for check in checks])
        return
    
    table = Table(title="Saara Runtime Requirements", show_header=True, header_style="brand")
    table.add_column("Status", width=12)
    table.add_column("Dependency", width=16)
    table.add_column("Required", width=10)
    table.add_column("Detail")

    for check in checks:
        status = "[success]OK[/]" if check.ok else "[error]MISSING[/]"
        required = "yes" if check.required else "no"
        table.add_row(status, check.name, required, check.detail)
        if not check.ok and check.install_url:
            table.add_row("", "", "", f"[dim]install: {check.install_url}[/]")

    console.print(table)


def cmd_setup(args: argparse.Namespace) -> None:
    results = run_setup(args.targets, yes=args.yes, dry_run=args.dry_run)
    console.print_json(data=results)


def cmd_models_health(args: argparse.Namespace) -> None:
    provider = create_provider(args.provider, args.model, args.base_url, args.api_key)
    with spinner("Checking model provider"):
        payload = provider.health()
    console.print_json(data=payload)


def cmd_firecrawl_health(args: argparse.Namespace) -> None:
    with spinner("Checking Firecrawl-local"):
        payload = FirecrawlClient(args.base_url).health()
    console.print_json(data=payload)


def cmd_generate_topic(args: argparse.Namespace) -> None:
    provider = create_provider(args.provider, args.model, args.base_url, args.api_key)
    output_dir = Path(args.output_dir) if args.output_dir else None
    store = ArtifactStore(output_dir or args.root)
    output_path = _resolve_output_path(
        explicit=args.out,
        output_dir=output_dir,
        subdir="datasets",
        stem=_slug(args.topic),
        fmt=args.format,
    )
    workflow = TopicDatasetWorkflow(
        topic=args.topic,
        samples=args.samples,
        provider=provider,
        output_format=args.format,
        output_path=output_path,
        research=args.research,
        firecrawl_url=args.firecrawl_url,
        store=store,
        dataset_type=args.dataset_type,
        system_prompt=_read_text_option(args.system_prompt, args.system_prompt_file),
        prompt_template=_read_text_option(args.prompt_template, args.prompt_template_file),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        include_reasoning=args.include_reasoning,
        include_tool_calls=args.include_tool_calls,
    )
    with spinner("Generating dataset"):
        manifest = workflow.run()
    success_pulse("Dataset generated")
    console.print_json(data=manifest)


def cmd_validate(args: argparse.Namespace) -> None:
    with spinner("Validating dataset"):
        report = validate_dataset(args.path)
    payload = report.to_dict()
    report_path = args.report
    if not report_path and args.output_dir:
        report_path = str(Path(args.output_dir) / "reports" / f"{Path(args.path).stem}-validation.json")
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    success_pulse("Validation complete")
    console.print_json(data=payload)
    if report.invalid_examples:
        raise SystemExit(1)


def cmd_label(args: argparse.Namespace) -> None:
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("--labels must include at least one label")
    provider = create_provider(args.provider, args.model, args.base_url, args.api_key)
    output_path = _resolve_output_path(
        explicit=args.out,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        subdir="datasets",
        stem=f"{Path(args.input).stem}-labeled",
        fmt=args.format,
    )
    if output_path is None:
        raise ValueError("label requires --out or --output-dir")
    workflow = LabelDatasetWorkflow(
        input_path=Path(args.input),
        output_path=output_path,
        provider=provider,
        labels=labels,
        label_field=args.label_field,
        output_format=args.format,
        system_prompt=_read_text_option(args.system_prompt, args.system_prompt_file),
        prompt_template=_read_text_option(args.prompt_template, args.prompt_template_file),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    with spinner("Labeling dataset"):
        manifest = workflow.run()
    success_pulse("Dataset labeled")
    console.print_json(data=manifest)


def cmd_distill(args: argparse.Namespace) -> None:
    provider = create_provider(args.provider, args.model, args.base_url, args.api_key)
    output_path = _resolve_output_path(
        explicit=args.out,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        subdir="datasets",
        stem=f"{Path(args.input).stem}-{args.method}",
        fmt=args.format,
    )
    if output_path is None:
        raise ValueError("distill requires --out or --output-dir")
    workflow = DistillWorkflow(
        input_path=Path(args.input),
        output_path=output_path,
        provider=provider,
        method=args.method,
        output_format=args.format,
    )
    with spinner("Distilling dataset"):
        manifest = workflow.run()
    success_pulse("Dataset distilled")
    console.print_json(data=manifest)


def cmd_export(args: argparse.Namespace) -> None:
    output_path = _resolve_output_path(
        explicit=args.out,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        subdir="exports",
        stem=Path(args.input).stem,
        fmt=args.to,
    )
    if output_path is None:
        raise ValueError("export requires --out or --output-dir")
    with spinner("Exporting dataset"):
        rows = load_rows(args.input)
        output = export_examples(rows, output_path, args.to)
    success_pulse("Export complete")
    console.print_json(data={"input": args.input, "output": str(output), "format": args.to, "rows": len(rows)})


def cmd_run(args: argparse.Namespace) -> None:
    config = load_workflow_config(args.config)
    kind = config.get("kind") or config.get("workflow")
    
    provider_config = config.get("provider", {})
    if not isinstance(provider_config, dict):
        raise ValueError("provider must be an object")
    output_config = config.get("output", {})
    if not isinstance(output_config, dict):
        raise ValueError("output must be an object")

    provider = create_provider(
        provider_config.get("name", "mock"),
        provider_config.get("model"),
        provider_config.get("base_url"),
        provider_config.get("api_key"),
    )
    output_dir = Path(output_config["dir"]) if output_config.get("dir") else None
    output_format = str(output_config.get("format", "jsonl"))

    if kind == "topic-dataset":
        output_path = _resolve_output_path(
            explicit=output_config.get("path"),
            output_dir=output_dir,
            subdir="datasets",
            stem=_slug(str(config["topic"])),
            fmt=output_format,
        )
        workflow = TopicDatasetWorkflow(
            topic=str(config["topic"]),
            samples=int(config.get("samples", 10)),
            provider=provider,
            output_format=output_format,
            output_path=output_path,
            research=str(config.get("research", "none")),
            firecrawl_url=str(config.get("firecrawl_url", "http://localhost:3002")),
            store=ArtifactStore(output_dir or args.root),
            dataset_type=str(config.get("dataset_type", "finetuning")),
            system_prompt=config.get("system_prompt") or _read_file_from_config(config.get("system_prompt_file")),
            prompt_template=config.get("prompt_template") or _read_file_from_config(config.get("prompt_template_file")),
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config["max_tokens"]) if config.get("max_tokens") else None,
            include_reasoning=bool(config.get("include_reasoning", False)),
            include_tool_calls=bool(config.get("include_tool_calls", False)),
        )
    elif kind == "label-dataset":
        output_path = _resolve_output_path(
            explicit=output_config.get("path"),
            output_dir=output_dir,
            subdir="datasets",
            stem=f"{Path(config['input']).stem}-labeled",
            fmt=output_format,
        )
        workflow = LabelDatasetWorkflow(
            input_path=Path(config["input"]),
            output_path=output_path,
            provider=provider,
            labels=config["labels"],
            label_field=config.get("label_field", "label"),
            output_format=output_format,
            system_prompt=config.get("system_prompt") or _read_file_from_config(config.get("system_prompt_file")),
            prompt_template=config.get("prompt_template") or _read_file_from_config(config.get("prompt_template_file")),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config["max_tokens"]) if config.get("max_tokens") else None,
        )
    else:
        raise ValueError(f"Unsupported workflow kind: {kind}")

    with spinner("Running workflow"):
        manifest = workflow.run()
    success_pulse("Workflow complete")
    console.print_json(data=manifest)


def _resolve_output_path(
    explicit: str | Path | None,
    output_dir: Path | None,
    subdir: str,
    stem: str,
    fmt: str,
) -> Path | None:
    if explicit:
        return Path(explicit)
    if not output_dir:
        return None
    suffix = "" if fmt == "hf" else f".{fmt}"
    return output_dir / subdir / f"{stem}{suffix}"


def _read_text_option(value: str | None, file_path: str | None) -> str | None:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    return value


def _read_file_from_config(value: object) -> str | None:
    if not value:
        return None
    return Path(str(value)).read_text(encoding="utf-8")


def _slug(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in safe.split("-") if part) or "dataset"
