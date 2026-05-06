from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlforge.core.artifacts import ArtifactStore
from mlforge.datasets.exporters import export_examples
from mlforge.datasets.io import load_rows
from mlforge.datasets.validators import validate_dataset
from mlforge.providers.factory import create_provider
from mlforge.setup.doctor import doctor_checks
from mlforge.setup.installer import run_setup
from mlforge.tools.firecrawl import FirecrawlClient
from mlforge.ui.animations import spinner, success_pulse
from mlforge.ui.interactive import launch_interactive
from mlforge.ui.splash import BRAND_NAME, render_splash
from mlforge.workflows.config import load_workflow_config
from mlforge.workflows.distill import DistillWorkflow
from mlforge.workflows.label_dataset import LabelDatasetWorkflow
from mlforge.workflows.topic_dataset import TopicDatasetWorkflow


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        render_splash(animated=False)
        print()
        build_parser().print_help()
        return
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saara",
        description=(
            "Saara is a local-first CLI for dataset generation, labeling, validation, "
            "research, and distillation workflows."
        ),
    )
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
        choices=["all", "docker", "ollama", "firecrawl", "data", "agents", "vllm"],
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
    topic.add_argument("--format", default="jsonl", choices=["json", "jsonl", "csv", "tsv", "parquet", "arrow", "hf"])
    topic.add_argument("--out", default=None)
    topic.add_argument("--research", default="none", choices=["none", "firecrawl"])
    topic.add_argument("--firecrawl-url", default="http://localhost:3002")
    topic.add_argument("--root", default=".mlforge")
    add_provider_args(topic)
    topic.set_defaults(func=cmd_generate_topic)

    label = sub.add_parser("label", help="Label an existing dataset")
    label.add_argument("input")
    label.add_argument("--labels", required=True, help="Comma-separated allowed labels")
    label.add_argument("--label-field", default="label")
    label.add_argument("--format", default="jsonl", choices=["json", "jsonl", "csv", "tsv", "parquet", "arrow", "hf"])
    label.add_argument("--out", required=True)
    add_provider_args(label)
    label.set_defaults(func=cmd_label)

    distill = sub.add_parser("distill", help="Create distilled SFT or DPO data from a dataset")
    distill.add_argument("input")
    distill.add_argument("--method", default="sft", choices=["sft", "dpo"])
    distill.add_argument("--format", default="jsonl", choices=["json", "jsonl", "csv", "tsv", "parquet", "arrow", "hf"])
    distill.add_argument("--out", required=True)
    add_provider_args(distill)
    distill.set_defaults(func=cmd_distill)

    validate = sub.add_parser("validate", help="Validate a dataset")
    validate.add_argument("path")
    validate.add_argument("--report", default=None)
    validate.set_defaults(func=cmd_validate)

    export = sub.add_parser("export", help="Convert a dataset to another format")
    export.add_argument("input")
    export.add_argument("--to", required=True, choices=["json", "jsonl", "csv", "tsv", "parquet", "arrow", "hf"])
    export.add_argument("--out", required=True)
    export.set_defaults(func=cmd_export)

    run = sub.add_parser("run", help="Run a workflow config")
    run.add_argument("config")
    run.add_argument("--root", default=".mlforge")
    run.set_defaults(func=cmd_run)

    return parser


def add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default="mock", choices=["mock", "ollama", "vllm", "openai-compatible"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)


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
        print(json.dumps([check.to_dict() for check in checks], indent=2, ensure_ascii=False))
        return
    for check in checks:
        marker = "OK" if check.ok else "MISSING"
        required = "required" if check.required else "optional"
        print(f"{marker:8} {check.name:16} {required:8} {check.detail}")
        if not check.ok and check.install_url:
            print(f"         install: {check.install_url}")


def cmd_setup(args: argparse.Namespace) -> None:
    results = run_setup(args.targets, yes=args.yes, dry_run=args.dry_run)
    print(json.dumps(results, indent=2, ensure_ascii=False))


def cmd_models_health(args: argparse.Namespace) -> None:
    provider = create_provider(args.provider, args.model, args.base_url, args.api_key)
    with spinner("Checking model provider"):
        payload = provider.health()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_firecrawl_health(args: argparse.Namespace) -> None:
    with spinner("Checking Firecrawl-local"):
        payload = FirecrawlClient(args.base_url).health()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_generate_topic(args: argparse.Namespace) -> None:
    provider = create_provider(args.provider, args.model, args.base_url, args.api_key)
    workflow = TopicDatasetWorkflow(
        topic=args.topic,
        samples=args.samples,
        provider=provider,
        output_format=args.format,
        output_path=Path(args.out) if args.out else None,
        research=args.research,
        firecrawl_url=args.firecrawl_url,
        store=ArtifactStore(args.root),
    )
    with spinner("Generating dataset"):
        manifest = workflow.run()
    success_pulse("Dataset generated")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def cmd_validate(args: argparse.Namespace) -> None:
    with spinner("Validating dataset"):
        report = validate_dataset(args.path)
    payload = report.to_dict()
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    success_pulse("Validation complete")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if report.invalid_examples:
        raise SystemExit(1)


def cmd_label(args: argparse.Namespace) -> None:
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise SystemExit("--labels must include at least one label")
    provider = create_provider(args.provider, args.model, args.base_url, args.api_key)
    workflow = LabelDatasetWorkflow(
        input_path=Path(args.input),
        output_path=Path(args.out),
        provider=provider,
        labels=labels,
        label_field=args.label_field,
        output_format=args.format,
    )
    with spinner("Labeling dataset"):
        manifest = workflow.run()
    success_pulse("Dataset labeled")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def cmd_distill(args: argparse.Namespace) -> None:
    provider = create_provider(args.provider, args.model, args.base_url, args.api_key)
    workflow = DistillWorkflow(
        input_path=Path(args.input),
        output_path=Path(args.out),
        provider=provider,
        method=args.method,
        output_format=args.format,
    )
    with spinner("Distilling dataset"):
        manifest = workflow.run()
    success_pulse("Dataset distilled")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def cmd_export(args: argparse.Namespace) -> None:
    with spinner("Exporting dataset"):
        rows = load_rows(args.input)
        output = export_examples(rows, args.out, args.to)
    success_pulse("Export complete")
    print(json.dumps({"input": args.input, "output": str(output), "format": args.to, "rows": len(rows)}, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    config = load_workflow_config(args.config)
    kind = config.get("kind") or config.get("workflow")
    if kind != "topic-dataset":
        raise SystemExit(f"Unsupported workflow kind: {kind}")

    provider_config = config.get("provider", {})
    if not isinstance(provider_config, dict):
        raise SystemExit("provider must be an object")
    output_config = config.get("output", {})
    if not isinstance(output_config, dict):
        raise SystemExit("output must be an object")

    provider = create_provider(
        provider_config.get("name", "mock"),
        provider_config.get("model"),
        provider_config.get("base_url"),
        provider_config.get("api_key"),
    )
    workflow = TopicDatasetWorkflow(
        topic=str(config["topic"]),
        samples=int(config.get("samples", 10)),
        provider=provider,
        output_format=str(output_config.get("format", "jsonl")),
        output_path=Path(output_config["path"]) if output_config.get("path") else None,
        research=str(config.get("research", "none")),
        firecrawl_url=str(config.get("firecrawl_url", "http://localhost:3002")),
        store=ArtifactStore(args.root),
    )
    with spinner("Running workflow"):
        manifest = workflow.run()
    success_pulse("Workflow complete")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
