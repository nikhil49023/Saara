from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.table import Table

from mlforge import __version__
from mlforge.core.artifacts import ArtifactStore
from mlforge.datasets.export_engine import ExportSpec, export_dataset, parse_splits
from mlforge.datasets.exporters import export_examples
from mlforge.datasets.io import load_rows
from mlforge.datasets.schemas import SCHEMAS
from mlforge.datasets.validators import validate_dataset
from mlforge.ingest.pdf import PDFIngestConfig, PDFIngestor
from mlforge.providers.factory import create_provider
from mlforge.setup.doctor import doctor_checks
from mlforge.setup.installer import TASKS, run_setup
from mlforge.setup.model import MODEL_CATALOG, HardwareDetector, ModelManager
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

QUICKSTART = """quick start:
  saara doctor                                    check runtime requirements
  saara models setup --dry-run                    pick a local Gemma model for this machine
  saara generate topic "solar energy" --samples 20 --provider mock --output-dir runs/solar
  saara validate runs/solar/datasets/solar-energy.jsonl
  saara ingest pdf ./papers --output-dir runs/papers --no-vision
  saara export runs/solar/datasets/solar-energy.jsonl --to jsonl --schema openai-chat \\
      --split train=0.9,val=0.05,test=0.05 --out exports/solar

docs: https://github.com/nikhil49023/Saara#readme
"""

EXPORT_EXAMPLES = """examples:
  saara export data.jsonl --to parquet --out data.parquet
  saara export data.jsonl --to jsonl --schema sharegpt --out exports/llamafactory
  saara export data.jsonl --to jsonl --schema openai-chat --split train=0.9,test=0.1 --group-by metadata.source --out exports/openai
  saara export pairs.jsonl --to parquet --schema dpo --shard-size 50000 --out exports/dpo
  saara export data.jsonl --to jsonl --schema vertex-gemini --compress gzip --out exports/vertex

schemas: openai-chat (OpenAI/Together/Unsloth/TRL), sharegpt (LLaMA-Factory/Axolotl), alpaca, chatml,
         vertex-gemini (Vertex AI tuning), dpo (TRL preference pairs), text (pretraining), saara (native)
"""


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
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
        sys.exit(130)
    except Exception as exc:
        if hasattr(args, "verbose") and args.verbose:
            raise
        console.print(f"[error]Error:[/] {escape(_friendly_error(exc))}")
        console.print("[dim]Run with --verbose for the full traceback.[/]")
        sys.exit(1)


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"file not found: {exc.filename or exc}"
    if isinstance(exc, ModuleNotFoundError):
        return f"missing optional dependency '{exc.name}'. Try: pip install 'saara-ai[all]'"
    return str(exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saara",
        description=(
            "Saara is a local-first CLI for dataset generation, labeling, validation, "
            "research, and distillation workflows."
        ),
        epilog=QUICKSTART,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"saara {__version__}")
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

    models = sub.add_parser("models", help="Inspect local model providers and manage local Gemma models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    health = models_sub.add_parser("health", help="Check provider health")
    add_provider_args(health)
    health.set_defaults(func=cmd_models_health)

    hardware = models_sub.add_parser("hardware", help="Inspect system hardware and recommended model tier")
    hardware.set_defaults(func=cmd_models_hardware)

    models_list = models_sub.add_parser("list", help="List registered local models")
    models_list.set_defaults(func=cmd_models_list)

    setup_m = models_sub.add_parser("setup", help="Configure and pull local Gemma models with hardware tiering")
    setup_m.add_argument("--model", default=None, help="Target model to install or register (e.g. gemma-4-e2b, gemma-4-e4b, gemma-4-12b)")
    setup_m.add_argument("--model-path", default=None, help="Register an existing local model file without downloading")
    setup_m.add_argument("--yes", action="store_true", help="Accept consent prompt non-interactively")
    setup_m.add_argument("--dry-run", action="store_true", help="Display recommendation and catalog without downloading")
    setup_m.set_defaults(func=cmd_models_setup)

    remove_m = models_sub.add_parser("remove", help="Remove a locally registered model")
    remove_m.add_argument("name", help="Name of model to remove")
    remove_m.set_defaults(func=cmd_models_remove)

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

    export = sub.add_parser("export", help="Convert a dataset to another format and training schema",
                            epilog=EXPORT_EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    export.add_argument("input", help="Dataset file (.jsonl, .json, .csv, .tsv)")
    export.add_argument("--to", required=True, choices=FORMATS, help="Storage format")
    export.add_argument("--schema", default=None, choices=sorted(SCHEMAS),
                        help="Training schema (openai-chat, sharegpt, alpaca, chatml, vertex-gemini, dpo, text, saara)")
    export.add_argument("--split", default=None, help="Deterministic splits, e.g. train=0.9,val=0.05,test=0.05")
    export.add_argument("--group-by", default=None, help="Field (dot path) that keeps related rows in one split")
    export.add_argument("--shard-size", type=int, default=None, help="Max rows per output file")
    export.add_argument("--compress", default=None, choices=["gzip"])
    export.add_argument("--name", default=None, help="Dataset name used for shard files and the dataset card")
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--strict", action="store_true", help="Fail instead of dropping rows the schema cannot hold")
    export.add_argument("--out", default=None, help="Output file, or output directory when using engine options")
    export.add_argument("--output-dir", default=None)
    export.set_defaults(func=cmd_export)

    ingest = sub.add_parser("ingest", help="Ingest documents into structured chunks")
    ingest_sub = ingest.add_subparsers(dest="ingest_command", required=True)
    ingest_pdf = ingest_sub.add_parser("pdf", help="Ingest a PDF with Docling and Gemma Vision OCR fallback")
    ingest_pdf.add_argument("path", help="PDF file or directory")
    ingest_pdf.add_argument("--out", default=None, help="Output JSONL chunk file path")
    ingest_pdf.add_argument("--output-dir", default=None, help="Directory to store ingested chunks")
    ingest_pdf.add_argument("--min-text-chars", type=int, default=40, help="Minimum characters threshold before vision fallback")
    ingest_pdf.add_argument("--no-vision", action="store_true", help="Disable vision model fallback")
    add_provider_args(ingest_pdf)
    ingest_pdf.set_defaults(func=cmd_ingest_pdf)

    run = sub.add_parser("run", help="Run a workflow config")
    run.add_argument("config")
    run.add_argument("--root", default=".mlforge")
    run.set_defaults(func=cmd_run)

    return parser


def add_provider_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Model Provider Settings")
    group.add_argument("--provider", default="mock", choices=["mock", "ollama", "vllm", "openai-compatible", "gemini"])
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


def cmd_models_hardware(args: argparse.Namespace) -> None:
    profile = HardwareDetector.detect()
    table = Table(title="Hardware Profile & Tier Recommendation", border_style="cyan")
    table.add_column("Component", style="bold")
    table.add_column("Detected Value", style="green")
    table.add_column("Recommendation / Note")

    table.add_row("System RAM", f"{profile.total_ram_gb} GB (avail: {profile.available_ram_gb} GB)", ">= 12GB recommended for local E4B")
    table.add_row("NVIDIA VRAM", f"{profile.vram_gb} GB" if profile.has_gpu else "None (CPU only)", "CUDA acceleration" if profile.has_gpu else "CPU inference mode")
    table.add_row("Free Storage", f"{profile.free_disk_gb} GB", ">= 10GB free recommended for Gemma 4 weights")

    tier_map = {
        "tier_1": "[cyan]Tier 1: Edge (gemma-4-E2B)[/cyan] - Low RAM / CPU-friendly",
        "tier_2": "[green]Tier 2: Laptop (gemma-4-E4B)[/green] - Balanced performance & speed",
        "tier_3": "[magenta]Tier 3: Workstation (gemma-4-12B)[/magenta] - High capacity",
    }
    table.add_row("Recommended Tier", tier_map.get(profile.recommended_tier, profile.recommended_tier), "Selected based on available RAM/VRAM")
    console.print(table)


def cmd_models_list(args: argparse.Namespace) -> None:
    manager = ModelManager()
    models = manager.list_models()
    if not models:
        console.print("[yellow]No local models registered yet. Run 'saara models setup' or use '--model-path'.[/yellow]")
        return

    table = Table(title="Registered Local Models", border_style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Size", style="green")
    table.add_column("Path", style="dim")
    table.add_column("SHA256 (prefix)")

    for m in models:
        sha = (m.get("sha256") or "")[:12] + "..." if m.get("sha256") else "N/A"
        table.add_row(m["name"], f"{m.get('size_gb', '?')} GB", m["path"], sha)
    console.print(table)


def cmd_models_setup(args: argparse.Namespace) -> None:
    profile = HardwareDetector.detect()
    manager = ModelManager()

    # If --model-path is provided, register directly with zero download
    if getattr(args, "model_path", None):
        target_path = Path(args.model_path)
        name = getattr(args, "model", None) or target_path.stem
        with spinner(f"Registering offline model from {target_path.name}"):
            entry = manager.register_local_model(name, target_path)
        success_pulse(f"Model '{name}' registered successfully.")
        console.print_json(data=entry)
        return

    # Determine recommended or requested model
    tier_to_model = {
        "tier_1": "gemma-4-e2b",
        "tier_2": "gemma-4-e4b",
        "tier_3": "gemma-4-12b",
    }
    model_name = getattr(args, "model", None) or tier_to_model.get(profile.recommended_tier, "gemma-4-e4b")
    manifest = MODEL_CATALOG.get(model_name)
    if not manifest:
        console.print(f"[red]Error: Unknown model '{model_name}'. Available catalog models: {', '.join(MODEL_CATALOG.keys())}[/red]")
        return

    console.print("\n[bold cyan]Saara Model Setup — Hardware Tiering & Consent Gate[/bold cyan]")
    console.print(f"• Detected RAM: [bold]{profile.total_ram_gb} GB[/bold] | GPU VRAM: [bold]{profile.vram_gb} GB[/bold] | Free Disk: [bold]{profile.free_disk_gb} GB[/bold]")
    console.print(f"• Selected Model: [bold green]{manifest.name}[/bold green] ({manifest.parameters})")
    console.print(f"• Download Footprint: [bold]{manifest.disk_size_gb} GB[/bold] | Minimum RAM: [bold]{manifest.min_ram_gb} GB[/bold]")
    console.print(f"• License: [bold]{manifest.license}[/bold] (Open weights, permissive commercial & research use)")
    console.print(f"• Ollama Target: [cyan]{manifest.ollama_tag}[/cyan]")

    if getattr(args, "dry_run", False):
        console.print("\n[dim]Dry-run mode: No model will be downloaded.[/dim]")
        return

    # Check explicit consent
    consent = False
    if getattr(args, "yes", False):
        consent = True
    else:
        # Interactive prompt
        console.print("\n[bold yellow]Privacy & Bandwidth Notice:[/bold yellow] Saara never downloads models without your permission.")
        try:
            choice = input(f"Do you consent to pulling '{manifest.ollama_tag}' via Ollama? [y/N]: ").strip().lower()
            consent = (choice == "y" or choice == "yes")
        except (KeyboardInterrupt, EOFError):
            consent = False

    if not consent:
        console.print("\n[yellow]Model download skipped: Consent not granted.[/yellow]")
        console.print("[dim]Tip: You can use external providers ('--provider openai-compatible' or '--provider mock'), or register an existing file via '--model-path <path>'.[/dim]\n")
        return

    # Execute pull
    with spinner(f"Pulling {manifest.ollama_tag} via Ollama"):
        try:
            success = manager.pull_ollama_model(manifest.ollama_tag)
        except Exception as e:
            console.print(f"[red]Failed to pull model: {e}[/red]")
            return

    if success:
        entry = {
            "name": manifest.name,
            "path": f"ollama:{manifest.ollama_tag}",
            "size_gb": manifest.disk_size_gb,
            "source": "ollama",
        }
        models = manager.list_models()
        models = [m for m in models if m.get("name") != manifest.name]
        models.append(entry)
        manager._save_registry(models)
        success_pulse(f"Model '{manifest.name}' is now active and ready.")
        console.print_json(data=entry)


def cmd_models_remove(args: argparse.Namespace) -> None:
    manager = ModelManager()
    removed = manager.remove_model(args.name)
    if removed:
        success_pulse(f"Model '{args.name}' removed from local registry.")
        console.print_json(data={"status": "removed", "name": args.name})
    else:
        console.print(f"[yellow]Model '{args.name}' not found in local registry.[/yellow]")


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
    if any([args.schema, args.split, args.shard_size, args.compress]):
        _cmd_export_engine(args)
        return
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


def cmd_ingest_pdf(args: argparse.Namespace) -> None:
    pdf_path = Path(args.path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"Path does not exist: {pdf_path}")

    provider = None
    if not getattr(args, "no_vision", False):
        provider = create_provider(
            getattr(args, "provider", "mock"),
            getattr(args, "model", None),
            getattr(args, "base_url", None),
            getattr(args, "api_key", None),
        )

    config = PDFIngestConfig(
        min_text_chars=args.min_text_chars,
        vision_fallback=not getattr(args, "no_vision", False),
    )
    ingestor = PDFIngestor(provider=provider, config=config)

    with spinner(f"Ingesting PDF: {pdf_path.name}"):
        if pdf_path.is_file():
            chunks = ingestor.ingest(pdf_path)
        else:
            pdf_files = sorted(pdf_path.glob("*.pdf"))
            chunks = ingestor.ingest_batch(pdf_files)

    output_path = _resolve_output_path(
        explicit=args.out,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        subdir="chunks",
        stem=f"{pdf_path.stem}-chunks",
        fmt="jsonl",
    )
    if output_path is None:
        output_path = Path(f"{pdf_path.stem}-chunks.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict()) + "\n")

    success_pulse(f"Ingested {len(chunks)} chunks from {pdf_path.name}")
    summary: dict[str, Any] = {
        "source": str(pdf_path),
        "output": str(output_path),
        "total_chunks": len(chunks),
        "extractors": {},
    }
    for c in chunks:
        method = c.metadata.get("extractor", "unknown")
        summary["extractors"][method] = summary["extractors"].get(method, 0) + 1

    console.print_json(data=summary)

def _cmd_export_engine(args: argparse.Namespace) -> None:
    name = args.name or _slug(Path(args.input).stem)
    out_dir = Path(args.out) if args.out else (Path(args.output_dir) / "exports" / name if args.output_dir else None)
    if out_dir is None:
        raise ValueError("export requires --out or --output-dir")
    spec = ExportSpec(
        fmt=args.to, schema=args.schema or "saara",
        splits=parse_splits(args.split) if args.split else {"train": 1.0},
        group_by=args.group_by, shard_size=args.shard_size, compress=args.compress,
        name=name, seed=args.seed, strict=args.strict,
    )
    with spinner(f"Exporting {spec.schema} as {spec.fmt}"):
        manifest = export_dataset(load_rows(args.input), out_dir, spec)
    success_pulse("Export complete")
    console.print_json(data={k: manifest[k] for k in ("schema", "format", "rows_in", "rows_out", "counts", "dropped",
                                                     "group_leakage")} | {"out_dir": str(out_dir)})


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
