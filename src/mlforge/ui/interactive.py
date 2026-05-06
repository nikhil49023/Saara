from __future__ import annotations

import json
from pathlib import Path

from mlforge.core.artifacts import ArtifactStore
from mlforge.datasets.exporters import export_examples
from mlforge.datasets.io import load_rows
from mlforge.datasets.validators import validate_dataset
from mlforge.providers.factory import create_provider
from mlforge.setup.doctor import doctor_checks
from mlforge.setup.installer import run_setup
from mlforge.tools.firecrawl import FirecrawlClient
from mlforge.ui.animations import animated_header, spinner, success_pulse
from mlforge.ui.splash import render_splash
from mlforge.workflows.config import load_workflow_config
from mlforge.workflows.distill import DistillWorkflow
from mlforge.workflows.label_dataset import LabelDatasetWorkflow
from mlforge.workflows.topic_dataset import TopicDatasetWorkflow


PROVIDERS = ["mock", "ollama", "vllm", "openai-compatible"]
FORMATS = ["jsonl", "json", "csv", "tsv", "parquet", "arrow", "hf"]


def launch_interactive() -> None:
    render_splash(animated=True, seconds=0.8)
    root = ".mlforge"
    while True:
        print()
        animated_header("Saara Interactive", f"workspace: {root}")
        print("1. Generate topic dataset")
        print("2. Label dataset")
        print("3. Distill dataset")
        print("4. Run workflow config")
        print("5. Other")
        print("6. Exit")
        choice = prompt("Choose", "1").strip()
        try:
            if choice == "1":
                action_generate_topic(root)
            elif choice == "2":
                action_label()
            elif choice == "3":
                action_distill()
            elif choice == "4":
                action_run(root)
            elif choice == "5":
                root = launch_other_menu(root)
            elif choice in {"6", "13"} or choice.lower() in {"q", "quit", "exit"}:
                print("Saara closed.")
                return
            else:
                print("Unknown choice.")
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as exc:  # noqa: BLE001 - interactive CLI should present errors cleanly.
            print(f"Error: {exc}")


def launch_other_menu(root: str) -> str:
    while True:
        print()
        animated_header("Other", f"workspace: {root}")
        print("1. Initialize workspace")
        print("2. Doctor dependency check")
        print("3. Setup dependencies")
        print("4. Check model provider")
        print("5. Check Firecrawl-local")
        print("6. Validate dataset")
        print("7. Export dataset")
        print("8. Change workspace root")
        print("9. Back")
        print("10. Exit")
        choice = prompt("Choose", "9").strip()
        try:
            if choice == "1":
                root = action_init(root)
            elif choice == "2":
                action_doctor()
            elif choice == "3":
                action_setup()
            elif choice == "4":
                action_model_health()
            elif choice == "5":
                action_firecrawl_health()
            elif choice == "6":
                action_validate()
            elif choice == "7":
                action_export()
            elif choice == "8":
                root = prompt("Workspace root", root)
            elif choice == "9" or choice.lower() in {"b", "back"}:
                return root
            elif choice == "10" or choice.lower() in {"q", "quit", "exit"}:
                print("Saara closed.")
                raise SystemExit(0)
            else:
                print("Unknown choice.")
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as exc:  # noqa: BLE001 - interactive CLI should present errors cleanly.
            print(f"Error: {exc}")


def action_init(root: str) -> str:
    root = prompt("Workspace root", root)
    store = ArtifactStore(root)
    with spinner("Initializing workspace"):
        store.init()
    success_pulse(f"Initialized Saara workspace at {store.root}")
    return root


def action_model_health() -> None:
    provider_name = choose("Provider", PROVIDERS, "mock")
    default_model = "qwen" if provider_name == "ollama" else None
    model = prompt_optional("Model", default_model)
    base_url = prompt_optional("Base URL", None)
    api_key = prompt_optional("API key", None)
    provider = create_provider(provider_name, model, base_url, api_key)
    with spinner("Checking model provider"):
        result = provider.health()
    print_json(result)


def action_doctor() -> None:
    firecrawl_url = prompt("Firecrawl URL", "http://localhost:3002")
    checks = doctor_checks(firecrawl_url)
    for check in checks:
        marker = "OK" if check.ok else "MISSING"
        required = "required" if check.required else "optional"
        print(f"{marker:8} {check.name:16} {required:8} {check.detail}")
        if not check.ok and check.install_url:
            print(f"         install: {check.install_url}")


def action_setup() -> None:
    targets = prompt("Targets (all/docker/ollama/firecrawl/data/agents/vllm)", "all")
    dry_run = prompt("Dry run only?", "yes").lower() not in {"n", "no"}
    chosen = [target.strip() for target in targets.split(",") if target.strip()]
    results = run_setup(chosen, yes=False, dry_run=dry_run)
    print_json(results)


def action_firecrawl_health() -> None:
    base_url = prompt("Firecrawl URL", "http://localhost:3002")
    with spinner("Checking Firecrawl-local"):
        result = FirecrawlClient(base_url).health()
    print_json(result)


def action_generate_topic(root: str) -> None:
    topic = prompt_required("Topic")
    samples = prompt_int("Samples", 10, minimum=1)
    provider_name = choose("Provider", PROVIDERS, "mock")
    model = prompt_optional("Model", "qwen" if provider_name == "ollama" else None)
    base_url = prompt_optional("Base URL", None)
    api_key = prompt_optional("API key", None)
    output_format = choose("Output format", FORMATS, "jsonl")
    research = choose("Research", ["none", "firecrawl"], "none")
    firecrawl_url = "http://localhost:3002"
    if research == "firecrawl":
        firecrawl_url = prompt("Firecrawl URL", firecrawl_url)
    out = prompt_optional("Output path", None)
    provider = create_provider(provider_name, model, base_url, api_key)
    workflow = TopicDatasetWorkflow(
        topic=topic,
        samples=samples,
        provider=provider,
        output_format=output_format,
        output_path=Path(out) if out else None,
        research=research,
        firecrawl_url=firecrawl_url,
        store=ArtifactStore(root),
    )
    with spinner("Generating dataset"):
        manifest = workflow.run()
    success_pulse("Dataset generated")
    print_json(manifest)


def action_validate() -> None:
    path = prompt_required("Dataset path")
    report_path = prompt_optional("Report path", None)
    with spinner("Validating dataset"):
        report = validate_dataset(path)
    payload = report.to_dict()
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    success_pulse("Validation complete")
    print_json(payload)


def action_label() -> None:
    input_path = prompt_required("Input dataset")
    labels = [label.strip() for label in prompt_required("Labels, comma-separated").split(",") if label.strip()]
    label_field = prompt("Label field", "label")
    output_format = choose("Output format", FORMATS, "jsonl")
    output_path = prompt_required("Output path")
    provider_name = choose("Provider", PROVIDERS, "mock")
    model = prompt_optional("Model", "qwen" if provider_name == "ollama" else None)
    base_url = prompt_optional("Base URL", None)
    api_key = prompt_optional("API key", None)
    provider = create_provider(provider_name, model, base_url, api_key)
    workflow = LabelDatasetWorkflow(
        input_path=Path(input_path),
        output_path=Path(output_path),
        provider=provider,
        labels=labels,
        label_field=label_field,
        output_format=output_format,
    )
    with spinner("Labeling dataset"):
        manifest = workflow.run()
    success_pulse("Dataset labeled")
    print_json(manifest)


def action_distill() -> None:
    input_path = prompt_required("Input dataset")
    method = choose("Method", ["sft", "dpo"], "sft")
    output_format = choose("Output format", FORMATS, "jsonl")
    output_path = prompt_required("Output path")
    provider_name = choose("Provider", PROVIDERS, "mock")
    model = prompt_optional("Model", "qwen" if provider_name == "ollama" else None)
    base_url = prompt_optional("Base URL", None)
    api_key = prompt_optional("API key", None)
    provider = create_provider(provider_name, model, base_url, api_key)
    workflow = DistillWorkflow(
        input_path=Path(input_path),
        output_path=Path(output_path),
        provider=provider,
        method=method,
        output_format=output_format,
    )
    with spinner("Distilling dataset"):
        manifest = workflow.run()
    success_pulse("Dataset distilled")
    print_json(manifest)


def action_export() -> None:
    input_path = prompt_required("Input dataset")
    output_format = choose("Output format", FORMATS, "jsonl")
    output_path = prompt_required("Output path")
    with spinner("Exporting dataset"):
        rows = load_rows(input_path)
        output = export_examples(rows, output_path, output_format)
    success_pulse("Export complete")
    print_json({"input": input_path, "output": str(output), "format": output_format, "rows": len(rows)})


def action_run(root: str) -> None:
    config_path = prompt_required("Workflow config path")
    config = load_workflow_config(config_path)
    kind = config.get("kind") or config.get("workflow")
    if kind != "topic-dataset":
        raise ValueError(f"Unsupported workflow kind: {kind}")
    provider_config = _as_dict(config.get("provider", {}), "provider")
    output_config = _as_dict(config.get("output", {}), "output")
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
        store=ArtifactStore(root),
    )
    with spinner("Running workflow"):
        manifest = workflow.run()
    success_pulse("Workflow complete")
    print_json(manifest)


def prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    try:
        value = input(f"{label}{suffix}: ").strip()
    except EOFError:
        print()
        raise SystemExit(0) from None
    if value:
        return value
    return default or ""


def prompt_required(label: str) -> str:
    while True:
        value = prompt(label)
        if value:
            return value
        print("Required.")


def prompt_optional(label: str, default: str | None = None) -> str | None:
    value = prompt(label, default)
    return value or None


def prompt_int(label: str, default: int, minimum: int | None = None) -> int:
    while True:
        value = prompt(label, str(default))
        try:
            number = int(value)
        except ValueError:
            print("Enter a number.")
            continue
        if minimum is not None and number < minimum:
            print(f"Enter a number >= {minimum}.")
            continue
        return number


def choose(label: str, choices: list[str], default: str) -> str:
    print(f"{label}:")
    for index, choice in enumerate(choices, start=1):
        marker = " default" if choice == default else ""
        print(f"  {index}. {choice}{marker}")
    while True:
        value = prompt(label, default)
        if value in choices:
            return value
        if value.isdigit():
            index = int(value)
            if 1 <= index <= len(choices):
                return choices[index - 1]
        print(f"Choose one of: {', '.join(choices)}")


def print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _as_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value
