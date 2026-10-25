# Saara

Saara is a local-first CLI for ML dataset workflows:

- topic-to-dataset generation using Firecrawl-local research
- PDF/document ingestion foundations
- local model provider routing for Ollama and vLLM-compatible servers
- canonical dataset examples with provenance
- labeling and distillation commands
- validation reports
- exports to JSON, JSONL, CSV, Parquet, Arrow, and Hugging Face Dataset directories

The current implementation is an MVP scaffold intended to be extended into the full CLI.

Full planning docs:

- [Architecture](docs/architecture.md)
- [Installation Guide](docs/install.md)
- [User Guide](docs/user-guide.md)
- [CLI Reference](docs/reference/cli.md)

Research artifact:

- [Experiments](experiments/README.md)
- [Baseline comparison matrix](experiments/baselines/comparison-matrix.md)
- [Paper outline](experiments/paper/outline.md)

Tutorials & Gemma Cookbook:

- [Fine-Tuning Gemma with Saara Verified Synthetic Data](notebooks/Fine_Tuning_Gemma_With_Saara_Verified_Synthetic_Data.ipynb) (Google Gemma Cookbook recipe for wishlist #158 & #132)

## Quick Start

```bash
pip install -e .
saara splash
saara wizard
saara init

# 1. Hardware-tiered model setup (pulled only with your explicit consent)
saara models hardware
saara models setup

# 2. PDF & Document ingestion (with Docling and Gemma Vision OCR fallback)
saara ingest pdf path/to/document.pdf --out chunks.jsonl

# 3. Generate dataset from topic or bounded documents
saara generate topic "robotics motion planning" --samples 20 --provider mock --format jsonl --output-dir runs/robotics
saara label .mlforge/datasets/robotics-motion-planning.jsonl --labels useful,not-useful --out labeled.jsonl
saara distill labeled.jsonl --method sft --out distilled.jsonl
saara validate .mlforge/datasets/robotics-motion-planning.jsonl
```

Running `saara` without arguments shows the splash screen and command help. Use `saara wizard`
for the interactive guided flow, and direct subcommands for scripts or automation.
Interactive sessions include terminal animations for the splash screen, menu headers, long-running
operations, and completion states. Scripted or piped output automatically falls back to plain text.

Use `--provider mock` for deterministic local smoke tests without a running model.

Run a declarative workflow:

```bash
saara run examples/topic-dataset.json
```

## Installation

Development install:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Install optional dataset exporters:

```bash
pip install -e '.[data]'
```

Install all optional local features:

```bash
pip install -e '.[all]'
```

Fresh machine runtime setup:

```bash
saara doctor
saara setup docker --dry-run
saara setup ollama --dry-run
saara setup docker ollama
```

On Debian/Ubuntu, Saara installs Docker Engine from Docker's official apt repository.
On Linux, Ollama is installed with the official Ollama installer. Review `--dry-run`
output before running setup commands. Saara does not pull or install models automatically;
choose a model based on your hardware tier.

After installation, use `saara` directly like a traditional CLI. The old `mlforge` command remains
available as a compatibility alias during development.

For an isolated user-level install, use `pipx` once this project is published or packaged:

```bash
pipx install .
```

## Model Distribution & Privacy Guarantee

Saara **never bundles model weights** and **never pulls models automatically**.
Models are pulled only with your explicit permission:

- Run `saara models hardware` to view detected RAM, GPU VRAM, and storage.
- Run `saara models setup` to select an **Apache 2.0 Gemma model** (`gemma-4-e2b`, `gemma-4-e4b`, `gemma-4-12b`) tailored to your system tier:
  - **Tier 1 (Edge / CPU)**: `gemma-4-e2b` (~1.6 GB disk, 4 GB RAM)
  - **Tier 2 (Laptop / Workstation)**: `gemma-4-e4b` (~2.9 GB disk, 8 GB RAM)
  - **Tier 3 (GPU Server / Workstation)**: `gemma-4-12b` (~7.8 GB disk, 16 GB RAM)
- Use `--yes` for non-interactive / CI automation.
- For air-gapped / offline environments with existing GGUF checkpoints, use:
  ```bash
  saara models setup --model-path /path/to/model.gguf --model my-local-gemma
  ```
- To reclaim storage, remove any local model with:
  ```bash
  saara models remove gemma-4-e2b
  ```

## Document Ingestion (Docling + Gemma 4 Vision)

Ingest complex PDFs, research papers, and financial invoices into structured chunks:

```bash
# Ingest single PDF with Docling layout parsing & Gemma Vision OCR fallback
saara ingest pdf invoice.pdf --out invoice_chunks.jsonl

# Ingest an entire directory of PDFs
saara ingest pdf ./documents/ --output-dir ./runs/ingested
```

- High-density text pages use fast semantic parsing.
- Scanned, sparse, or handwritten pages automatically trigger Gemma 4 Vision OCR.
- Disable vision fallback with `--no-vision` for pure digital PDFs.

## Firecrawl Local

Topic generation can use Firecrawl-local at `http://localhost:3002`:

```bash
saara generate topic "dataset distillation" \
  --provider ollama \
  --model qwen \
  --research firecrawl \
  --samples 100
```

The Firecrawl integration is exposed as a typed agent tool named `firecrawl_local`.
The topic workflow uses a bounded `ResearchAgent` that calls:

- `firecrawl_local.search(query, limit)`
- `firecrawl_local.scrape(url)`

LangChain is not required for the core workflow. Saara uses its own small typed tool interface
so Firecrawl-local calls are deterministic, auditable, and easy to test. A small adapter is
included for projects that want LangChain-compatible tools via the optional `saara-ai[agents]`
extra.

## Configurable Dataset Modes

Generation can target multiple training dataset shapes:

- `finetuning`: chat/SFT-style message examples
- `pretraining`: plain text examples in `output.text`
- `reasoning`: examples with a `reasoning` field
- `tool-calling`: examples with `tools` and `tool_calls`

Most runtime and prompting behavior is user-configurable from CLI flags or workflow JSON:
provider base URLs, model names, API keys, Firecrawl URL, system prompt, prompt template,
temperature, max tokens, output format, and output directory. When `--output-dir` is used,
Saara writes datasets, reports, and run artifacts into that directory.

## Runtime Providers

- `mock`: deterministic development provider
- `ollama`: `http://localhost:11434`
- `vllm`: OpenAI-compatible endpoint, default `http://localhost:8000/v1`

## Dataset Formats

Supported exports:

- `json`
- `jsonl`
- `csv`
- `parquet` with optional `pyarrow`
- `arrow` with optional `pyarrow`
- `hf` with optional `datasets`
