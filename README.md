# Saara

[![PyPI](https://img.shields.io/pypi/v/saara-ai.svg)](https://pypi.org/project/saara-ai/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Hacktoberfest](https://img.shields.io/badge/Hacktoberfest-2026-orange.svg)](CONTRIBUTING.md)

**Saara is a local-first CLI for building ML datasets you can trust.** It turns topics, web research and
PDFs into training data, labels, validates and distills it, and exports it in the formats today's
fine-tuning tools expect, all on your own machine with local models such as Gemma 4.

```text
 research / PDFs ──► ingest ──► generate ──► label ──► validate ──► distill ──► export
 (Firecrawl-local)  (Docling +   (local LLM)                      (SFT / DPO)  (OpenAI, ShareGPT,
                     Gemma OCR)                                                 Alpaca, Vertex, ...)
```

## Contents
- [Install](#install)
- [Quick start (5 minutes, no GPU)](#quick-start-5-minutes-no-gpu)
- [Features](#features)
- [Exporting for training tools](#exporting-for-training-tools)
- [Local models and privacy](#local-models-and-privacy)
- [saara-gemma: Saara's own fine-tuned model](#saara-gemma-saaras-own-fine-tuned-model)
- [Documentation](#documentation)
- [Contributing (Hacktoberfest welcome)](#contributing)

## Install

```bash
pip install saara-ai                 # core CLI
pip install 'saara-ai[all]'          # + Parquet/Arrow/HF export, PDF ingestion, rich UI
```

From source:

```bash
git clone https://github.com/nikhil49023/Saara.git && cd Saara
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[all,dev]'
saara --version
```

## Quick start (5 minutes, no GPU)

Every command below runs offline with `--provider mock`, a deterministic development provider.
Swap in `--provider ollama --model gemma4` (or `vllm` / `openai-compatible`) for real generation.

```bash
saara doctor                                   # check what is installed
saara generate topic "solar energy" --samples 20 --provider mock --output-dir runs/solar
saara validate runs/solar/datasets/solar-energy.jsonl
saara label runs/solar/datasets/solar-energy.jsonl --labels useful,noise --provider mock --output-dir runs/solar
saara distill runs/solar/datasets/solar-energy.jsonl --method dpo --provider mock --output-dir runs/solar
saara export runs/solar/datasets/solar-energy.jsonl --to jsonl --schema openai-chat \
    --split train=0.9,val=0.05,test=0.05 --out exports/solar
```

Prefer a guided flow? Run `saara wizard`. Running `saara` with no arguments shows the splash screen and help.

## Features

| Stage | Command | What it does |
|---|---|---|
| Research | `generate topic --research firecrawl` | Bounded research agent over [Firecrawl-local](https://github.com/mendableai/firecrawl) (`search`, `scrape`); retrieved text is treated as data, not instructions |
| Ingest | `ingest pdf` | Docling layout parsing; PyMuPDF text; Gemma 4 vision OCR fallback for scanned or handwritten pages |
| Generate | `generate topic` | `finetuning`, `pretraining`, `reasoning` and `tool-calling` dataset types; configurable prompts |
| Label | `label` | Constrained label sets with a local model |
| Validate | `validate` | Schema checks, duplicates, label distribution; machine-readable report |
| Distill | `distill --method sft\|dpo` | SFT conversations, or DPO pairs with real negatives (another example's answer, or a truncated answer) |
| Export | `export` | 7 storage formats × 8 training schemas, deterministic splits, sharding, checksums ([details](#exporting-for-training-tools)) |
| Workflows | `run config.json` | Declarative, repeatable pipelines ([example](examples/topic-dataset.json)) |
| Models | `models hardware\|setup\|list\|remove` | Hardware-tiered local Gemma setup, downloads only with consent |

Every run records provenance (sources, run IDs, hashes) so a dataset can be traced back to where it came from.

## Exporting for training tools

```bash
saara export data.jsonl --to jsonl   --schema sharegpt     --out exports/llamafactory   # + dataset_info.json
saara export data.jsonl --to jsonl   --schema openai-chat  --split train=0.9,test=0.1 --group-by metadata.source --out exports/openai
saara export data.jsonl --to parquet --schema chatml       --shard-size 50000 --out exports/chatml
saara export pairs.jsonl --to jsonl  --schema dpo          --out exports/dpo
saara export data.jsonl --to jsonl   --schema vertex-gemini --compress gzip --out exports/vertex
```

| Schema | Use it with |
|---|---|
| `openai-chat` | OpenAI fine-tuning, Together, Fireworks, Azure OpenAI, Unsloth, TRL `SFTTrainer` |
| `sharegpt` | LLaMA-Factory, Axolotl (tool calls as `function_call` / `observation` turns) |
| `alpaca` | Instruction tuners expecting `instruction` / `input` / `output` (+ `history`) |
| `chatml` | Raw-text SFT trainers (`text` field with `<\|im_start\|>` markers) |
| `vertex-gemini` | Vertex AI / Gemini supervised tuning (`contents`, `parts`, `functionCall`) |
| `dpo` | TRL `DPOTrainer` and other preference trainers (`prompt` / `chosen` / `rejected`) |
| `text` | Pretraining / continued pretraining |
| `saara` | Saara's native rows, unchanged |

Storage formats: `jsonl`, `json`, `csv`, `tsv`, `parquet`, `arrow`, `hf` (Hugging Face `save_to_disk`).

Each engine export (`--schema`, `--split`, `--shard-size` or `--compress`) writes:
- `<split>/<name>-00000-of-0000N.<ext>`: data shards
- `export_manifest.json`: counts, dropped rows **with reasons**, sha256 per file, group-leakage check
- `README.md`: a Hugging Face dataset card
- `dataset_info.json`: LLaMA-Factory registration (`sharegpt` and `alpaca`)

Splits are deterministic (seeded hashing), and `--group-by` keeps related rows (same document,
same source) in one split so evaluation data never leaks into training. Rows that a schema cannot
represent (for example tool calls in `alpaca`) are dropped and counted, or fail with `--strict`.
Without engine options, `saara export --to FORMAT --out FILE` still writes a single file as before.

## Local models and privacy

Saara **never bundles model weights and never downloads a model without asking**.

```bash
saara models hardware                     # RAM / VRAM / disk and a recommended tier
saara models setup --dry-run              # show what would be pulled
saara models setup                        # asks for consent, then pulls via Ollama
saara models setup --model-path ./model.gguf --model my-gemma   # offline / air-gapped
saara models remove gemma-4-e4b
```

| Tier | Model | Download | RAM |
|---|---|---|---|
| Edge / CPU | `gemma-4-e2b` | ~1.6 GB | 4 GB |
| Laptop | `gemma-4-e4b` | ~2.9 GB | 8 GB |
| Workstation / GPU | `gemma-4-12b` | ~7.8 GB | 16 GB |

Providers: `mock` (offline, deterministic), `ollama` (`http://localhost:11434`), `vllm` and
`openai-compatible` (default `http://localhost:8000/v1`). Use `--yes` for non-interactive setup in CI.

## saara-gemma: Saara's own fine-tuned model

`experiments/saara-gemma/` contains a production-style pipeline to fine-tune **Gemma 4 E4B** for
Saara's core skills: document OCR (including historical and degraded scans), cleaning noisy text,
and multi-step tool workflows. It uses **public datasets only**: olmOCR-mix, Post-OCR-Correction,
fineweb-edu, Toucan-1.5M, When2Call, Tulu-3.

The pipeline covers group-aware leak-free splits, a contamination guard, assistant-only loss, LoRA
targets chosen from the loaded architecture, early stopping on a task metric, paired-bootstrap
evaluation against base Gemma, and an auto-generated model card. It becomes Saara's default model only
if it beats base Gemma on every skill. **Status: pipeline built and tested; the training run is scheduled.**
See [docs/saara-gemma.md](docs/saara-gemma.md).

## Documentation

- [Installation](docs/install.md) · [User guide](docs/user-guide.md) · [CLI reference](docs/reference/cli.md)
- [Export engine](docs/export.md) · [Model runtime guide](docs/reference/models.md) · [saara-gemma training](docs/saara-gemma.md)
- [Architecture](docs/architecture.md) · [Changelog](CHANGELOG.md)

## Contributing

Contributions are welcome, including for **Hacktoberfest**. Start with [CONTRIBUTING.md](CONTRIBUTING.md),
then pick an issue labelled `good first issue` or `hacktoberfest`. Please follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

```bash
pip install -e '.[all,dev]'
python -m pytest -q
```

## License

MIT, see [LICENSE](LICENSE). Gemma models are released by Google under Apache 2.0; datasets keep their own licenses.
