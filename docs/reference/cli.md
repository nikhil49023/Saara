# CLI Reference

## Global

```bash
saara
saara --help
```

Running `saara` without arguments prints the splash screen and command help.

## Splash

```bash
saara splash
saara splash --no-animation
saara splash --seconds 0.5
```

## Wizard

```bash
saara wizard
saara interactive
```

`interactive` is an alias for `wizard`.

The wizard opens on the main dataset workflow actions:

- Generate topic dataset
- Label dataset
- Distill dataset
- Run workflow config

Setup, health checks, validation, export, and workspace settings are grouped under `Other`.

## Init

```bash
saara init
saara init --root .mlforge
```

Creates workspace directories for runs, datasets, reports, and cache.

## Tools

```bash
saara tools firecrawl-health
saara tools firecrawl-health --base-url http://localhost:3002
```

## Generate

```bash
saara generate topic "robotics motion planning" \
  --samples 20 \
  --provider mock \
  --format jsonl \
  --output-dir runs/robotics
```

With Firecrawl-local:

```bash
saara generate topic "dataset distillation" \
  --research firecrawl \
  --firecrawl-url http://localhost:3002 \
  --provider ollama \
  --model qwen \
  --base-url http://localhost:11434 \
  --output-dir runs/distillation
```

Configurable generation:

```bash
saara generate topic "agent tool use" \
  --dataset-type tool-calling \
  --include-reasoning \
  --include-tool-calls \
  --system-prompt-file prompts/system.txt \
  --prompt-template-file prompts/topic-template.txt \
  --temperature 0.1 \
  --max-tokens 1024 \
  --output-dir runs/tool-use
```

Supported `--dataset-type` values:

- `finetuning`: chat/SFT-style messages
- `pretraining`: plain text in `output.text`
- `reasoning`: SFT-style examples with a reasoning field
- `tool-calling`: messages plus `tools` and `tool_calls`

When `--output-dir` is set, Saara writes datasets under `datasets/`, reports under
`reports/`, and run artifacts under that directory.

## Label

```bash
saara label dataset.jsonl \
  --labels useful,not-useful \
  --label-field quality \
  --provider mock \
  --system-prompt-file prompts/label-system.txt \
  --prompt-template-file prompts/label-template.txt \
  --output-dir runs/labeling
```

## Distill

```bash
saara distill dataset.jsonl --method sft --out sft.jsonl
saara distill dataset.jsonl --method dpo --out dpo.jsonl
```

## Validate

```bash
saara validate dataset.jsonl
saara validate dataset.jsonl --report validation.json
saara validate dataset.jsonl --output-dir runs/validation
```

The command exits with code `1` if invalid examples are found.

## Export

Single-file conversion (unchanged behaviour):

```bash
saara export dataset.jsonl --to json --out dataset.json
saara export dataset.jsonl --to csv --out dataset.csv
saara export dataset.jsonl --to parquet --out dataset.parquet
saara export dataset.jsonl --to hf --out hf_dataset/
```

Export engine (training schemas, splits, shards, manifest):

```bash
saara export dataset.jsonl --to jsonl --schema openai-chat --split train=0.9,val=0.05,test=0.05 --group-by metadata.source --out exports/openai
saara export dataset.jsonl --to jsonl --schema sharegpt --out exports/llamafactory
saara export pairs.jsonl --to parquet --schema dpo --shard-size 50000 --out exports/dpo
saara export dataset.jsonl --to jsonl --schema vertex-gemini --compress gzip --out exports/vertex
```

Schemas: `openai-chat`, `sharegpt`, `alpaca`, `chatml`, `vertex-gemini`, `dpo`, `text`, `saara`.
Flags: `--split`, `--group-by`, `--shard-size`, `--compress gzip`, `--name`, `--seed`, `--strict`.
Parquet, Arrow, and Hugging Face exports require `saara-ai[data]`. Full details: [Export engine](../export.md).

## Ingest

```bash
saara ingest pdf paper.pdf --out paper-chunks.jsonl
saara ingest pdf ./papers --output-dir runs/papers
saara ingest pdf scan.pdf --provider ollama --model gemma4 --min-text-chars 40
saara ingest pdf ./digital-pdfs --no-vision
```

Docling is used when installed; otherwise PyMuPDF extracts text. Pages with fewer than `--min-text-chars`
characters are sent to the vision model (unless `--no-vision`).

## Models

```bash
saara models hardware
saara models setup --dry-run
saara models setup --model gemma-4-e4b
saara models setup --model-path ./model.gguf --model my-gemma
saara models list
saara models remove gemma-4-e4b
saara models health --provider mock
saara models health --provider ollama --model gemma4 --base-url http://localhost:11434
saara models health --provider vllm --model google/gemma-4-E4B-it --base-url http://localhost:8000/v1
```

`models setup` never downloads without consent; pass `--yes` for non-interactive use.
Supported providers: `mock`, `ollama`, `vllm`, `openai-compatible`.

## Run

```bash
saara run workflow.json
saara run workflow.json --root .mlforge
```

Supported workflow kinds: `topic-dataset` and `label-dataset`.
