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

## Models

```bash
saara models health --provider mock
saara models health --provider ollama --model qwen --base-url http://localhost:11434
saara models health --provider vllm --model Qwen/Qwen3 --base-url http://localhost:8000/v1
```

Supported providers:

- `mock`
- `ollama`
- `vllm`
- `openai-compatible`

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

```bash
saara export dataset.jsonl --to json --out dataset.json
saara export dataset.jsonl --to json --output-dir runs/export
saara export dataset.jsonl --to csv --out dataset.csv
saara export dataset.jsonl --to parquet --out dataset.parquet
saara export dataset.jsonl --to hf --out hf_dataset/
```

Parquet, Arrow, and Hugging Face exports require `saara-ai[data]`.

## Run

```bash
saara run workflow.json
saara run workflow.json --root .mlforge
```

The first supported workflow kind is `topic-dataset`.
