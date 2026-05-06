# User Guide

## Initialize

```bash
saara splash
saara wizard
saara init
```

Running `saara` without arguments shows the splash screen and command help. Use `saara wizard`
for the interactive guided flow, and direct subcommands for scripts and automation.

Saara uses terminal animations for interactive sessions: splash animation, section headers,
operation spinners, and completion pulses. Piped output and CI logs stay plain text.

This creates `.mlforge/` with run, cache, dataset, and report directories.

## Check Local Tools

```bash
saara tools firecrawl-health
saara models health --provider ollama --model qwen
```

## Generate a Topic Dataset

Without web research:

```bash
saara generate topic "robotics motion planning" \
  --samples 20 \
  --provider ollama \
  --model qwen \
  --format jsonl
```

With Firecrawl-local research:

```bash
saara generate topic "dataset distillation" \
  --samples 100 \
  --provider ollama \
  --model qwen \
  --research firecrawl \
  --firecrawl-url http://localhost:3002 \
  --format jsonl
```

## Validate a Dataset

```bash
saara validate .mlforge/datasets/dataset-distillation.jsonl \
  --report .mlforge/reports/dataset-distillation-validation.json
```

## Label a Dataset

```bash
saara label data.jsonl \
  --labels useful,not-useful \
  --provider mock \
  --out labeled.jsonl
```

## Distill a Dataset

Create SFT data:

```bash
saara distill data.jsonl --method sft --provider mock --out sft.jsonl
```

Create DPO preference data:

```bash
saara distill data.jsonl --method dpo --provider mock --out dpo.jsonl
```

## Export Formats

```bash
saara export data.jsonl --to csv --out data.csv
saara export data.jsonl --to parquet --out data.parquet
saara export data.jsonl --to hf --out hf_dataset/
```

## Declarative Workflow

```json
{
  "kind": "topic-dataset",
  "topic": "robotics motion planning",
  "samples": 100,
  "research": "firecrawl",
  "provider": {
    "name": "ollama",
    "model": "qwen"
  },
  "output": {
    "format": "jsonl",
    "path": ".mlforge/datasets/robotics-motion-planning.jsonl"
  }
}
```

Run it:

```bash
saara run workflow.json
```

## Runtime Choices

- Use `mock` for smoke tests.
- Use `ollama` for easy local model use.
- Use `vllm` for GPU throughput and OpenAI-compatible serving.
- Use `llama.cpp` for CPU/GGUF support when added.

## Artifact Model

Each workflow writes:

- final dataset
- validation report
- workflow manifest
- intermediate research chunks
- run metadata

Generated examples include source provenance whenever research or document ingestion is used.
