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
  --format jsonl \
  --output-dir runs/robotics
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

For pretraining text, reasoning traces, or tool-calling examples:

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

Use `--dataset-type pretraining` for plain text corpora and `--dataset-type finetuning`
for chat/SFT-style datasets.

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
  "dataset_type": "finetuning",
  "research": "firecrawl",
  "temperature": 0.2,
  "max_tokens": 1024,
  "include_reasoning": false,
  "include_tool_calls": false,
  "system_prompt": "You generate grounded dataset examples and return JSON only.",
  "provider": {
    "name": "ollama",
    "model": "qwen",
    "base_url": "http://localhost:11434"
  },
  "output": {
    "format": "jsonl",
    "dir": "runs/robotics",
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
When an output directory is provided, datasets, reports, and run artifacts are kept together
under that directory for easier experiment tracking.
