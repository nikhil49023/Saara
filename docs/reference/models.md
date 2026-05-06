# Model Runtime Guide

Saara never downloads or pulls models automatically. Users should choose models based on
their hardware, install them themselves, and pass the selected model through CLI flags or
workflow config.

## Runtime Choice

Use Ollama when you want the simplest local developer experience:

- CPU or laptop GPU experiments
- quick local smoke tests
- Windows/macOS/Linux desktop users
- single-user workflows

Use vLLM when you want a cloud or server inference endpoint:

- Linux GPU machines
- Colab, Kaggle, rented GPU notebooks, or cloud VMs
- batch dataset generation
- OpenAI-compatible HTTP serving
- multi-user or long-running generation jobs

Saara can also call hosted OpenAI-compatible endpoints with `--provider openai-compatible`.

## Hardware Tiers

| Tier | Typical hardware | Runtime | Model class to choose |
| --- | --- | --- | --- |
| CPU only / 8-16 GB RAM | laptop or small VM | Ollama | 1B-3B instruct models, preferably quantized |
| Small GPU / 6-8 GB VRAM | entry NVIDIA/AMD GPU | Ollama | 3B-4B instruct models, quantized |
| Colab/Kaggle free GPU / 16 GB VRAM | T4, P100, or similar | vLLM for service, Ollama for simplicity | 7B-8B instruct models in FP16/quantized form |
| Prosumer GPU / 24 GB VRAM | RTX 3090/4090 class | vLLM or Ollama | 7B-14B instruct models, quantized larger models |
| Workstation / 48-80 GB VRAM | A6000, L40S, A100/H100 | vLLM | 14B-32B instruct models, some 70B quantized or sharded |
| Multi-GPU server | A100/H100/L40S cluster | vLLM | 32B-70B+ models with tensor parallelism |
| Hosted API | OpenAI-compatible service | openai-compatible | provider-managed model selected by cost/quality |

## Dataset Workflow Recommendations

- Pretraining text: use a base or instruct model only to synthesize/clean text, then export
  `--dataset-type pretraining`.
- Fine-tuning/SFT: use an instruct model and `--dataset-type finetuning`.
- Reasoning datasets: choose reasoning-capable instruct models and set `--include-reasoning`.
- Tool-calling datasets: choose models that support structured/tool-call outputs and set
  `--dataset-type tool-calling --include-tool-calls`.

## Cloud Notebook Guidance

For Colab/Kaggle-style notebooks, prefer vLLM when the notebook has a CUDA GPU and you want
an HTTP endpoint Saara can call repeatedly:

```bash
vllm serve <model-id> --host 0.0.0.0 --port 8000 --dtype auto --api-key token-abc123
saara generate topic "dataset distillation" \
  --provider vllm \
  --model <model-id> \
  --base-url http://localhost:8000/v1 \
  --api-key token-abc123 \
  --output-dir runs/distillation
```

Use Ollama on notebooks only when you want a simple one-user runtime and are comfortable
with the notebook's package/service limitations. For repeatable cloud inference, vLLM is
the better default.
