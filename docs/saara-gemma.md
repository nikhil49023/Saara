# saara-gemma: training plan and pipeline

`saara-gemma` is a **planned** fine-tune of **Gemma 4 E4B** (`google/gemma-4-E4B-it`, Apache 2.0) built to be Saara's
default local model. This page documents the full process. Numbers for a specific release are generated
automatically by `report.py` into `MODEL_CARD.md` and `TRAINING_REPORT.md`; nothing is typed in by hand.

> **Status: NOT TRAINED.** No saara-gemma model has been trained, evaluated or released. The pipeline is built and
> tested end to end only on a tiny random-weight model with Gemma 4's architecture (`make_tiny_gemma4.py`).
> No real-model training run has happened, so there are no results yet.
> Until a release passes the gate below, Saara keeps using base Gemma 4 models.

## Goal: three skills Saara needs
1. **OCR**: page image → clean Markdown, including historical, handwritten and degraded scans.
2. **Cleaning**: noisy or OCR-damaged text, or text full of web clutter → clean text, without inventing content.
3. **Workflows**: multi-step tool calling, including knowing when *not* to call a tool.

## Data: public datasets only
No synthetic data from other LLMs is used. Every dataset revision is pinned in `data_manifest.json`.

| Skill | Dataset | License | How it is used |
|---|---|---|---|
| OCR | `allenai/olmOCR-mix-1025` | ODC-BY | PDF pages rendered to images, target `natural_text`; historical sources (LoC transcripts, national archives) up-weighted |
| OCR | same + seeded degradations | ODC-BY | Blur, noise, JPEG artifacts, low contrast, faded ink, yellowed paper, skew; target unchanged |
| Cleaning | `PleIAs/Post-OCR-Correction` (English) | CC0 | Real OCR noise → corrected text |
| Cleaning | `HuggingFaceFW/fineweb-edu` + programmatic clutter | ODC-BY | Navigation, links, banners, HTML and whitespace added by seeded code; target is the original text |
| Workflows | `Agent-Ark/Toucan-1.5M` (SFT) | Apache-2.0 | Real multi-step MCP tool trajectories |
| Workflows | `NousResearch/hermes-function-calling-v1` | Apache-2.0 | Function calling with tool responses |
| Workflows | `nvidia/When2Call` | CC-BY-4.0 | Asking for missing information, declining when no tool fits |
| Replay | `allenai/tulu-3-sft-mixture` | ODC-BY | ~5% general data so general ability is not lost |


### Train/inference parity
- Pages are rendered with **PyMuPDF at 150 DPI, the same renderer and setting as `saara ingest pdf`**. Oversized
  scans are capped at 2048 px on the longest side; Gemma's processor would shrink them to its token budget anyway.
- Prompts live in one versioned file (`prompts.py`), meant to be shared with Saara's runtime.
- The image token budget is **1120** (Gemma 4's highest, recommended for OCR) in both training and inference.
- Thinking is off, matching how Saara calls the model.

## Splitting and leakage control
- Splits are made **by group, never by row**: OCR by source document, Post-OCR by newspaper title (so test
  measures unseen publications), web text by domain, tool workflows by MCP server set (so test measures
  unseen tools).
- OCR and When2Call use their **official held-out sets** as test.
- The test set is frozen and hashed before training (`test_row_ids_sha256`).
- Exact and MinHash near-duplicates (Jaccard ≥ 0.8) are removed; test rows are kept first, so any train/val
  row that duplicates a test row is dropped as contamination. Drop counts are recorded by reason.

## Model and training
Architecture facts that shaped the setup (from the Gemma 4 config and model card):

| Gemma 4 E4B fact | Decision |
|---|---|
| Multimodal `Gemma4ForConditionalGeneration` (vision ~150M, audio ~300M) | Loaded with `AutoModelForMultimodalLM`; vision and audio towers frozen |
| 42 layers; the last 18 reuse earlier layers' keys/values (no `k_proj`/`v_proj`) | LoRA targets are discovered from the loaded model, never hard-coded |
| Per-layer embeddings: 4.5B effective / 8B with embeddings, tied embeddings | Embeddings, per-layer tables and `lm_head` frozen |
| Audio tower does not support SDPA attention | Attention implementation chosen per sub-model by transformers |

Training configuration: bf16 LoRA r=32, alpha=32, rsLoRA, dropout 0.05 on language-model attention and MLP
projections; AdamW, LR 1e-4, cosine schedule, 3% warmup, grad clip 1.0, effective batch 32, max length 8192
(longer rows dropped, never truncated); **loss on assistant tokens only** (system, user and tool results masked);
token-balanced skill mixture (OCR 40%, cleaning 25%, workflows 30%, replay 5%). The best checkpoint is chosen,
and early stopping triggered, on a **validation task score** computed by generation, not on validation loss.

## Evaluation
- Base Gemma 4 E4B and saara-gemma are scored on the **same frozen test set** with the same prompts, greedy
  decoding and image budget.
- Metrics: OCR normalized edit distance (OmniDocBench-style; Google reports 0.181 for base E4B on OmniDocBench),
  split by source and clean/degraded page; cleaning CER, invented-word rate and clutter-removal rate; workflow
  tool-name F1, argument exact match and When2Call accuracy.
- Regression guard: GSM8K (thinking on) and MMLU-Pro samples.
- Every metric has a 95% bootstrap CI. Base vs fine-tuned is compared with a **paired bootstrap** on identical rows.

### Release gate
saara-gemma becomes Saara's default only if it improves the primary metric of **all three skills** (p < 0.05)
**and** loses no more than 2 points on the regression guard. `report.py` checks this automatically and writes the
verdict into the model card.

## Reproduce

```bash
cd experiments/saara-gemma
pip install -r requirements.txt
bash run.sh prefetch                       # parallel downloads into the HF cache
python prepare_data.py --out data
bash run.sh test                           # pipeline tests (tiny Gemma 4 tests need TINY_GEMMA4)
bash run.sh pilot                          # 5% of data: throughput, memory, loss sanity
bash run.sh train
bash run.sh eval
bash run.sh report                         # writes release/<version>/MODEL_CARD.md and TRAINING_REPORT.md
```

Hardware used for planning: one 96 GB GPU (RTX PRO 6000) runs the full config. A 40 GB A100 works with
`--batch 2 --grad-accum 16`.
