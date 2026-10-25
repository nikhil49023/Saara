"""LoRA fine-tune of Gemma 4 E4B for Saara (OCR, cleaning, tool workflows). Spec: saara-gemma_Training_Spec.md

Usage:
  python train.py --data data --out runs/v0.1                      # full run
  python train.py --data data --out runs/pilot --pilot 0.05        # pilot: throughput + loss sanity
Key choices (from the Gemma 4 architecture review):
- AutoModelForMultimodalLM/AutoProcessor; image token budget 1120 for OCR (train == inference)
- LoRA only on language-model linear layers found at load time (KV-shared layers have no k/v proj)
- embeddings / per-layer embeddings / lm_head / vision / audio towers frozen
- loss only on assistant tokens; tool responses and prompts masked
- best checkpoint chosen by validation task score, not validation loss
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FROZEN_PARTS = ("vision", "audio", "embed", "lm_head", "per_layer", "multi_modal_projector")


# ---------- data ----------

def load_rows(path: Path, data_dir: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        for message in row["messages"]:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part.get("type") == "image" and "path" in part:
                        part["path"] = str((data_dir / part["path"]).resolve())
        rows.append(row)
    return rows


def render(processor, row: dict, add_generation_prompt: bool = False, messages: list | None = None) -> str:
    return processor.apply_chat_template(
        messages or row["messages"], tools=row.get("tools") or None, tokenize=False,
        add_generation_prompt=add_generation_prompt, enable_thinking=False,
    )


def images_of(messages: list[dict]) -> list[Image.Image]:
    found = []
    for message in messages:
        if isinstance(message.get("content"), list):
            found += [Image.open(p["path"]).convert("RGB") for p in message["content"] if p.get("type") == "image"]
    return found


def approx_tokens(processor, row: dict, image_tokens: int) -> int:
    return len(processor.tokenizer(render(processor, row), add_special_tokens=False).input_ids) + image_tokens * len(
        [1 for m in row["messages"] if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image"])


def apply_mixture(rows: list[dict], lengths: list[int], mix: dict[str, float], max_tokens: int | None, seed: int):
    """Downsample skills so token shares match `mix`; the scarcest skill (relative to its share) is kept whole."""
    by_skill: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_skill[row["skill"]].append(index)
    available = {s: sum(lengths[i] for i in idx) for s, idx in by_skill.items()}
    total = min(available[s] / mix[s] for s in mix if s in available and mix[s] > 0)
    if max_tokens:
        total = min(total, max_tokens)
    rng, chosen, report = random.Random(seed), [], {}
    for skill, indices in by_skill.items():
        budget = mix.get(skill, 0.0) * total
        rng.shuffle(indices)
        used = 0
        for i in indices:
            if used + lengths[i] > budget:
                continue
            chosen.append(i)
            used += lengths[i]
        report[skill] = {"available_tokens": available[skill], "used_tokens": used,
                         "rows": sum(1 for i in chosen if rows[i]["skill"] == skill)}
    rng.shuffle(chosen)
    return [rows[i] for i in chosen], report


class Collator:
    """Batch rows -> model inputs with assistant-only labels."""

    def __init__(self, processor, max_len: int):
        self.processor, self.max_len = processor, max_len
        tok = processor.tokenizer
        ids = lambda t: tok.convert_tokens_to_ids(t)  # noqa: E731
        self.turn_start, self.turn_end = ids("<|turn>"), ids("<turn|>")
        self.resp_start, self.resp_end = ids("<|tool_response>"), ids("<tool_response|>")
        self.model_ids = tok("model", add_special_tokens=False).input_ids
        self.newline = tok("\n", add_special_tokens=False).input_ids[-1]
        self.no_loss = {tok.pad_token_id, getattr(processor, "image_token_id", None), ids("<|image|>")} - {None}

    def label(self, input_ids: list[int]) -> list[int]:
        labels, in_model, in_resp, i, n = [-100] * len(input_ids), False, False, 0, len(input_ids)
        while i < n:
            tid = input_ids[i]
            if tid == self.turn_start and input_ids[i + 1 : i + 1 + len(self.model_ids)] == self.model_ids:
                i += 1 + len(self.model_ids)
                if i < n and input_ids[i] == self.newline:
                    i += 1
                in_model = True
                continue
            if in_model:
                if tid == self.resp_start:
                    in_resp = True
                if not in_resp and tid not in self.no_loss:
                    labels[i] = tid
                if tid == self.resp_end:
                    in_resp = False
                if tid == self.turn_end:
                    in_model = False
            i += 1
        return labels

    def __call__(self, rows: list[dict]) -> dict:
        texts = [render(self.processor, row) for row in rows]
        images = [images_of(row["messages"]) for row in rows]
        kwargs = {"images": images} if any(images) else {}
        batch = self.processor(text=texts, padding=True, return_tensors="pt", **kwargs)
        labels = torch.tensor([self.label(ids.tolist()) for ids in batch["input_ids"]])
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return dict(batch)


# ---------- model ----------

def lora_targets(model) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in PROJ and "language_model" in name:
            if not any(part in name for part in FROZEN_PARTS):
                names.append(name)
    if not names:
        raise RuntimeError("no LoRA targets found: inspect model.named_modules()")
    return names


def set_image_budget(processor, tokens: int) -> None:
    processor.image_processor.max_soft_tokens = tokens
    for attr in ("image_seq_length",):
        if hasattr(processor, attr):
            setattr(processor, attr, tokens)
        if hasattr(processor.image_processor, attr):
            setattr(processor.image_processor, attr, tokens)


def env_info() -> dict:
    import peft
    import transformers

    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True, cwd=Path(__file__).parent).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--image-tokens", type=int, default=1120, choices=[70, 140, 280, 560, 1120])
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--mix", default="ocr=0.40,clean=0.25,workflow=0.30,general=0.05")
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--pilot", type=float, default=None, help="train on this fraction only (throughput/loss sanity)")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--val-task-rows", type=int, default=48, help="val rows generated at each eval for the task score")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()

    from evaluate import generate_outputs
    from metrics import score_row
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    processor = AutoProcessor.from_pretrained(args.model)
    set_image_budget(processor, args.image_tokens)
    train_rows = load_rows(args.data / "train.jsonl", args.data)
    val_rows = load_rows(args.data / "val.jsonl", args.data)

    # Length preflight: drop (never truncate) rows over max_len.
    lengths, kept = [], []
    dropped = Counter()
    for row in train_rows:
        length = approx_tokens(processor, row, args.image_tokens)
        if length > args.max_len:
            dropped[row["skill"]] += 1
            continue
        kept.append(row)
        lengths.append(length)
    mix = {k: float(v) for k, v in (part.split("=") for part in args.mix.split(","))}
    train_rows, mixture = apply_mixture(kept, lengths, mix, args.max_train_tokens, args.seed)
    if args.pilot:
        train_rows = train_rows[: max(8, int(len(train_rows) * args.pilot))]
    val_rows = [r for r in val_rows if approx_tokens(processor, r, args.image_tokens) <= args.max_len]
    rng = random.Random(args.seed)
    val_task = rng.sample(val_rows, min(args.val_task_rows, len(val_rows)))
    for row in val_task:  # val rows have no `reference`; score the final assistant turn as the target
        if row["skill"] == "workflow":
            ref_index = next((i for i, m in enumerate(row["messages"]) if m["role"] == "assistant" and m.get("tool_calls")), None)
            if ref_index is not None:
                row["reference"] = {"prompt_len": ref_index,
                                    "tool_calls": [c["function"] for c in row["messages"][ref_index]["tool_calls"]]}
    val_task = [r for r in val_task if r["skill"] != "workflow" or r.get("reference")]
    print(f"train rows {len(train_rows)} | val rows {len(val_rows)} | val task rows {len(val_task)} | dropped(len) {dict(dropped)}")
    print("mixture:", json.dumps(mixture))

    # No forced attn_implementation: the audio tower does not support sdpa; transformers picks per sub-model.
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    targets = lora_targets(model)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, use_rslora=True,
                      target_modules=targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    target_summary = Counter(name.split(".")[-1] for name in targets)
    print(f"LoRA targets {dict(target_summary)} | trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    collator = Collator(processor, args.max_len)

    class SaaraTrainer(Trainer):
        # Task metrics are added inside evaluation_loop so that logging, early stopping and best-checkpoint
        # selection (which all run inside Trainer.evaluate) see them.
        def evaluation_loop(self, *a, **kw):
            output = super().evaluation_loop(*a, **kw)
            prefix = kw.get("metric_key_prefix", "eval")
            generations = generate_outputs(self.model, processor, val_task, engine="hf", batch_size=4, max_new_tokens=1024)
            by_skill = defaultdict(list)
            for row, text in zip(val_task, generations):
                by_skill[row["skill"]].append(score_row(row, text)["score"])
            per_skill = {s: sum(v) / len(v) for s, v in by_skill.items() if s != "general"}
            output.metrics[f"{prefix}_task_score"] = sum(per_skill.values()) / max(1, len(per_skill))
            output.metrics.update({f"{prefix}_task_{s}": v for s, v in per_skill.items()})
            return output

    targs = TrainingArguments(
        output_dir=str(args.out / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03, weight_decay=0.0, max_grad_norm=1.0, bf16=torch.cuda.is_available(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="eval_task_score",
        greater_is_better=True, logging_steps=5, report_to=["tensorboard"] if _has("tensorboard") else "none",
        remove_unused_columns=False, dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        seed=args.seed, data_seed=args.seed,
    )
    trainer = SaaraTrainer(model=model, args=targs, train_dataset=train_rows, eval_dataset=val_rows,
                           data_collator=collator, callbacks=[EarlyStoppingCallback(args.patience)])
    result = trainer.train()

    adapter_dir, merged_dir = args.out / "adapter", args.out / "merged"
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)

    manifest_path = args.data / "data_manifest.json"
    run_info = {
        "model": args.model, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "env": env_info(), "data_manifest": json.loads(manifest_path.read_text()) if manifest_path.exists() else None,
        "mixture": mixture, "dropped_over_max_len": dict(dropped), "train_rows": len(train_rows), "val_rows": len(val_rows),
        "lora": {"targets_by_type": dict(target_summary), "n_targets": len(targets), "trainable_params": trainable,
                 "total_params": total, "r": args.lora_r, "alpha": args.lora_alpha, "rslora": True, "dropout": 0.05},
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_metric": trainer.state.best_metric,
        "train_runtime_s": result.metrics.get("train_runtime"), "wall_time_s": time.time() - started,
        "log_history": trainer.state.log_history,
    }
    (args.out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    print(f"done: adapter {adapter_dir} | merged {merged_dir} | best {trainer.state.best_metric}")


if __name__ == "__main__":
    main()
