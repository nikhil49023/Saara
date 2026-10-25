"""Evaluate a model on the frozen test split (+ regression benchmarks). Greedy decoding, same prompts
and image budget for every model.

Usage:
  python evaluate.py --data data --model google/gemma-4-E4B-it --name base --out results
  python evaluate.py --data data --model runs/v0.1/merged --name saara-gemma --out results
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from metrics import bootstrap_ci, score_row, strip_thinking

END_MARKERS = ("<turn|>", "<eos>", "<end_of_turn>")


def prompt_messages(row: dict) -> list[dict]:
    if row["skill"] == "workflow":
        return row["messages"][: row["reference"]["prompt_len"]]
    if row["skill"] == "regression":
        return row["messages"]
    return row["messages"][:-1]


def _clean_output(text: str) -> str:
    for marker in END_MARKERS:
        text = text.split(marker)[0]
    return text.replace("<pad>", "").strip()


def render_prompt(processor, row: dict) -> str:
    return processor.apply_chat_template(
        prompt_messages(row), tools=row.get("tools") or None, tokenize=False,
        add_generation_prompt=True, enable_thinking=bool(row.get("thinking")),
    )


def generate_outputs(model, processor, rows: list[dict], engine: str = "hf", batch_size: int = 8,
                     max_new_tokens: int = 2048) -> list[str]:
    from train import images_of

    prompts = [render_prompt(processor, row) for row in rows]
    images = [images_of(prompt_messages(row)) for row in rows]
    if engine == "vllm":
        from vllm import SamplingParams

        requests = [{"prompt": p, **({"multi_modal_data": {"image": im}} if im else {})} for p, im in zip(prompts, images)]
        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, skip_special_tokens=False)
        return [_clean_output(o.outputs[0].text) for o in model.generate(requests, params)]

    import torch

    was_training = model.training
    model.train(False)
    processor.tokenizer.padding_side = "left"
    outputs: list[str] = []
    for start in range(0, len(rows), batch_size):
        texts, ims = prompts[start : start + batch_size], images[start : start + batch_size]
        kwargs = {"images": ims} if any(ims) else {}
        batch = processor(text=texts, padding=True, return_tensors="pt", **kwargs).to(model.device)
        with torch.no_grad():
            generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = generated[:, batch["input_ids"].shape[1] :]
        outputs += [_clean_output(t) for t in processor.batch_decode(new_tokens, skip_special_tokens=False)]
    processor.tokenizer.padding_side = "right"
    model.train(was_training)
    return outputs


def regression_rows(n_gsm8k: int, n_mmlu: int, seed: int) -> list[dict]:
    """General-capability guard (spec §6): GSM8K with thinking on, MMLU-Pro without."""
    from datasets import load_dataset

    rows, rng = [], random.Random(seed)
    if n_gsm8k:
        gsm = list(load_dataset("openai/gsm8k", "main", split="test"))
        for index, item in enumerate(rng.sample(gsm, min(n_gsm8k, len(gsm)))):
            rows.append({"id": f"regression:gsm8k:{index}", "skill": "regression", "thinking": True,
                         "messages": [{"role": "user", "content": item["question"] + "\nEnd with 'Final answer: <number>'."}],
                         "reference": {"kind": "gsm8k", "answer": item["answer"].split("####")[-1].strip().replace(",", "")}})
    if n_mmlu:
        mmlu = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
        letters = "ABCDEFGHIJ"
        for item in rng.sample(mmlu, min(n_mmlu, len(mmlu))):
            options = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(item["options"]))
            rows.append({"id": f"regression:mmlu_pro:{item['question_id']}", "skill": "regression",
                         "messages": [{"role": "user", "content": f"{item['question']}\n{options}\nAnswer with the letter only."}],
                         "reference": {"kind": "mmlu_pro", "answer": item["answer"], "category": item["category"]}})
    return rows


def score_regression(row: dict, output: str) -> dict:
    output, ref = strip_thinking(output), row["reference"]
    if ref["kind"] == "gsm8k":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", output.replace(",", ""))
        correct = float(bool(numbers) and float(numbers[-1]) == float(ref["answer"]))
    else:
        match = re.search(r"\b([A-J])\b", output)
        correct = float(bool(match) and match.group(1) == ref["answer"])
    return {"correct": correct, "score": correct}


def slice_of(row: dict) -> str:
    meta = row.get("meta") or {}
    if row["skill"] == "ocr":
        return f"ocr/{meta.get('ocr_source')}/{'noisy' if meta.get('degraded') else 'clean'}"
    if row["skill"] == "clean":
        return f"clean/{meta.get('kind')}"
    if row["skill"] == "workflow":
        return "workflow/when2call" if "when2call" in row["reference"] else f"workflow/{row['source']['dataset'].split('/')[-1]}"
    return f"regression/{row['reference']['kind']}"


def summarize(rows: list[dict], scores: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row, score in zip(rows, scores):
        groups[row["skill"]].append(score)
        groups[slice_of(row)].append(score)
    summary = {}
    for name, items in sorted(groups.items()):
        entry = {"n": len(items)}
        for key in sorted({k for s in items for k, v in s.items() if isinstance(v, (int, float))}):
            mean, low, high = bootstrap_ci([s[key] for s in items if key in s])
            entry[key] = {"mean": round(mean, 4), "ci95": [round(low, 4), round(high, 4)]}
        summary[name] = entry
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--image-tokens", type=int, default=1120)
    parser.add_argument("--limit", type=int, default=None, help="max test rows per skill (debug only)")
    parser.add_argument("--regression", nargs=2, type=int, default=[200, 300], metavar=("GSM8K", "MMLU_PRO"))
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    args = parser.parse_args()

    from transformers import AutoProcessor

    from train import load_rows, set_image_budget

    rows = load_rows(args.data / "test.jsonl", args.data)
    if args.limit:
        seen: dict[str, int] = defaultdict(int)
        limited = []
        for row in rows:
            seen[row["skill"]] += 1
            if seen[row["skill"]] <= args.limit:
                limited.append(row)
        rows = limited
    rows += regression_rows(*args.regression, seed=0)

    processor = AutoProcessor.from_pretrained(args.model)
    set_image_budget(processor, args.image_tokens)
    if args.engine == "vllm":
        from vllm import LLM

        model = LLM(model=args.model, max_model_len=16384, limit_mm_per_prompt={"image": 1},
                    mm_processor_kwargs={"max_soft_tokens": args.image_tokens}, gpu_memory_utilization=0.9, seed=0)
    else:
        import torch
        from transformers import AutoModelForMultimodalLM

        model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    outputs = generate_outputs(model, processor, rows, engine=args.engine, max_new_tokens=args.max_new_tokens)
    scores = [score_regression(r, o) if r["skill"] == "regression" else score_row(r, o) for r, o in zip(rows, outputs)]

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"{args.name}_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for row, output, score in zip(rows, outputs, scores):
            handle.write(json.dumps({"id": row["id"], "skill": row["skill"], "slice": slice_of(row), "output": output,
                                     **score}, ensure_ascii=False) + "\n")
    summary = {"model": args.model, "name": args.name, "engine": args.engine, "decoding": "greedy",
               "image_tokens": args.image_tokens, "n": len(rows), "metrics": summarize(rows, scores)}
    (args.out / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v.get("score") for k, v in summary["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
