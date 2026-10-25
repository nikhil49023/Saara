"""Generate release documentation from run artifacts (never hand-typed numbers).

Usage: python report.py --data data --run runs/v0.1 --results results --out release
Writes: comparison.json (paired tests + release gate), MODEL_CARD.md, TRAINING_REPORT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import paired_bootstrap

SKILLS = ("ocr", "clean", "workflow")
REGRESSION_TOLERANCE = 0.02


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    """{group_name: {row_id: score}} for skills and slices."""
    groups: dict[str, dict[str, float]] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        for name in (row["skill"], row["slice"]):
            groups.setdefault(name, {})[row["id"]] = row["score"]
    return groups


def compare(base_outputs: Path, new_outputs: Path) -> dict:
    base, new = load_scores(base_outputs), load_scores(new_outputs)
    tests = {name: paired_bootstrap(base[name], new[name]) for name in sorted(set(base) & set(new))}
    gate = {}
    for skill in SKILLS:
        test = tests.get(skill, {})
        gate[skill] = bool(test.get("n")) and test["delta"] > 0 and test["p_value"] < 0.05
    regression = tests.get("regression", {})
    gate["regression_guard"] = not regression.get("n") or regression["delta"] >= -REGRESSION_TOLERANCE
    return {"tests": tests, "gate": gate, "release": all(gate.values())}


def _fmt(metric: dict | None) -> str:
    if not metric:
        return "n/a"
    low, high = metric["ci95"]
    return f"{metric['mean']:.3f} [{low:.3f}, {high:.3f}]"


PRIMARY = {"ocr": "ned", "clean": "cer", "workflow": "score", "regression": "score"}
LOWER_IS_BETTER = {"ned", "cer", "novel_word_rate"}


def eval_table(base: dict, new: dict, comparison: dict) -> str:
    lines = ["| Group | n | Metric | Base Gemma 4 E4B | saara-gemma | Δ score (paired) | p |", "|---|---|---|---|---|---|---|"]
    for name in sorted(base["metrics"]):
        b, n = base["metrics"][name], new["metrics"].get(name, {})
        top = name.split("/")[0]
        key = PRIMARY.get(top, "score")
        test = comparison["tests"].get(name, {})
        delta = f"{test['delta']:+.3f} [{test['ci95'][0]:+.3f}, {test['ci95'][1]:+.3f}]" if test.get("n") else "n/a"
        p_value = f"{test['p_value']:.3f}" if test.get("n") else "n/a"
        arrow = "↓" if key in LOWER_IS_BETTER else "↑"
        lines.append(f"| {name} | {b['n']} | {key} {arrow} | {_fmt(b.get(key))} | {_fmt(n.get(key))} | {delta} | {p_value} |")
    return "\n".join(lines)


def loss_curve(history: list[dict]) -> str:
    rows = ["| Step | Train loss | Eval loss | Val task score |", "|---|---|---|---|"]
    by_step: dict[int, dict] = {}
    for entry in history:
        by_step.setdefault(entry.get("step", 0), {}).update(entry)
    for step in sorted(by_step):
        e = by_step[step]
        if "eval_loss" in e or "eval_task_score" in e or step % 50 == 0:
            rows.append(f"| {step} | {e.get('loss', '')} | {e.get('eval_loss', '')} | {e.get('eval_task_score', '')} |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--new-name", default="saara-gemma")
    parser.add_argument("--version", default="v0.1")
    args = parser.parse_args()

    manifest = json.loads((args.data / "data_manifest.json").read_text())
    run = json.loads((args.run / "run_info.json").read_text())
    base = json.loads((args.results / f"{args.base_name}_summary.json").read_text())
    new = json.loads((args.results / f"{args.new_name}_summary.json").read_text())
    comparison = compare(args.results / f"{args.base_name}_outputs.jsonl", args.results / f"{args.new_name}_outputs.jsonl")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(comparison, indent=2))

    a, lora, env = run["args"], run["lora"], run["env"]
    data_rows = "\n".join(f"| `{repo}` | {info['license']} | `{info['revision'][:12]}` |" for repo, info in manifest["datasets"].items())
    counts = "\n".join(f"| {k} | {v} |" for k, v in manifest["counts"].items())
    drops = "\n".join(f"| {k} | {v} |" for k, v in sorted(manifest["drops"].items())) or "| none | 0 |"
    mixture = "\n".join(f"| {s} | {m['rows']} | {m['used_tokens']:,} | {m['available_tokens']:,} |" for s, m in run["mixture"].items())
    gate = "\n".join(f"- {'✅' if ok else '❌'} {name}" for name, ok in comparison["gate"].items())
    verdict = "**PASSED** (eligible to become Saara's default model)" if comparison["release"] else "**NOT PASSED** (do not make default)"
    table = eval_table(base, new, comparison)
    hyper = (f"LoRA r={lora['r']}, alpha={lora['alpha']}, rsLoRA, dropout {lora['dropout']}; "
             f"{lora['n_targets']} target modules {lora['targets_by_type']}; trainable {lora['trainable_params']:,} / "
             f"{lora['total_params']:,} params. LR {a['lr']} cosine, warmup 3%, AdamW, wd 0, grad clip 1.0, "
             f"batch {a['batch']} × grad-accum {a['grad_accum']}, max {a['epochs']} epochs with early stopping "
             f"(patience {a['patience']}) on the validation task score; max length {a['max_len']}; image budget "
             f"{a['image_tokens']} tokens; bf16; thinking disabled; loss on assistant tokens only.")

    card = f"""---
license: apache-2.0
base_model: google/gemma-4-E4B-it
library_name: transformers
pipeline_tag: image-text-to-text
tags: [gemma4, saara, ocr, data-cleaning, function-calling, lora]
datasets:
{chr(10).join(f"- {repo}" for repo in manifest["datasets"])}
---

# saara-gemma-e4b-{args.version}

Fine-tune of `google/gemma-4-E4B-it` for [Saara](https://github.com/nikhil49023/Saara): document OCR (including historical
and degraded scans), cleaning noisy or OCR-damaged text, and multi-step tool-use workflows.
Trained only on public datasets; no synthetic data from other LLMs.

**Release gate:** {verdict}
{gate}

## Evaluation (frozen test split; greedy decoding; image budget {base['image_tokens']})
Values are mean [95% bootstrap CI]. Δ score is the paired difference on identical rows (higher = better for saara-gemma).
{table}

## Training data
| Dataset | License | Revision |
|---|---|---|
{data_rows}

## Training procedure
{hyper}
Hardware: {env['gpu']} · CUDA {env['cuda']} · torch {env['torch']} · transformers {env['transformers']} · peft {env['peft']}.
Training time: {run['train_runtime_s'] or 0:.0f} s. Best checkpoint: `{run['best_checkpoint']}` (val task score {run['best_metric']}).

## Intended use and limitations
- Intended for Saara's local pipeline: page-image transcription, text cleaning, tool-calling agents.
- OCR quality depends on scan quality; check outputs on critical documents. Handwriting coverage is limited to the olmOCR historical sources.
- Workflow skills are evaluated on held-out MCP servers; unseen real-world tools may behave differently.
- Inherits the limitations and biases of Gemma 4 E4B and the training datasets.

## License and attribution
Model: Apache 2.0 (same as Gemma 4). Data attributions: olmOCR-mix-1025 and fineweb-edu (ODC-BY), Post-OCR-Correction (CC0),
Toucan-1.5M and hermes-function-calling-v1 (Apache 2.0), When2Call (CC-BY-4.0), Tulu-3 SFT mixture (ODC-BY).
"""
    (args.out / "MODEL_CARD.md").write_text(card)

    report = f"""# saara-gemma {args.version}: Training Report

Generated by `report.py` from `data_manifest.json`, `run_info.json` and the evaluation outputs.

## 1. Pipeline
1. `prepare_data.py`: download pinned public datasets → filter → format into Gemma 4 chat format (native tool calls) →
   **group-aware split** (OCR by source document, cleaning by publication/domain, workflows by MCP server set) →
   exact + MinHash near-duplicate removal with test rows taking priority (contamination guard) → manifest.
2. `train.py`: LoRA on language-model linear layers (found at load time), assistant-only loss, token-balanced
   skill mixture, early stopping on the validation task score.
3. `evaluate.py`: base and fine-tuned models on the same frozen test split plus GSM8K/MMLU-Pro regression probes.
4. `report.py`: paired bootstrap tests, release gate, this report and the model card.

## 2. Data
Prompt version `{manifest['prompt_version']}`, seed {manifest['seed']}.
Test file sha256 `{manifest['files']['test']['sha256']}`; test row-id sha256 `{manifest['test_row_ids_sha256']}`.

| Split/skill | Rows |
|---|---|
{counts}

Dropped rows by reason:

| Reason | Rows |
|---|---|
{drops}

## 3. Training mixture (token-balanced)
| Skill | Rows | Tokens used | Tokens available |
|---|---|---|---|
{mixture}
Rows over max length (dropped, not truncated): {run['dropped_over_max_len']}

## 4. Configuration
{hyper}

Environment: `{json.dumps(env)}`

## 5. Training curve
{loss_curve(run['log_history'])}

## 6. Evaluation
{table}

Release gate ({'passed' if comparison['release'] else 'not passed'}):
{gate}

## 7. Reproduce
```bash
python prepare_data.py --out data
python train.py --data data --out runs/{args.version}
python evaluate.py --data data --model google/gemma-4-E4B-it --name base --out results
python evaluate.py --data data --model runs/{args.version}/merged --name saara-gemma --out results
python report.py --data data --run runs/{args.version} --results results --out release
```
"""
    (args.out / "TRAINING_REPORT.md").write_text(report)
    print(json.dumps(comparison["gate"], indent=2), "\nrelease:", comparison["release"])


if __name__ == "__main__":
    main()
