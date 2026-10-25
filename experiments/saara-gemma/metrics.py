"""Scoring for saara-gemma: per-row scores, Gemma tool-call parsing, bootstrap statistics."""
from __future__ import annotations

import json
import random
import re
from typing import Any

from rapidfuzz.distance import Levenshtein

from clutter import clutter_left

TOOL_CALL_RE = re.compile(r"<\|tool_call>call:([\w.\-]+)(\{.*?\})<tool_call\|>", re.S)


def _ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_thinking(text: str) -> str:
    return re.sub(r"<\|channel>thought.*?<channel\|>", "", text or "", flags=re.S).strip()


def ned(pred: str, ref: str) -> float:
    """Normalized edit distance in [0, 1] (OmniDocBench-style, lower is better)."""
    pred, ref = _ws(pred), _ws(ref)
    if not pred and not ref:
        return 0.0
    return Levenshtein.distance(pred, ref) / max(len(pred), len(ref))


def cer(pred: str, ref: str) -> float:
    ref = _ws(ref)
    return Levenshtein.distance(_ws(pred), ref) / max(1, len(ref))


def novel_word_rate(pred: str, source: str) -> float:
    """Share of output words that appear nowhere in the input: an invention/hallucination proxy."""
    words = re.findall(r"[a-z0-9]+", (pred or "").lower())
    vocab = set(re.findall(r"[a-z0-9]+", (source or "").lower()))
    return sum(w not in vocab for w in words) / len(words) if words else 0.0


def _gemma_args_to_json(raw: str) -> Any:
    text = raw.replace('<|"|>', '"')
    text = re.sub(r'([{,]\s*)([A-Za-z_][\w\-]*)\s*:', r'\1"\2":', text)
    return json.loads(text)


def parse_tool_calls(text: str) -> tuple[list[dict], bool]:
    """Parse Gemma 4 native tool calls. Returns (calls, all_parsed_ok)."""
    calls, ok = [], True
    for name, raw_args in TOOL_CALL_RE.findall(text or ""):
        try:
            calls.append({"name": name, "arguments": _gemma_args_to_json(raw_args)})
        except (json.JSONDecodeError, ValueError):
            calls.append({"name": name, "arguments": None})
            ok = False
    return calls, ok


def _norm_args(args: Any) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False) if args is not None else "<unparsed>"


def tool_call_scores(pred: list[dict], ref: list[dict]) -> dict:
    pred_names, ref_names = [c["name"] for c in pred], [c["name"] for c in ref]
    remaining = list(ref_names)
    hits = 0
    for name in pred_names:
        if name in remaining:
            remaining.remove(name)
            hits += 1
    precision = hits / len(pred_names) if pred_names else 0.0
    recall = hits / len(ref_names) if ref_names else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ref_full = [(c["name"], _norm_args(c["arguments"])) for c in ref]
    exact = sum((c["name"], _norm_args(c["arguments"])) in ref_full for c in pred)
    return {"tool_f1": f1, "args_exact": exact / len(ref) if ref else 0.0}


def when2call_label(text: str) -> str:
    calls, _ = parse_tool_calls(text)
    if calls:
        return "tool_call"
    lowered = _ws(text).lower()
    if re.search(r"\b(unable|can't|cannot|can not|don't have (the )?(ability|capability|access))\b", lowered):
        return "cannot_answer"
    if lowered.endswith("?") or re.search(r"\b(could you|please provide|can you (tell|provide|specify))\b", lowered):
        return "request_for_info"
    return "direct"


def score_row(row: dict, output: str) -> dict:
    """Return metrics for one test row plus `score` in [0, 1] (higher is better) for paired comparison."""
    output = strip_thinking(output)
    skill, ref = row["skill"], row.get("reference") or {}
    if skill == "ocr":
        value = ned(output, row["messages"][-1]["content"])
        return {"ned": value, "score": 1 - value}
    if skill == "clean":
        source = row["messages"][1]["content"]
        value = cer(output, row["messages"][-1]["content"])
        result = {"cer": value, "novel_word_rate": novel_word_rate(output, source), "score": max(0.0, 1 - value)}
        if (row.get("meta") or {}).get("kind") == "web_clutter":
            result["clutter_removed"] = float(clutter_left(output) == 0)
        return result
    if skill == "workflow" and "when2call" in ref:
        predicted = when2call_label(output)
        return {"when2call_label": predicted, "when2call_correct": float(predicted == ref["when2call"]),
                "score": float(predicted == ref["when2call"])}
    if skill == "workflow":
        calls, parsed_ok = parse_tool_calls(output)
        result = tool_call_scores(calls, ref.get("tool_calls") or [])
        return {**result, "calls_parse_ok": float(parsed_ok and bool(calls)), "score": result["tool_f1"]}
    return {"score": 0.0}


def bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(values) / n, means[int(0.025 * iters)], means[int(0.975 * iters)]


def paired_bootstrap(base: dict[str, float], new: dict[str, float], iters: int = 2000, seed: int = 0) -> dict:
    """Paired test on shared row ids: mean delta, 95% CI, one-sided p-value that new <= base."""
    keys = sorted(set(base) & set(new))
    deltas = [new[k] - base[k] for k in keys]
    if not deltas:
        return {"n": 0}
    rng = random.Random(seed)
    n = len(deltas)
    samples = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    p_value = sum(s <= 0 for s in samples) / iters
    return {"n": n, "delta": sum(deltas) / n, "ci95": [samples[int(0.025 * iters)], samples[int(0.975 * iters)]],
            "p_value": p_value}
