"""Pipeline checks against a tiny random-weight Gemma 4 (same architecture, small sizes).

Run: TINY_GEMMA4=/path/to/tiny-gemma4 python -m pytest test_pipeline.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from clutter import add_clutter, clutter_left
from metrics import parse_tool_calls, score_row, tool_call_scores
from prompts import CLEAN_USER, OCR_USER, SYSTEM

TINY = os.environ.get("TINY_GEMMA4")
needs_tiny = pytest.mark.skipif(not TINY, reason="set TINY_GEMMA4 to a tiny Gemma 4 checkpoint")
HERE = Path(__file__).parent


def _rows(tmp: Path) -> list[dict]:
    (tmp / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (300, 400), "white").save(tmp / "images/page.jpg")
    tools = [{"type": "function", "function": {"name": "web_search", "description": "Search the web",
                                               "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                                                              "required": ["query"]}}}]
    call = {"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": {"query": "gemma 4"}}}
    workflow = [{"role": "system", "content": SYSTEM["workflow"]}, {"role": "user", "content": "Find Gemma 4 news"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_1", "content": "TOOL_RESULT_SECRET"},
                {"role": "assistant", "content": "Gemma 4 was released."}]
    return [
        {"id": "o1", "skill": "ocr", "split": "train", "group": "g1", "meta": {"ocr_source": "x", "degraded": []},
         "messages": [{"role": "system", "content": SYSTEM["ocr"]},
                      {"role": "user", "content": [{"type": "image", "path": "images/page.jpg"}, {"type": "text", "text": OCR_USER}]},
                      {"role": "assistant", "content": "PAGE TEXT"}]},
        {"id": "c1", "skill": "clean", "split": "train", "group": "g2", "meta": {"kind": "ocr_noise"},
         "messages": [{"role": "system", "content": SYSTEM["clean"]}, {"role": "user", "content": CLEAN_USER.format(text="Tbe cat")},
                      {"role": "assistant", "content": "The cat"}]},
        {"id": "w1", "skill": "workflow", "split": "train", "group": "g3", "tools": tools, "messages": workflow},
    ]


def test_tool_call_parser_and_scores():
    text = '<|tool_call>call:web_search{query:<|"|>gemma 4<|"|>,limit:5}<tool_call|>'
    calls, ok = parse_tool_calls(text)
    assert ok and calls == [{"name": "web_search", "arguments": {"query": "gemma 4", "limit": 5}}]
    assert tool_call_scores(calls, calls) == {"tool_f1": 1.0, "args_exact": 1.0}


def test_clutter_is_deterministic_and_detectable():
    a, kinds = add_clutter("Para one.\n\nPara two.", 7)
    assert (a, kinds) == add_clutter("Para one.\n\nPara two.", 7)
    assert clutter_left("Para one.\n\nPara two.") == 0


def test_scores_are_in_range():
    row = {"skill": "ocr", "messages": [{}, {}, {"role": "assistant", "content": "hello world"}]}
    assert score_row(row, "hello world")["score"] == 1.0
    assert 0.0 <= score_row(row, "something else")["score"] < 1.0


@needs_tiny
def test_labels_cover_only_assistant_tokens(tmp_path):
    from transformers import AutoProcessor

    from train import Collator, load_rows, set_image_budget

    rows = _rows(tmp_path)
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    rows = load_rows(tmp_path / "train.jsonl", tmp_path)
    processor = AutoProcessor.from_pretrained(TINY)
    set_image_budget(processor, 280)
    collator = Collator(processor, 8192)
    tok = processor.tokenizer
    for row in rows:
        batch = collator([row])
        labels = batch["labels"][0]
        learned = tok.decode([t for t in labels.tolist() if t != -100])
        prompt_side = tok.decode([t for t, l in zip(batch["input_ids"][0].tolist(), labels.tolist()) if l == -100])
        print(row["id"], "LEARNED:", repr(learned))
        assert "TOOL_RESULT_SECRET" not in learned
        assert SYSTEM[row["skill"]][:30] not in learned and SYSTEM[row["skill"]][:30] in prompt_side
        final = row["messages"][-1]["content"]
        assert final in learned
        if row["skill"] == "workflow":
            assert "call:web_search" in learned


@needs_tiny
def test_reference_tool_call_round_trip(tmp_path):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(TINY)
    row = _rows(tmp_path)[2]
    text = processor.apply_chat_template(row["messages"][:3], tools=row["tools"], tokenize=False)
    calls, ok = parse_tool_calls(text)
    assert ok and calls == [{"name": "web_search", "arguments": {"query": "gemma 4"}}]


@needs_tiny
def test_train_two_steps_end_to_end(tmp_path):
    rows = _rows(tmp_path)
    for split in ("train", "val"):
        with (tmp_path / f"{split}.jsonl").open("w") as handle:
            for row in rows * 4:
                handle.write(json.dumps({**row, "split": split}) + "\n")
    (tmp_path / "data_manifest.json").write_text("{}")
    out = tmp_path / "run"
    cmd = [sys.executable, str(HERE / "train.py"), "--data", str(tmp_path), "--out", str(out), "--model", TINY,
           "--image-tokens", "280", "--epochs", "1", "--batch", "2", "--grad-accum", "1", "--eval-steps", "2",
           "--val-task-rows", "3", "--mix", "ocr=0.34,clean=0.33,workflow=0.33", "--lora-r", "4", "--lora-alpha", "4"]
    result = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=900)
    print(result.stdout[-3000:], result.stderr[-3000:])
    assert result.returncode == 0
    info = json.loads((out / "run_info.json").read_text())
    assert info["lora"]["n_targets"] > 0 and (out / "merged").exists() and (out / "adapter").exists()
