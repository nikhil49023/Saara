from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlforge.datasets.export_engine import ExportSpec, assign_split, export_dataset, parse_splits
from mlforge.datasets.schemas import (
    SCHEMAS,
    SchemaError,
    alpaca,
    dpo,
    openai_chat,
    sharegpt,
    vertex_gemini,
)

ROOT = Path(__file__).resolve().parents[1]

CALL = {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": {"q": "gemma"}}}
TOOL_ROW = {
    "id": "t1", "task_type": "tool-calling", "metadata": {"source": "doc-a"},
    "tools": [{"type": "function", "function": {"name": "search", "description": "Web search",
                                                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}],
    "messages": [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Find Gemma news"},
        {"role": "assistant", "content": "", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"result\": \"Gemma 4 released\"}"},
        {"role": "assistant", "content": "Gemma 4 was released."},
    ],
}
PLAIN_ROW = {"id": "p1", "task_type": "finetuning", "metadata": {"source": "doc-b"},
             "input": {"instruction": "Summarize", "input": "Long text"}, "output": {"output": "Short text"}}


def test_openai_chat_serializes_arguments_as_json_string() -> None:
    record = openai_chat(TOOL_ROW)
    call = record["messages"][2]["tool_calls"][0]
    assert call["function"] == {"name": "search", "arguments": "{\"q\": \"gemma\"}"}
    assert record["messages"][3]["tool_call_id"] == "call_1"
    assert record["tools"][0]["function"]["name"] == "search"


def test_sharegpt_uses_function_call_and_observation_turns() -> None:
    record = sharegpt(TOOL_ROW)
    roles = [turn["from"] for turn in record["conversations"]]
    assert roles == ["human", "function_call", "observation", "gpt"]
    assert record["system"] == "Use tools."
    assert json.loads(record["conversations"][1]["value"]) == {"name": "search", "arguments": {"q": "gemma"}}


def test_alpaca_from_instruction_output_row_and_rejects_tool_calls() -> None:
    assert alpaca(PLAIN_ROW) == {"instruction": "Summarize\n\nLong text", "input": "", "output": "Short text"}
    with pytest.raises(SchemaError, match="alpaca_cannot_hold_tool_calls"):
        alpaca(TOOL_ROW)


def test_vertex_gemini_maps_roles_and_function_parts() -> None:
    record = vertex_gemini(TOOL_ROW)
    assert record["systemInstruction"]["parts"][0]["text"] == "Use tools."
    assert [c["role"] for c in record["contents"]] == ["user", "model", "user", "model"]
    assert record["contents"][1]["parts"] == [{"functionCall": {"name": "search", "args": {"q": "gemma"}}}]
    assert record["contents"][2]["parts"][0]["functionResponse"] == {"name": "search", "response": {"result": "Gemma 4 released"}}


def test_dpo_requires_distinct_pair() -> None:
    assert dpo({"prompt": "p", "chosen": "a", "rejected": "b"}) == {"prompt": "p", "chosen": "a", "rejected": "b"}
    with pytest.raises(SchemaError):
        dpo({"prompt": "p", "chosen": "a", "rejected": "a"})


def test_every_schema_handles_or_explains_each_row() -> None:
    for name, convert in SCHEMAS.items():
        for row in (TOOL_ROW, PLAIN_ROW):
            try:
                assert isinstance(convert(row), dict)
            except SchemaError as exc:
                assert str(exc), name


def test_splits_are_deterministic_and_group_aware(tmp_path: Path) -> None:
    splits = parse_splits("train=0.8,val=0.1,test=0.1")
    assert assign_split("doc-a", splits, 0) == assign_split("doc-a", splits, 0)
    rows = [dict(PLAIN_ROW, id=f"r{i}", metadata={"source": f"doc-{i % 7}"}) for i in range(200)]
    manifest = export_dataset(rows, tmp_path, ExportSpec(fmt="jsonl", schema="openai-chat", splits=splits,
                                                         group_by="metadata.source"))
    assert manifest["rows_out"] == 200 and manifest["group_leakage"] == 0
    again = export_dataset(rows, tmp_path / "again", ExportSpec(fmt="jsonl", schema="openai-chat", splits=splits,
                                                                group_by="metadata.source"))
    assert [f["sha256"] for f in manifest["files"]] == [f["sha256"] for f in again["files"]]


def test_sharding_compression_manifest_card_and_llamafactory_info(tmp_path: Path) -> None:
    rows = [dict(TOOL_ROW, id=f"t{i}") for i in range(25)] + [PLAIN_ROW]
    manifest = export_dataset(rows, tmp_path / "gz", ExportSpec(fmt="jsonl", schema="alpaca", shard_size=10,
                                                                compress="gzip", name="demo"))
    assert manifest["dropped"] == {"alpaca_cannot_hold_tool_calls": 25}
    shard = tmp_path / "gz" / manifest["files"][0]["path"]
    assert shard.name.endswith(".jsonl.gz")
    assert json.loads(gzip.decompress(shard.read_bytes()).splitlines()[0])["output"] == "Short text"
    sharded = export_dataset(rows, tmp_path / "sg", ExportSpec(fmt="jsonl", schema="sharegpt", shard_size=10, name="demo"))
    assert [f["rows"] for f in sharded["files"]] == [10, 10, 6]
    assert (tmp_path / "sg" / "README.md").read_text().startswith("---")
    info = json.loads((tmp_path / "sg" / "dataset_info.json").read_text())
    assert info["demo_train"]["formatting"] == "sharegpt"


def test_parquet_export(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    manifest = export_dataset([TOOL_ROW, PLAIN_ROW], tmp_path, ExportSpec(fmt="parquet", schema="chatml"))
    assert manifest["files"][0]["path"].endswith(".parquet") and manifest["rows_out"] == 2


def test_strict_mode_fails_on_unexportable_row(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be exported as alpaca"):
        export_dataset([TOOL_ROW], tmp_path, ExportSpec(schema="alpaca", strict=True))


def test_invalid_spec_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExportSpec(splits={"train": 0.5})
    with pytest.raises(ValueError):
        ExportSpec(fmt="parquet", compress="gzip")


def test_cli_export_engine(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in [TOOL_ROW, PLAIN_ROW]) + "\n")
    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "mlforge", "export", str(source), "--to", "jsonl", "--schema", "openai-chat",
         "--split", "train=0.5,test=0.5", "--out", str(out)],
        text=True, capture_output=True, cwd=ROOT, check=True,
    )
    assert "Export complete" in result.stdout
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["schema"] == "openai-chat" and manifest["rows_out"] == 2


def test_parquet_and_hf_handle_empty_nested_fields(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    from mlforge.datasets.exporters import export_examples

    rows = [{"id": "a", "labels": {}, "metadata": {}, "tools": []}, {"id": "b", "labels": {"x": 1}, "metadata": {}, "tools": []}]
    assert export_examples(rows, tmp_path / "x.parquet", "parquet").exists()


def test_dpo_rejected_is_a_real_negative(tmp_path: Path) -> None:
    from mlforge.workflows.distill import _rejected_for

    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"answer {i}"}]} for i in range(3)]
    rejected, strategy = _rejected_for("answer 0", 1, rows)
    assert (rejected, strategy) == ("answer 1", "mismatched_answer")
    single = [rows[0]]
    rejected, strategy = _rejected_for("First point. Second point. Third point.", 1, single)
    assert strategy == "truncated_answer" and rejected != "First point. Second point. Third point."


def test_cli_version_and_friendly_missing_file() -> None:
    version = subprocess.run([sys.executable, "-m", "mlforge", "--version"], text=True, capture_output=True, cwd=ROOT)
    assert version.stdout.startswith("saara ")
    missing = subprocess.run([sys.executable, "-m", "mlforge", "export", "nope.jsonl", "--to", "jsonl", "--out", "x.jsonl"],
                             text=True, capture_output=True, cwd=ROOT)
    assert missing.returncode == 1 and "file not found: nope.jsonl" in missing.stdout
