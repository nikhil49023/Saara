# Export engine

`saara export` turns any Saara dataset into the files a training tool expects.

```bash
saara export INPUT --to FORMAT [--schema SCHEMA] [--split SPEC] [--group-by FIELD]
             [--shard-size N] [--compress gzip] [--name NAME] [--seed N] [--strict] --out DIR
```

Input can be `.jsonl`, `.json` (a list, or an object with an `examples` list), `.csv` or `.tsv`.

## Schemas

| `--schema` | Record shape | Typical consumer |
|---|---|---|
| `openai-chat` | `{"messages": [...], "tools": [...]}`; tool-call arguments are JSON **strings** | OpenAI fine-tuning, Together, Fireworks, Azure OpenAI, Unsloth, TRL `SFTTrainer` |
| `sharegpt` | `{"conversations": [{"from": "human"/"gpt"/"function_call"/"observation", "value": ...}], "system", "tools", "images"}` | LLaMA-Factory, Axolotl |
| `alpaca` | `{"instruction", "input", "output", "system"?, "history"?}` | Alpaca-style instruction tuners |
| `chatml` | `{"text": "<\|im_start\|>user\n...<\|im_end\|>..."}` | Raw-text SFT trainers |
| `vertex-gemini` | `{"systemInstruction", "contents": [{"role": "user"/"model", "parts": [...]}]}` with `functionCall` / `functionResponse` parts | Vertex AI / Gemini supervised tuning |
| `dpo` | `{"prompt", "chosen", "rejected"}` | TRL `DPOTrainer` and other preference trainers |
| `text` | `{"text": ...}` | Pretraining, continued pretraining |
| `saara` | Unchanged native rows | Saara itself, custom pipelines |

Saara normalizes every row to chat messages first, so all of these work from any row shape: `messages`, or
`input`/`output` dicts with `instruction`/`output` fields, plus optional `tools` and `tool_calls`.

### Rows a schema cannot hold
Some combinations are impossible, for example tool calls in `alpaca`, or a `dpo` row whose `chosen` equals
`rejected`. Those rows are **dropped and counted by reason** in `export_manifest.json` and the dataset card.
With `--strict` the export fails on the first such row instead.

## Storage formats
`jsonl`, `json`, `csv`, `tsv`, `parquet`, `arrow`, `hf` (Hugging Face `save_to_disk`). Parquet, Arrow and HF need
`pip install 'saara-ai[data]'`. Nested values that Arrow cannot type consistently are stored as JSON strings.
`--compress gzip` applies to `jsonl`, `json`, `csv` and `tsv`.

## Splits
`--split train=0.9,val=0.05,test=0.05` assigns each row by a seeded hash, so the same input always produces
the same split on every machine. `--group-by metadata.source` (any dot path) hashes the group instead of the
row ID, keeping every row from one document, source or conversation in a single split. The manifest reports
`group_leakage`: the number of groups found in more than one split, which should be 0.

## Output layout

```text
out/
  train/<name>-00000-of-00003.jsonl.gz
  train/<name>-00001-of-00003.jsonl.gz
  ...
  test/<name>-00000-of-00001.jsonl.gz
  export_manifest.json   # schema, format, splits, counts, dropped reasons, sha256 per file
  README.md              # Hugging Face dataset card (configs point at the split folders)
  dataset_info.json      # LLaMA-Factory registration (sharegpt / alpaca, uncompressed jsonl/json)
```

The output folder can be pushed to the Hugging Face Hub as-is (`huggingface-cli upload ...`), and loaded with
`datasets.load_dataset("path/to/out")`.

## Backwards compatibility
Without `--schema`, `--split`, `--shard-size` or `--compress`, `saara export INPUT --to FORMAT --out FILE`
writes a single file exactly as in earlier versions.
