#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="${TMPDIR:-/tmp}/saara-smoke-$$"
mkdir -p "$TMP_DIR"

cd "$ROOT_DIR"

SAARA="${SAARA:-.venv/bin/saara}"
if [[ ! -x "$SAARA" ]]; then
  SAARA="python -m mlforge"
fi

$SAARA >/tmp/saara-smoke-help.txt
$SAARA splash --no-animation >/tmp/saara-smoke-splash.txt
printf '6\n' | $SAARA wizard >/tmp/saara-smoke-wizard.txt
$SAARA init --root "$TMP_DIR/workspace"
$SAARA models health --provider mock
$SAARA tools firecrawl-health --base-url http://127.0.0.1:9
$SAARA generate topic "release smoke" --samples 2 --provider mock --out "$TMP_DIR/dataset.jsonl" --root "$TMP_DIR/workspace"
$SAARA validate "$TMP_DIR/dataset.jsonl" --report "$TMP_DIR/report.json"
$SAARA export "$TMP_DIR/dataset.jsonl" --to json --out "$TMP_DIR/dataset.json"
$SAARA label "$TMP_DIR/dataset.jsonl" --labels useful,not-useful --provider mock --out "$TMP_DIR/labeled.jsonl"
$SAARA distill "$TMP_DIR/dataset.jsonl" --method dpo --provider mock --out "$TMP_DIR/dpo.jsonl"

cat > "$TMP_DIR/workflow.json" <<JSON
{
  "kind": "topic-dataset",
  "topic": "release config",
  "samples": 1,
  "provider": {"name": "mock"},
  "output": {"format": "jsonl", "path": "$TMP_DIR/workflow.jsonl"}
}
JSON

$SAARA run "$TMP_DIR/workflow.json" --root "$TMP_DIR/workspace"

echo "Saara smoke test passed: $TMP_DIR"
