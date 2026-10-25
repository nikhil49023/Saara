#!/usr/bin/env bash
# saara-gemma end-to-end on one GPU box: setup -> parallel prefetch -> data -> tests -> pilot -> train -> eval -> report.
# Usage: bash run.sh [stage]   stages: setup prefetch data test pilot train eval report all (default: all)
set -euo pipefail
cd "$(dirname "$0")"
STAGE="${1:-all}"
VERSION="${VERSION:-v0.1}"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
DATA="${DATA:-data}"
RUN="runs/$VERSION"
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DOWNLOAD_TIMEOUT=60 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p logs

run() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

if run setup; then
  pip install -q -U -r requirements.txt
  pip install -q -U vllm || echo "vLLM install failed: evaluate.py will need --engine hf"
fi

if run prefetch; then
  # Everything prepare_data.py and train.py need, downloaded concurrently into the HF cache.
  hf download "$MODEL" --max-workers 16 > logs/prefetch_model.log 2>&1 &
  hf download allenai/olmOCR-mix-1025 --repo-type dataset --max-workers 32 \
    --include "*.parquet" "pdf_tarballs/*_eval_*" \
              "pdf_tarballs/00_documents_train_00000.tar.gz" "pdf_tarballs/01_books_train_00000.tar.gz" \
              "pdf_tarballs/02_loc_transcripts_train_0000[01].tar.gz" "pdf_tarballs/03_national_archives_train_0000[0-3].tar.gz" \
    > logs/prefetch_ocr.log 2>&1 &
  hf download Agent-Ark/Toucan-1.5M --repo-type dataset --include "SFT/train-00000-of-00003.parquet" > logs/prefetch_toucan.log 2>&1 &
  hf download NousResearch/hermes-function-calling-v1 --repo-type dataset > logs/prefetch_hermes.log 2>&1 &
  hf download nvidia/When2Call --repo-type dataset > logs/prefetch_w2c.log 2>&1 &
  wait
  echo "prefetch done"
fi

if run data; then
  python prepare_data.py --out "$DATA" 2>&1 | tee logs/prepare_data.log
fi

if run test; then
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_pipeline.py -q -p no:cacheprovider 2>&1 | tee logs/tests.log
fi

if run pilot; then
  # ~5% of data, one short epoch: measures tokens/s + memory and checks the loss goes down.
  python train.py --data "$DATA" --out runs/pilot --model "$MODEL" --pilot 0.05 --epochs 1 --eval-steps 25 \
    --val-task-rows 24 2>&1 | tee logs/pilot.log
fi

if run train; then
  python train.py --data "$DATA" --out "$RUN" --model "$MODEL" 2>&1 | tee logs/train.log
fi

if run eval; then
  ENGINE="$(python -c 'import vllm' 2>/dev/null && echo vllm || echo hf)"
  python evaluate.py --data "$DATA" --model "$MODEL" --name base --out "results/$VERSION" --engine "$ENGINE" 2>&1 | tee logs/eval_base.log
  python evaluate.py --data "$DATA" --model "$RUN/merged" --name saara-gemma --out "results/$VERSION" --engine "$ENGINE" 2>&1 | tee logs/eval_ft.log
fi

if run report; then
  python report.py --data "$DATA" --run "$RUN" --results "results/$VERSION" --out "release/$VERSION" --version "$VERSION" 2>&1 | tee logs/report.log
fi
