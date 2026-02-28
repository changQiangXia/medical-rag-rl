#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/medical-rag-rl"
cd "$ROOT"

python scripts/check_env.py --strict-train
python scripts/verify_data.py
python scripts/prepare_pubmed.py --split all
python scripts/build_index.py --split train
python scripts/run_rag_baseline.py --split train
python scripts/build_sft_data.py --split all
python scripts/train_sft.py
python scripts/build_preference_data.py
python scripts/train_dpo.py
python scripts/train_rlvr.py
python scripts/run_eval.py --pred-path "$ROOT/outputs/baseline/rag_baseline_train.jsonl" --name "full_pipeline_placeholder"

echo "Full staged pipeline (with scaffold training steps) complete."
