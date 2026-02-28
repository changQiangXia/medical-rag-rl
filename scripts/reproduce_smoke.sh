#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/medical-rag-rl"
cd "$ROOT"

python scripts/check_env.py
python scripts/verify_data.py
python scripts/prepare_pubmed.py --split train --limit 200
python scripts/build_index.py --split train --limit 2000
python scripts/run_rag_baseline.py --split train --query "What did the prednisolone trial report for knee osteoarthritis?"
python scripts/run_eval.py --pred-path "$ROOT/outputs/baseline/rag_baseline_train.jsonl" --name "base_rag_smoke"

echo "Smoke reproduction complete."
