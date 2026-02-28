#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/medical-rag-rl"
cd "$ROOT_DIR"

python scripts/guardrail_audit.py \
  --name release_models_guardrail_clean5_all \
  --pred \
    outputs/baseline/sft_1k_v2_clean5_test.jsonl \
    outputs/baseline/rlvr_1k_v2_clean5_test.jsonl \
    outputs/baseline/sft_1k_v2_clean5_max.jsonl \
    outputs/baseline/rlvr_1k_v2_clean5_max.jsonl

python scripts/release_quality_gate.py \
  --name release_quality_gate_clean5_all_v1 \
  --model sft_1k_v2_clean5_test artifacts/metrics/sft_1k_v2_clean5_test.json outputs/baseline/sft_1k_v2_clean5_test.jsonl \
  --model rlvr_1k_v2_clean5_test artifacts/metrics/rlvr_1k_v2_clean5_test.json outputs/baseline/rlvr_1k_v2_clean5_test.jsonl \
  --model sft_1k_v2_clean5_max artifacts/metrics/sft_1k_v2_clean5_max.json outputs/baseline/sft_1k_v2_clean5_max.jsonl \
  --model rlvr_1k_v2_clean5_max artifacts/metrics/rlvr_1k_v2_clean5_max.json outputs/baseline/rlvr_1k_v2_clean5_max.jsonl \
  --require-token-f1 \
  --max-doc-repeat 0.05 \
  --max-no-citation-ratio 0.45 \
  --max-prompt-residue-ratio 0.02

echo "[DONE] release checks completed"
