#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/medical-rag-rl"
mkdir -p "$ROOT"/{configs,scripts,src,tests,outputs,logs,artifacts}
mkdir -p "$ROOT/data"/{raw,processed,synthetic,preference,eval,synthetic_bootstrap}
mkdir -p "$ROOT/data/processed/pubmed"
mkdir -p "$ROOT/models"

echo "Directory bootstrap complete: $ROOT"
