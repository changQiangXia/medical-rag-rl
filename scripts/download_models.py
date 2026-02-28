#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def download_from_modelscope(models: list[str], cache_dir: str) -> bool:
    try:
        from modelscope import snapshot_download

        for m in models:
            print(f"[ModelScope] downloading {m} ...")
            snapshot_download(m, cache_dir=cache_dir)
        return True
    except Exception as e:
        print(f"ModelScope failed: {e}")
        return False


def download_from_hf(models: list[str], cache_dir: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        for m in models:
            print(f"[HF fallback] downloading {m} ...")
            snapshot_download(repo_id=m, local_dir=cache_dir)
        return True
    except Exception as e:
        print(f"HF fallback failed: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/medical-rag-rl/models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen2-7B-Instruct", "BAAI/bge-large-en-v1.5"],
    )
    args = parser.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    ok = download_from_modelscope(args.models, args.cache_dir)
    if not ok:
        ok = download_from_hf(args.models, args.cache_dir)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
