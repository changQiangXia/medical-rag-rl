#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config


def fmt(ok: bool) -> str:
    return "[OK]" if ok else "[FAIL]"


def check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    return ok, f"Python {v.major}.{v.minor}.{v.micro}"


def check_torch() -> tuple[bool, str]:
    try:
        import torch

        cuda = torch.cuda.is_available()
        msg = f"torch={torch.__version__}, cuda_available={cuda}"
        if cuda:
            msg += f", gpu={torch.cuda.get_device_name(0)}"
        return True, msg
    except Exception as e:
        return False, f"torch import failed: {e}"


def check_packages(strict_train: bool = False) -> tuple[bool, list[str]]:
    required = {
        "pyyaml": "6.0",
        "numpy": "1.26",
    }
    optional = [
        "pandas",
        "tqdm",
        "transformers",
        "peft",
        "trl",
        "accelerate",
        "sentence-transformers",
        "faiss-cpu",
    ]

    ok = True
    lines: list[str] = []

    for pkg, expected_prefix in required.items():
        try:
            v = md.version(pkg)
            matched = v.startswith(expected_prefix)
            ok = ok and matched
            lines.append(f"{fmt(matched)} required {pkg}={v} (expected {expected_prefix}.x)")
        except Exception:
            ok = False
            lines.append(f"[FAIL] required package missing: {pkg}")

    for pkg in optional:
        try:
            v = md.version(pkg)
            lines.append(f"[WARN] optional {pkg}={v}")
        except Exception:
            lines.append(f"[WARN] optional package missing: {pkg}")

    if strict_train:
        train_required = ["transformers", "peft", "accelerate", "bitsandbytes"]
        for pkg in train_required:
            try:
                v = md.version(pkg)
                lines.append(f"[OK] train required {pkg}={v}")
            except Exception:
                ok = False
                lines.append(f"[FAIL] train required package missing: {pkg}")

    return ok, lines


def check_paths() -> tuple[bool, list[str]]:
    data_cfg = load_config(ROOT / "configs" / "data.yaml")
    required_dirs = [
        ROOT / "configs",
        ROOT / "scripts",
        ROOT / "src",
        ROOT / "tests",
        ROOT / "data",
        Path(data_cfg["raw_dir"]),
        Path(data_cfg["processed_markdown_dir"]),
        Path(data_cfg["synthetic_bootstrap_dir"]),
    ]
    lines = []
    ok = True
    for d in required_dirs:
        exists = d.exists()
        ok = ok and exists
        lines.append(f"{fmt(exists)} path: {d}")
    return ok, lines


def check_api_env() -> tuple[bool, str]:
    enabled = os.getenv("API_ENABLED", "false").lower() == "true"
    if not enabled:
        return True, "API disabled by default (expected)."

    provider = os.getenv("API_PROVIDER", "zhipu")
    if provider == "zhipu":
        ok = bool(os.getenv("ZHIPU_API_KEY"))
        return ok, "ZHIPU_API_KEY found" if ok else "ZHIPU_API_KEY missing"

    if provider == "openai":
        ok = bool(os.getenv("OPENAI_API_KEY"))
        return ok, "OPENAI_API_KEY found" if ok else "OPENAI_API_KEY missing"

    return False, f"Unsupported API_PROVIDER: {provider}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-train",
        action="store_true",
        help="require training dependencies (transformers/peft/accelerate/bitsandbytes)",
    )
    args = parser.parse_args()

    print("== medical-rag-rl environment check ==")
    print(f"Platform: {platform.platform()}")
    print(f"Project root: {ROOT}")

    ok_py, msg_py = check_python()
    print(f"{fmt(ok_py)} {msg_py}")

    ok_torch, msg_torch = check_torch()
    print(f"{fmt(ok_torch)} {msg_torch}")

    ok_pkg, pkg_lines = check_packages(strict_train=args.strict_train)
    for line in pkg_lines:
        print(line)

    ok_paths, path_lines = check_paths()
    for line in path_lines:
        print(line)

    ok_api, msg_api = check_api_env()
    print(f"{fmt(ok_api)} {msg_api}")

    all_ok = ok_py and ok_torch and ok_pkg and ok_paths and ok_api
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
