#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REQUIRED = {
    "raw": Path("data/raw/PubMed_20k_RCT"),
    "processed": Path("data/processed/pubmed/pubmed_20k_rct"),
    "synthetic": Path("data/synthetic_bootstrap/synthetic"),
}


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/autodl-tmp/medical-rag-rl")
    parser.add_argument("--sha-json", default="")
    args = parser.parse_args()

    root = Path(args.root)
    ok = True

    print("== verify data layout ==")
    for name, rel in REQUIRED.items():
        p = root / rel
        exists = p.exists() and p.is_dir()
        print(f"[{ 'OK' if exists else 'FAIL' }] {name}: {p}")
        ok = ok and exists

    if args.sha_json:
        sha_map = json.loads(Path(args.sha_json).read_text(encoding="utf-8"))
        print("== verify sha256 ==")
        for rel, expected in sha_map.items():
            fp = root / rel
            if not fp.exists() or not fp.is_file():
                print(f"[FAIL] missing file for sha check: {fp}")
                ok = False
                continue
            actual = sha256_of_file(fp)
            matched = actual.lower() == expected.lower()
            print(f"[{ 'OK' if matched else 'FAIL' }] {rel}")
            ok = ok and matched

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
