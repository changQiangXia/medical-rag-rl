#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io_utils import write_jsonl
from src.data.dataset_builder import build_chunked_rows
from src.data.pubmed_parser import parse_pubmed_split


def validate_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    seen = set()
    valid: list[dict] = []
    dropped = {"empty": 0, "duplicate": 0}

    for row in rows:
        rid = row.get("id", "")
        text = row.get("text", "").strip()
        if not rid or not text:
            dropped["empty"] += 1
            continue
        if rid in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(rid)
        valid.append(row)

    return valid, dropped


def process_split(split: str, cfg: dict, limit: int = 0) -> dict:
    records = parse_pubmed_split(cfg["raw_dir"], split)
    if limit > 0:
        records = records[:limit]

    rows, stats = build_chunked_rows(
        records,
        chunk_size=int(cfg["chunk_size"]),
        chunk_overlap=int(cfg["chunk_overlap"]),
        min_chars=int(cfg["min_chars"]),
        max_chars=int(cfg["max_chars"]),
    )
    rows, dropped = validate_rows(rows)

    out_dir = Path(cfg["processed_jsonl_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"
    write_jsonl(out_path, rows)

    stats.update(
        {
            "split": split,
            "output_path": str(out_path),
            "dropped": dropped,
        }
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--split", default="all", choices=["all", "train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=0, help="limit records per split for smoke run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    all_stats = {}
    for split in splits:
        stats = process_split(split, cfg, limit=args.limit)
        all_stats[split] = stats
        print(f"[OK] {split}: rows={stats['num_rows']} avg_chunk_words={stats['avg_chunk_words']}")

    stats_dir = ROOT / "artifacts" / "data_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_file = stats_dir / "prepare_pubmed_stats.json"
    stats_file.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] stats written: {stats_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
