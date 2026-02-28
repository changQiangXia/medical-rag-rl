#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io_utils import read_jsonl
from tqdm import tqdm


INSTRUCTION_TEMPLATES = [
    "Summarize the clinical trial evidence and provide the main efficacy takeaway using the given document.",
    "Based only on the retrieved document, describe study design, key result, and practical medical implication.",
    "Write a concise evidence-grounded answer from the document and include one citation marker [Doc 1].",
    "Explain what this study suggests for clinical decision-making, strictly grounded in the provided text.",
]

UPSAMPLE_HINTS_ZH = [
    "请保持结论简洁，并给出必要引用。",
    "请只根据证据回答，避免无依据推断。",
    "请优先给出临床可执行结论。",
    "请在证据不足时明确说明。",
]

ANSWER_SPLIT_RE = re.compile(r"\$Answer\$\s*[:：]?\s*", re.IGNORECASE)
DOC_MARKER_RE = re.compile(r"Document\s*(\d+)\s*:", re.IGNORECASE)


def clip_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def extract_answer(output_text: str) -> str:
    text = (output_text or "").strip()
    if not text:
        return ""
    if "$Answer$" not in text:
        return text
    parts = ANSWER_SPLIT_RE.split(text, maxsplit=1)
    if len(parts) >= 2 and parts[1].strip():
        return parts[1].strip()
    return text


def normalize_doc_markers(input_text: str) -> str:
    return DOC_MARKER_RE.sub(r"[Doc \1]", input_text or "")


def clean_bootstrap_row(row: dict, split: str, keep_cot: bool = False) -> dict:
    instruction = (row.get("instruction") or "").strip()
    input_text = normalize_doc_markers(row.get("input", ""))
    output_text = (row.get("output") or "").strip()
    if not keep_cot:
        output_text = extract_answer(output_text)
    if output_text and "[Doc" not in output_text:
        output_text = f"{output_text} [Doc 1]"
    if not instruction or not output_text:
        return {}
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": row.get("metadata", {}) | {"source_split": split, "source": "bootstrap_clean"},
    }


def build_pubmed_row(src_row: dict, rnd: random.Random) -> dict:
    text = src_row.get("text", "").strip()
    if not text:
        return {}

    instruction = INSTRUCTION_TEMPLATES[rnd.randint(0, len(INSTRUCTION_TEMPLATES) - 1)]
    input_text = f"Document 1: {clip_words(text, 220)}"
    output_text = (
        "Evidence-based summary: "
        + clip_words(text, 120)
        + " [Doc 1]"
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": {
            "source": "pubmed_train_jsonl",
            "paper_id": src_row.get("paper_id", ""),
            "chunk_id": src_row.get("chunk_id", -1),
            "sample_id": src_row.get("id", ""),
        },
    }


def load_bootstrap_rows(synth_dir: Path, split: str) -> list[dict]:
    splits = ["train", "val", "test"] if split == "all" else [split]
    rows: list[dict] = []
    for sp in splits:
        p = synth_dir / f"{sp}.jsonl"
        if not p.exists():
            continue
        for row in tqdm(read_jsonl(p), desc=f"bootstrap:{sp}", unit="sample"):
            built = clean_bootstrap_row(row, split=sp, keep_cot=False)
            if built:
                rows.append(built)
    return rows


def load_pubmed_rows(data_cfg: dict, seed: int) -> list[dict]:
    src_path = Path(data_cfg["processed_jsonl_dir"]) / "train.jsonl"
    if not src_path.exists():
        raise FileNotFoundError(f"Missing processed pubmed JSONL: {src_path}. Run prepare_pubmed.py first.")

    rnd = random.Random(seed)
    rows: list[dict] = []
    for src in tqdm(read_jsonl(src_path), desc="pubmed:train", unit="chunk"):
        built = build_pubmed_row(src, rnd)
        if not built:
            continue
        rows.append(built)
    return rows


def dedup_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen = set()
    for row in rows:
        key = (
            row.get("instruction", "").strip(),
            row.get("input", "").strip(),
            row.get("output", "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def upsample_rows(rows: list[dict], target: int, rnd: random.Random) -> list[dict]:
    if target <= 0 or len(rows) >= target or not rows:
        return rows

    out = list(rows)
    i = 0
    while len(out) < target:
        base = rows[i % len(rows)]
        hint = UPSAMPLE_HINTS_ZH[i % len(UPSAMPLE_HINTS_ZH)]
        extra = {
            "instruction": f"{base.get('instruction', '').strip()}\n\n{hint}",
            "input": base.get("input", ""),
            "output": base.get("output", ""),
            "metadata": dict(base.get("metadata", {}))
            | {"augmented": True, "aug_idx": len(out), "aug_seed": rnd.randint(0, 10_000_000)},
        }
        out.append(extra)
        i += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--source", default="pubmed", choices=["bootstrap", "pubmed", "mixed"])
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upsample", action="store_true")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    synth_dir = Path(data_cfg["synthetic_bootstrap_dir"])
    if not synth_dir.exists() and args.source in ("bootstrap", "mixed"):
        print(f"Missing synthetic bootstrap dir: {synth_dir}")
        return 1

    rows: list[dict] = []
    if args.source in ("bootstrap", "mixed"):
        rows.extend(load_bootstrap_rows(synth_dir, args.split))
    if args.source in ("pubmed", "mixed"):
        rows.extend(load_pubmed_rows(data_cfg, seed=args.seed))

    rows = dedup_rows(rows)
    rnd = random.Random(args.seed)
    rnd.shuffle(rows)
    if args.max_samples > 0 and args.upsample:
        rows = upsample_rows(rows, target=args.max_samples, rnd=rnd)
    if args.max_samples > 0 and len(rows) > args.max_samples:
        rows = rows[: args.max_samples]

    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sft_train.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="write:sft_train", unit="sample"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "source": args.source,
        "rows": len(rows),
        "split": args.split,
        "seed": args.seed,
        "upsample": bool(args.upsample),
        "output_path": str(out_path),
    }
    stats_path = ROOT / "artifacts" / "data_stats" / "sft_data_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] SFT data rows={len(rows)} -> {out_path}")
    print(f"[OK] stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
