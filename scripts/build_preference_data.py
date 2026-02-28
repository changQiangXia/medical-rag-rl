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

from src.common.io_utils import read_jsonl


ANSWER_SPLIT_RE = re.compile(r"\$Answer\$\s*[:：]?\s*", re.IGNORECASE)
DOC_RE = re.compile(r"\[Doc\s*(\d+)\]", re.IGNORECASE)
DOC_MARKER_RE = re.compile(r"Document\s*(\d+)\s*:", re.IGNORECASE)


def extract_answer(text: str) -> str:
    text = (text or "").strip()
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


def build_prompt(instruction: str, input_text: str) -> str:
    return (
        "You are a medical assistant. Answer using only retrieved evidence.\n"
        "If evidence is insufficient, explicitly state that evidence is insufficient.\n\n"
        f"Question:\n{instruction.strip()}\n\n"
        f"Retrieved Evidence:\n{normalize_doc_markers(input_text).strip()}\n\n"
        "Answer:\n"
    )


def make_rejected(answer: str, rnd: random.Random) -> str:
    base = answer.strip()
    no_cite = DOC_RE.sub("", base).strip()
    wrong_cite = DOC_RE.sub("[Doc 99]", base)
    cite_spam = (no_cite + " " + " ".join(["[Doc 1]", "[Doc 2]", "[Doc 3]", "[Doc 4]", "[Doc 5]"] * 4)).strip()
    refusal = "证据不足，无法给出可靠结论。"
    truncated = (base[: max(60, int(len(base) * 0.45))] + " ...").strip() if len(base) > 80 else base

    candidates = [x for x in [no_cite, wrong_cite, cite_spam, refusal, truncated] if x and x != base]
    if not candidates:
        return "证据不足，无法给出可靠结论。"
    return candidates[rnd.randint(0, len(candidates) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", default=str(ROOT / "data" / "synthetic" / "sft_train.jsonl"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sft_path = Path(args.sft_data)
    if not sft_path.exists():
        print(f"Missing SFT data: {sft_path}")
        return 1

    rnd = random.Random(args.seed)
    rows = []
    for row in read_jsonl(sft_path):
        instruction = row.get("instruction", "")
        input_text = row.get("input", "")
        prompt = build_prompt(instruction, input_text)

        chosen = extract_answer(row.get("output", ""))
        if not prompt or not chosen:
            continue

        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": make_rejected(chosen, rnd=rnd),
                "metadata": row.get("metadata", {}),
            }
        )
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break

    out_dir = ROOT / "data" / "preference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dpo_train.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] preference rows={len(rows)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
