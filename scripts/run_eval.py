#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io_utils import read_jsonl
from src.eval.generation_metrics import token_f1
from src.eval.medical_grounding_metrics import (
    citation_consistency,
    doc_tag_repeat_ratio,
    evidence_hit_rate,
    safety_refusal_rate,
)
from src.eval.report import write_markdown_report, write_metrics_json


ANSWER_SPLIT_RE = re.compile(r"\$Answer\$\s*[:：]?\s*", re.IGNORECASE)


def extract_reference_answer(reference: str) -> str:
    ref = (reference or "").strip()
    if not ref:
        return ""
    if "$Answer$" not in ref:
        return ref
    parts = ANSWER_SPLIT_RE.split(ref, maxsplit=1)
    if len(parts) >= 2:
        return parts[1].strip()
    return ref


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", default=str(ROOT / "outputs" / "baseline" / "rag_baseline_train.jsonl"))
    parser.add_argument("--name", default="base_rag")
    args = parser.parse_args()

    pred_path = Path(args.pred_path)
    if not pred_path.exists():
        print(f"Missing predictions: {pred_path}")
        return 1

    rows = list(read_jsonl(pred_path))
    if not rows:
        print("Empty predictions file")
        return 1

    citation_scores = []
    evidence_scores = []
    safety_scores = []
    f1_scores = []
    response_lens = []
    doc_repeat_scores = []

    pbar = tqdm(rows, total=len(rows), desc=f"Metric Eval {args.name}", unit="sample")
    for row in pbar:
        resp = row.get("response", "")
        docs = row.get("docs", [])
        ref = extract_reference_answer(row.get("reference", ""))

        citation_scores.append(citation_consistency(resp, len(docs)))
        evidence_scores.append(evidence_hit_rate(resp, docs))
        safety_scores.append(safety_refusal_rate(resp))
        response_lens.append(len(resp.split()))
        doc_repeat_scores.append(doc_tag_repeat_ratio(resp))

        if ref:
            f1_scores.append(token_f1(resp, ref))
    pbar.close()

    metrics = {
        "num_samples": len(rows),
        "avg_response_words": round(statistics.mean(response_lens), 4),
        "avg_citation_consistency": round(statistics.mean(citation_scores), 4),
        "avg_evidence_hit_rate": round(statistics.mean(evidence_scores), 4),
        "avg_safety_refusal_rate": round(statistics.mean(safety_scores), 4),
        "avg_doc_tag_repeat_ratio": round(statistics.mean(doc_repeat_scores), 4),
    }
    if f1_scores:
        metrics["avg_token_f1"] = round(statistics.mean(f1_scores), 4)

    out_json = ROOT / "artifacts" / "metrics" / f"{args.name}.json"
    out_md = ROOT / "artifacts" / "reports" / f"eval_{args.name}.md"
    write_metrics_json(metrics, out_json)
    write_markdown_report(f"Evaluation: {args.name}", metrics, out_md)

    print(f"[OK] metrics -> {out_json}")
    print(f"[OK] report  -> {out_md}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
