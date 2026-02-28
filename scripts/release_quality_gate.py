#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.guardrail_audit import analyze_prediction_file


@dataclass
class Thresholds:
    min_citation: float
    min_evidence: float
    min_token_f1: float
    max_doc_repeat: float
    max_safety_refusal: float
    max_no_citation_ratio: float
    max_prompt_residue_ratio: float
    max_incomplete_tail_ratio: float
    min_human_overall: float
    require_token_f1: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="发布质量门禁：聚合指标、护栏与人工评分，输出统一 PASS/FAIL 报告。"
    )
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("NAME", "METRIC_JSON", "PRED_JSONL"),
        required=True,
        help="模型条目，可重复传入。",
    )
    parser.add_argument("--name", default="release_quality_gate")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "release"))

    parser.add_argument("--min-citation", type=float, default=0.60)
    parser.add_argument("--min-evidence", type=float, default=0.65)
    parser.add_argument("--min-token-f1", type=float, default=0.50)
    parser.add_argument("--max-doc-repeat", type=float, default=0.05)
    parser.add_argument("--max-safety-refusal", type=float, default=0.05)
    parser.add_argument("--max-no-citation-ratio", type=float, default=0.25)
    parser.add_argument("--max-prompt-residue-ratio", type=float, default=0.02)
    parser.add_argument("--max-incomplete-tail-ratio", type=float, default=0.0)
    parser.add_argument("--require-token-f1", action="store_true")

    parser.add_argument("--human-score-csv", default="")
    parser.add_argument("--min-human-overall", type=float, default=3.00)
    parser.add_argument(
        "--human-alias",
        nargs=2,
        action="append",
        metavar=("MODEL_NAME", "CSV_MODEL_NAME"),
        help="当 model 名与人工评分文件 model_name 不一致时使用。",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_human_scores(path: Path) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    if not path.exists():
        raise FileNotFoundError(f"missing human score csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = str(row.get("model_name", "") or "").strip()
            if not model:
                continue
            try:
                overall = float(row.get("overall_0_5", ""))
            except (TypeError, ValueError):
                continue
            scores.setdefault(model, []).append(overall)

    return {k: round(sum(v) / max(1, len(v)), 4) for k, v in scores.items()}


def check_thresholds(
    model_name: str,
    metric: dict[str, Any],
    guardrail: dict[str, Any],
    human_overall: float | None,
    th: Thresholds,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(field: str, observed: float | None, op: str, target: float, pass_flag: bool, note: str = "") -> None:
        checks.append(
            {
                "field": field,
                "observed": observed,
                "operator": op,
                "target": target,
                "pass": pass_flag,
                "note": note,
            }
        )

    citation = float(metric.get("avg_citation_consistency", 0.0))
    add_check("avg_citation_consistency", citation, ">=", th.min_citation, citation >= th.min_citation)

    evidence = float(metric.get("avg_evidence_hit_rate", 0.0))
    add_check("avg_evidence_hit_rate", evidence, ">=", th.min_evidence, evidence >= th.min_evidence)

    doc_repeat = float(metric.get("avg_doc_tag_repeat_ratio", 0.0))
    add_check("avg_doc_tag_repeat_ratio", doc_repeat, "<=", th.max_doc_repeat, doc_repeat <= th.max_doc_repeat)

    safety = float(metric.get("avg_safety_refusal_rate", 0.0))
    add_check("avg_safety_refusal_rate", safety, "<=", th.max_safety_refusal, safety <= th.max_safety_refusal)

    token_f1 = metric.get("avg_token_f1")
    if token_f1 is None:
        add_check(
            "avg_token_f1",
            None,
            ">=",
            th.min_token_f1,
            not th.require_token_f1,
            "missing avg_token_f1 in metric json",
        )
    else:
        token_f1_v = float(token_f1)
        add_check("avg_token_f1", token_f1_v, ">=", th.min_token_f1, token_f1_v >= th.min_token_f1)

    num_samples = max(1, int(guardrail.get("num_samples", 0)))
    no_citation_ratio = float(guardrail.get("no_citation_count", 0)) / num_samples
    add_check(
        "no_citation_ratio",
        round(no_citation_ratio, 4),
        "<=",
        th.max_no_citation_ratio,
        no_citation_ratio <= th.max_no_citation_ratio,
    )

    add_check(
        "empty_response_count",
        float(guardrail.get("empty_response_count", 0)),
        "==",
        0.0,
        int(guardrail.get("empty_response_count", 0)) == 0,
    )
    add_check(
        "chat_marker_count",
        float(guardrail.get("chat_marker_count", 0)),
        "==",
        0.0,
        int(guardrail.get("chat_marker_count", 0)) == 0,
    )
    add_check(
        "malformed_doc_tag_count",
        float(guardrail.get("malformed_doc_tag_count", 0)),
        "==",
        0.0,
        int(guardrail.get("malformed_doc_tag_count", 0)) == 0,
    )
    prompt_residue_ratio = float(guardrail.get("prompt_residue_count", 0)) / num_samples
    add_check(
        "prompt_residue_ratio",
        round(prompt_residue_ratio, 4),
        "<=",
        th.max_prompt_residue_ratio,
        prompt_residue_ratio <= th.max_prompt_residue_ratio,
    )
    incomplete_tail_ratio = float(guardrail.get("incomplete_doc_tail_count", 0)) / num_samples
    add_check(
        "incomplete_doc_tail_ratio",
        round(incomplete_tail_ratio, 4),
        "<=",
        th.max_incomplete_tail_ratio,
        incomplete_tail_ratio <= th.max_incomplete_tail_ratio,
    )

    if human_overall is None:
        add_check(
            "human_overall_0_5",
            None,
            ">=",
            th.min_human_overall,
            True,
            "human score not provided for this model",
        )
    else:
        add_check("human_overall_0_5", human_overall, ">=", th.min_human_overall, human_overall >= th.min_human_overall)

    passed = all(c["pass"] for c in checks)
    return {
        "model": model_name,
        "pass": passed,
        "checks": checks,
        "metric_path": metric.get("__metric_path__", ""),
        "pred_path": guardrail.get("file", ""),
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# 发布质量门禁报告：{report['name']}")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- 模型数：{len(report['models'])}")
    lines.append(f"- 通过模型数：{report['summary']['passed_models']}")
    lines.append(f"- 全量通过：{'YES' if report['summary']['all_passed'] else 'NO'}")
    lines.append("")

    lines.append("## 阈值")
    lines.append("")
    for k, v in report["thresholds"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    lines.append("## 模型结论")
    lines.append("")
    lines.append("| 模型 | 结果 | 通过项/总项 |")
    lines.append("| --- | --- | ---: |")
    for model in report["models"]:
        pass_n = sum(1 for c in model["checks"] if c["pass"])
        total_n = len(model["checks"])
        lines.append(f"| {model['model']} | {'PASS' if model['pass'] else 'FAIL'} | {pass_n}/{total_n} |")

    for model in report["models"]:
        lines.append("")
        lines.append(f"### {model['model']}")
        lines.append("")
        lines.append(f"- 指标文件：`{model['metric_path']}`")
        lines.append(f"- 预测文件：`{model['pred_path']}`")
        lines.append("")
        lines.append("| 检查项 | 观测值 | 阈值 | 通过 | 备注 |")
        lines.append("| --- | ---: | --- | --- | --- |")
        for c in model["checks"]:
            observed = "NA" if c["observed"] is None else c["observed"]
            lines.append(
                f"| {c['field']} | {observed} | {c['operator']} {c['target']} | {'PASS' if c['pass'] else 'FAIL'} | {c.get('note','')} |"
            )

    lines.append("")
    lines.append("## 结论")
    lines.append("")
    if report["summary"]["all_passed"]:
        lines.append("- 发布门禁通过：可以进入下一步交付流程。")
    else:
        lines.append("- 发布门禁未通过：请先修复 FAIL 项后再发布。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = Thresholds(
        min_citation=float(args.min_citation),
        min_evidence=float(args.min_evidence),
        min_token_f1=float(args.min_token_f1),
        max_doc_repeat=float(args.max_doc_repeat),
        max_safety_refusal=float(args.max_safety_refusal),
        max_no_citation_ratio=float(args.max_no_citation_ratio),
        max_prompt_residue_ratio=float(args.max_prompt_residue_ratio),
        max_incomplete_tail_ratio=float(args.max_incomplete_tail_ratio),
        min_human_overall=float(args.min_human_overall),
        require_token_f1=bool(args.require_token_f1),
    )

    alias_map: dict[str, str] = {}
    for pair in args.human_alias or []:
        alias_map[pair[0]] = pair[1]

    human_score_map: dict[str, float] = {}
    if args.human_score_csv:
        human_score_map = load_human_scores(Path(args.human_score_csv))

    model_reports: list[dict[str, Any]] = []
    for model_name, metric_path_str, pred_path_str in args.model:
        metric_path = Path(metric_path_str)
        pred_path = Path(pred_path_str)

        metric = load_json(metric_path)
        metric["__metric_path__"] = str(metric_path)
        guardrail = analyze_prediction_file(pred_path)

        human_key = alias_map.get(model_name, model_name)
        human_overall = human_score_map.get(human_key)

        model_reports.append(check_thresholds(model_name, metric, guardrail, human_overall, thresholds))

    passed_models = sum(1 for m in model_reports if m["pass"])
    report = {
        "name": args.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "thresholds": {
            "min_citation": thresholds.min_citation,
            "min_evidence": thresholds.min_evidence,
            "min_token_f1": thresholds.min_token_f1,
            "max_doc_repeat": thresholds.max_doc_repeat,
            "max_safety_refusal": thresholds.max_safety_refusal,
            "max_no_citation_ratio": thresholds.max_no_citation_ratio,
            "max_prompt_residue_ratio": thresholds.max_prompt_residue_ratio,
            "max_incomplete_tail_ratio": thresholds.max_incomplete_tail_ratio,
            "min_human_overall": thresholds.min_human_overall,
            "require_token_f1": thresholds.require_token_f1,
        },
        "summary": {
            "total_models": len(model_reports),
            "passed_models": passed_models,
            "all_passed": passed_models == len(model_reports),
        },
        "models": model_reports,
    }

    json_path = output_dir / f"{args.name}.json"
    md_path = output_dir / f"{args.name}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print(f"[OK] quality gate json -> {json_path}")
    print(f"[OK] quality gate md   -> {md_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
