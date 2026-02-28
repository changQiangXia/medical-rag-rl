#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io_utils import read_jsonl


CHAT_MARKER_RE = re.compile(r"(?:Human|Assistant|User|System)\s*[:：]", re.IGNORECASE)
DOC_ANY_RE = re.compile(r"\[[^\]]*Doc[^\]]*\]", re.IGNORECASE)
DOC_VALID_RE = re.compile(r"\[Doc\s*\d+\]", re.IGNORECASE)
DOC_FULLWIDTH_RE = re.compile(r"\[Doc\s*[０-９]+\]", re.IGNORECASE)
PROMPT_RESIDUE_RE = re.compile(
    r"(?:^|[\s(（])(?:请总结(?:这篇文献)?|请用中文回答|请总结回答以上问题|Please summarize|Answer in Chinese)",
    re.IGNORECASE,
)
INCOMPLETE_DOC_TAIL_RE = re.compile(r"\[\s*Doc\s*[0-9０-９]*\s*$", re.IGNORECASE)


def analyze_prediction_file(path: Path) -> dict:
    result = {
        "file": str(path),
        "exists": path.exists(),
        "num_samples": 0,
        "empty_response_count": 0,
        "chat_marker_count": 0,
        "malformed_doc_tag_count": 0,
        "fullwidth_doc_tag_count": 0,
        "no_citation_count": 0,
        "too_long_response_count": 0,
        "prompt_residue_count": 0,
        "incomplete_doc_tail_count": 0,
        "avg_response_chars": 0.0,
        "pass": False,
    }
    if not path.exists():
        return result

    rows = list(read_jsonl(path))
    result["num_samples"] = len(rows)
    if not rows:
        return result

    total_chars = 0
    for row in rows:
        resp = str(row.get("response", "") or "")
        total_chars += len(resp)

        if not resp.strip():
            result["empty_response_count"] += 1
            continue

        if CHAT_MARKER_RE.search(resp):
            result["chat_marker_count"] += 1

        tags = DOC_ANY_RE.findall(resp)
        for tag in tags:
            if DOC_FULLWIDTH_RE.search(tag):
                result["fullwidth_doc_tag_count"] += 1
            if not DOC_VALID_RE.fullmatch(tag.strip()):
                result["malformed_doc_tag_count"] += 1

        valid_cites = DOC_VALID_RE.findall(resp)
        if not valid_cites:
            result["no_citation_count"] += 1

        if len(resp) > 800:
            result["too_long_response_count"] += 1

        if PROMPT_RESIDUE_RE.search(resp):
            result["prompt_residue_count"] += 1

        if INCOMPLETE_DOC_TAIL_RE.search(resp):
            result["incomplete_doc_tail_count"] += 1

    result["avg_response_chars"] = round(total_chars / max(1, len(rows)), 2)
    result["pass"] = (
        result["empty_response_count"] == 0
        and result["chat_marker_count"] == 0
        and result["malformed_doc_tag_count"] == 0
        and result["prompt_residue_count"] == 0
        and result["incomplete_doc_tail_count"] == 0
    )
    return result


def write_markdown(report: dict, out_path: Path) -> None:
    lines = [
        f"# 护栏审计报告：{report['name']}",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 总文件数：{len(report['files'])}",
        f"- 通过文件数：{report['summary']['passed_files']}",
        "",
        "| 文件 | 样本数 | 空回复 | 对话残片 | 引用标签异常 | 全角标签 | 无引用 | 过长回复 | 提示词残留 | 不完整引用尾巴 | 通过 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["files"]:
        lines.append(
            "| {file} | {num_samples} | {empty_response_count} | {chat_marker_count} | "
            "{malformed_doc_tag_count} | {fullwidth_doc_tag_count} | {no_citation_count} | "
            "{too_long_response_count} | {prompt_residue_count} | {incomplete_doc_tail_count} | {pass_flag} |".format(
                file=item["file"],
                num_samples=item["num_samples"],
                empty_response_count=item["empty_response_count"],
                chat_marker_count=item["chat_marker_count"],
                malformed_doc_tag_count=item["malformed_doc_tag_count"],
                fullwidth_doc_tag_count=item["fullwidth_doc_tag_count"],
                no_citation_count=item["no_citation_count"],
                too_long_response_count=item["too_long_response_count"],
                prompt_residue_count=item["prompt_residue_count"],
                incomplete_doc_tail_count=item["incomplete_doc_tail_count"],
                pass_flag="PASS" if item["pass"] else "FAIL",
            )
        )

    lines.extend(
        [
            "",
            "## 判定规则",
            "",
            "- `PASS` 条件：`空回复=0` 且 `对话残片=0` 且 `引用标签异常=0` 且 `提示词残留=0` 且 `不完整引用尾巴=0`。",
            "- `无引用`、`过长回复` 为提示项，不单独判 FAIL，可按业务要求追加阈值。",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", nargs="+", required=True, help="预测 jsonl 文件路径列表")
    parser.add_argument("--name", default="release_guardrail")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "release"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_reports = [analyze_prediction_file(Path(p)) for p in args.pred]
    summary = {
        "total_files": len(file_reports),
        "passed_files": sum(1 for x in file_reports if x["pass"]),
        "total_samples": sum(int(x["num_samples"]) for x in file_reports),
        "total_chat_markers": sum(int(x["chat_marker_count"]) for x in file_reports),
        "total_malformed_doc_tags": sum(int(x["malformed_doc_tag_count"]) for x in file_reports),
        "total_empty_responses": sum(int(x["empty_response_count"]) for x in file_reports),
        "total_prompt_residues": sum(int(x["prompt_residue_count"]) for x in file_reports),
        "total_incomplete_doc_tails": sum(int(x["incomplete_doc_tail_count"]) for x in file_reports),
    }
    generated_at = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    report = {
        "name": args.name,
        "generated_at": generated_at,
        "summary": summary,
        "files": file_reports,
    }

    json_path = output_dir / f"{args.name}.json"
    md_path = output_dir / f"{args.name}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(f"[OK] guardrail json -> {json_path}")
    print(f"[OK] guardrail md   -> {md_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
