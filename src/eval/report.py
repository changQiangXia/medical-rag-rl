from __future__ import annotations

import json
from pathlib import Path


def write_metrics_json(metrics: dict, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def write_markdown_report(title: str, metrics: dict, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "| Metric | Value |", "|---|---:|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
