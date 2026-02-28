from __future__ import annotations

from src.common.io_utils import read_jsonl


def load_sft_rows(path: str) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        if all(k in row for k in ("instruction", "output")):
            rows.append(row)
    return rows
