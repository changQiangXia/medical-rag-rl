from __future__ import annotations

from src.common.io_utils import read_jsonl


def load_preference_rows(path: str) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        if all(k in row for k in ("prompt", "chosen", "rejected")):
            rows.append(row)
    return rows
