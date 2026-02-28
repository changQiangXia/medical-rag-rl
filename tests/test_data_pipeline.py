from __future__ import annotations

from pathlib import Path

from src.data.pubmed_parser import parse_pubmed_split


def test_pubmed_parser_basic(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "train.txt").write_text(
        "###1\n"
        "OBJECTIVE\tA test objective.\n"
        "METHODS\tA test method.\n\n",
        encoding="utf-8",
    )

    rows = parse_pubmed_split(raw, "train")
    assert len(rows) == 1
    assert rows[0].paper_id == "1"
    assert len(rows[0].sentences) == 2
