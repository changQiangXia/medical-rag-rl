from __future__ import annotations

from pathlib import Path

from src.data.schema import AbstractRecord, SentenceRecord


SPLIT_FILES = {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}


def parse_pubmed_split(raw_dir: str | Path, split: str) -> list[AbstractRecord]:
    raw_dir = Path(raw_dir)
    if split not in SPLIT_FILES:
        raise ValueError(f"Unsupported split: {split}")

    file_path = raw_dir / SPLIT_FILES[split]
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    records: list[AbstractRecord] = []
    current: AbstractRecord | None = None

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if current is not None and current.sentences:
                    records.append(current)
                current = None
                continue

            if line.startswith("###"):
                paper_id = line[3:].strip()
                if current is not None and current.sentences:
                    records.append(current)
                current = AbstractRecord(paper_id=paper_id, split=split)
                continue

            if current is None:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            section, text = parts[0].strip(), parts[1].strip()
            if not text:
                continue
            current.sentences.append(SentenceRecord(section=section, text=text))

    if current is not None and current.sentences:
        records.append(current)

    return records


def parse_pubmed_all(raw_dir: str | Path, splits: list[str] | None = None) -> dict[str, list[AbstractRecord]]:
    if splits is None:
        splits = ["train", "dev", "test"]
    return {split: parse_pubmed_split(raw_dir, split) for split in splits}
