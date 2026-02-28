from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SentenceRecord:
    section: str
    text: str


@dataclass
class AbstractRecord:
    paper_id: str
    split: str
    sentences: list[SentenceRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["abstract_text"] = " ".join(s.text for s in self.sentences).strip()
        return d
