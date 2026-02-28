from __future__ import annotations

from pathlib import Path


class SFTTrainerLite:
    def __init__(self, config: dict):
        self.config = config

    def run(self, train_rows: list[dict]) -> str:
        out_dir = Path(self.config.get("output_dir", "outputs/raft-sft"))
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / "TRAINING_SKIPPED.txt"
        marker.write_text(
            "SFT trainer scaffold ready. Integrate TRL SFTTrainer for full training.\n"
            f"rows={len(train_rows)}\n",
            encoding="utf-8",
        )
        return str(marker)
