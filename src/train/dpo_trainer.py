from __future__ import annotations

from pathlib import Path


class DPOTrainerLite:
    def __init__(self, config: dict):
        self.config = config

    def run(self, preference_rows: list[dict]) -> str:
        out_dir = Path(self.config.get("output_dir", "outputs/raft-dpo"))
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / "TRAINING_SKIPPED.txt"
        marker.write_text(
            "DPO trainer scaffold ready. Integrate TRL DPOTrainer for full training.\n"
            f"rows={len(preference_rows)}\n",
            encoding="utf-8",
        )
        return str(marker)
