from __future__ import annotations

from pathlib import Path


class RLVRTrainerLite:
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> str:
        out_dir = Path(self.config.get("output_dir", "outputs/raft-rlvr"))
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / "TRAINING_SKIPPED.txt"
        marker.write_text(
            "RLVR scaffold ready. Integrate PPO/RLOO loop for full training.\n",
            encoding="utf-8",
        )
        return str(marker)
