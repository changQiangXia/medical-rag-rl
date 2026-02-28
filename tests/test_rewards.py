from __future__ import annotations

from src.train.rlvr_reward import compute_rlvr_reward


def test_reward_weighted_sum() -> None:
    r = compute_rlvr_reward(1.0, 0.5, 0.5, 1.0)
    assert abs(r - 0.775) < 1e-6
