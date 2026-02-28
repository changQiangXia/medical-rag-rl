from __future__ import annotations


def compute_rlvr_reward(
    evidence_hit: float,
    groundedness: float,
    citation: float,
    safety: float,
    repetition_penalty: float = 0.0,
    weights: dict | None = None,
) -> float:
    if weights is None:
        weights = {
            "evidence_hit": 0.35,
            "groundedness": 0.25,
            "citation": 0.20,
            "safety": 0.20,
            "repetition_penalty": 0.15,
        }
    reward = (
        weights["evidence_hit"] * evidence_hit
        + weights["groundedness"] * groundedness
        + weights["citation"] * citation
        + weights["safety"] * safety
        - weights.get("repetition_penalty", 0.15) * repetition_penalty
    )
    return max(-1.0, min(1.0, reward))
