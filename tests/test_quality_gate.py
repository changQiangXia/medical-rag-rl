from __future__ import annotations

from scripts.release_quality_gate import Thresholds, check_thresholds


def test_quality_gate_thresholds_pass() -> None:
    metric = {
        "avg_citation_consistency": 0.7,
        "avg_evidence_hit_rate": 0.8,
        "avg_doc_tag_repeat_ratio": 0.02,
        "avg_safety_refusal_rate": 0.0,
        "avg_token_f1": 0.55,
        "__metric_path__": "m.json",
    }
    guardrail = {
        "file": "p.jsonl",
        "num_samples": 20,
        "no_citation_count": 2,
        "empty_response_count": 0,
        "chat_marker_count": 0,
        "malformed_doc_tag_count": 0,
    }
    th = Thresholds(
        min_citation=0.6,
        min_evidence=0.65,
        min_token_f1=0.5,
        max_doc_repeat=0.05,
        max_safety_refusal=0.05,
        max_no_citation_ratio=0.25,
        max_prompt_residue_ratio=0.02,
        max_incomplete_tail_ratio=0.0,
        min_human_overall=3.0,
        require_token_f1=True,
    )
    rep = check_thresholds("x", metric, guardrail, human_overall=3.2, th=th)
    assert rep["pass"] is True


def test_quality_gate_missing_token_f1_fails_when_required() -> None:
    metric = {
        "avg_citation_consistency": 0.7,
        "avg_evidence_hit_rate": 0.8,
        "avg_doc_tag_repeat_ratio": 0.02,
        "avg_safety_refusal_rate": 0.0,
        "__metric_path__": "m.json",
    }
    guardrail = {
        "file": "p.jsonl",
        "num_samples": 20,
        "no_citation_count": 2,
        "empty_response_count": 0,
        "chat_marker_count": 0,
        "malformed_doc_tag_count": 0,
    }
    th = Thresholds(
        min_citation=0.6,
        min_evidence=0.65,
        min_token_f1=0.5,
        max_doc_repeat=0.05,
        max_safety_refusal=0.05,
        max_no_citation_ratio=0.25,
        max_prompt_residue_ratio=0.02,
        max_incomplete_tail_ratio=0.0,
        min_human_overall=3.0,
        require_token_f1=True,
    )
    rep = check_thresholds("x", metric, guardrail, human_overall=3.2, th=th)
    assert rep["pass"] is False
