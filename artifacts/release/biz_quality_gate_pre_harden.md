# 发布质量门禁报告：biz_quality_gate_pre_harden

- 生成时间：2026-02-28T09:33:37
- 模型数：2
- 通过模型数：0
- 全量通过：NO

## 阈值

- `min_citation`: `0.6`
- `min_evidence`: `0.65`
- `min_token_f1`: `0.5`
- `max_doc_repeat`: `0.1`
- `max_safety_refusal`: `0.05`
- `max_no_citation_ratio`: `0.25`
- `min_human_overall`: `3.0`
- `require_token_f1`: `False`

## 模型结论

| 模型 | 结果 | 通过项/总项 |
| --- | --- | ---: |
| biz_real_rlvr_pre_harden | FAIL | 10/12 |
| biz_real_sft_pre_harden | FAIL | 11/12 |

### biz_real_rlvr_pre_harden

- 指标文件：`artifacts/metrics/biz_real_rlvr_pre_harden.json`
- 预测文件：`outputs/baseline/biz_real_rlvr_pre_harden.jsonl`

| 检查项 | 观测值 | 阈值 | 通过 | 备注 |
| --- | ---: | --- | --- | --- |
| avg_citation_consistency | 0.8052 | >= 0.6 | PASS |  |
| avg_evidence_hit_rate | 0.85 | >= 0.65 | PASS |  |
| avg_doc_tag_repeat_ratio | 0.0896 | <= 0.1 | PASS |  |
| avg_safety_refusal_rate | 0.0 | <= 0.05 | PASS |  |
| avg_token_f1 | NA | >= 0.5 | PASS | missing avg_token_f1 in metric json |
| no_citation_ratio | 0.15 | <= 0.25 | PASS |  |
| empty_response_count | 0.0 | == 0.0 | PASS |  |
| chat_marker_count | 0.0 | == 0.0 | PASS |  |
| malformed_doc_tag_count | 0.0 | == 0.0 | PASS |  |
| prompt_residue_count | 1.0 | == 0.0 | FAIL |  |
| incomplete_doc_tail_count | 3.0 | == 0.0 | FAIL |  |
| human_overall_0_5 | 3.25 | >= 3.0 | PASS |  |

### biz_real_sft_pre_harden

- 指标文件：`artifacts/metrics/biz_real_sft_pre_harden.json`
- 预测文件：`outputs/baseline/biz_real_sft_pre_harden.jsonl`

| 检查项 | 观测值 | 阈值 | 通过 | 备注 |
| --- | ---: | --- | --- | --- |
| avg_citation_consistency | 0.7375 | >= 0.6 | PASS |  |
| avg_evidence_hit_rate | 0.85 | >= 0.65 | PASS |  |
| avg_doc_tag_repeat_ratio | 0.025 | <= 0.1 | PASS |  |
| avg_safety_refusal_rate | 0.0 | <= 0.05 | PASS |  |
| avg_token_f1 | NA | >= 0.5 | PASS | missing avg_token_f1 in metric json |
| no_citation_ratio | 0.15 | <= 0.25 | PASS |  |
| empty_response_count | 0.0 | == 0.0 | PASS |  |
| chat_marker_count | 0.0 | == 0.0 | PASS |  |
| malformed_doc_tag_count | 0.0 | == 0.0 | PASS |  |
| prompt_residue_count | 4.0 | == 0.0 | FAIL |  |
| incomplete_doc_tail_count | 0.0 | == 0.0 | PASS |  |
| human_overall_0_5 | 3.1 | >= 3.0 | PASS |  |

## 结论

- 发布门禁未通过：请先修复 FAIL 项后再发布。
