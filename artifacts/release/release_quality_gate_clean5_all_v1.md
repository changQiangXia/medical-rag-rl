# 发布质量门禁报告：release_quality_gate_clean5_all_v1

- 生成时间：2026-02-28T10:08:33
- 模型数：4
- 通过模型数：4
- 全量通过：YES

## 阈值

- `min_citation`: `0.6`
- `min_evidence`: `0.65`
- `min_token_f1`: `0.5`
- `max_doc_repeat`: `0.05`
- `max_safety_refusal`: `0.05`
- `max_no_citation_ratio`: `0.45`
- `max_prompt_residue_ratio`: `0.02`
- `max_incomplete_tail_ratio`: `0.0`
- `min_human_overall`: `3.0`
- `require_token_f1`: `True`

## 模型结论

| 模型 | 结果 | 通过项/总项 |
| --- | --- | ---: |
| sft_1k_v2_clean5_test | PASS | 12/12 |
| rlvr_1k_v2_clean5_test | PASS | 12/12 |
| sft_1k_v2_clean5_max | PASS | 12/12 |
| rlvr_1k_v2_clean5_max | PASS | 12/12 |

### sft_1k_v2_clean5_test

- 指标文件：`artifacts/metrics/sft_1k_v2_clean5_test.json`
- 预测文件：`outputs/baseline/sft_1k_v2_clean5_test.jsonl`

| 检查项 | 观测值 | 阈值 | 通过 | 备注 |
| --- | ---: | --- | --- | --- |
| avg_citation_consistency | 0.8 | >= 0.6 | PASS |  |
| avg_evidence_hit_rate | 0.9 | >= 0.65 | PASS |  |
| avg_doc_tag_repeat_ratio | 0.0 | <= 0.05 | PASS |  |
| avg_safety_refusal_rate | 0.0 | <= 0.05 | PASS |  |
| avg_token_f1 | 0.5555 | >= 0.5 | PASS |  |
| no_citation_ratio | 0.1 | <= 0.45 | PASS |  |
| empty_response_count | 0.0 | == 0.0 | PASS |  |
| chat_marker_count | 0.0 | == 0.0 | PASS |  |
| malformed_doc_tag_count | 0.0 | == 0.0 | PASS |  |
| prompt_residue_ratio | 0.0 | <= 0.02 | PASS |  |
| incomplete_doc_tail_ratio | 0.0 | <= 0.0 | PASS |  |
| human_overall_0_5 | NA | >= 3.0 | PASS | human score not provided for this model |

### rlvr_1k_v2_clean5_test

- 指标文件：`artifacts/metrics/rlvr_1k_v2_clean5_test.json`
- 预测文件：`outputs/baseline/rlvr_1k_v2_clean5_test.jsonl`

| 检查项 | 观测值 | 阈值 | 通过 | 备注 |
| --- | ---: | --- | --- | --- |
| avg_citation_consistency | 0.8 | >= 0.6 | PASS |  |
| avg_evidence_hit_rate | 0.8 | >= 0.65 | PASS |  |
| avg_doc_tag_repeat_ratio | 0.0 | <= 0.05 | PASS |  |
| avg_safety_refusal_rate | 0.0 | <= 0.05 | PASS |  |
| avg_token_f1 | 0.5086 | >= 0.5 | PASS |  |
| no_citation_ratio | 0.2 | <= 0.45 | PASS |  |
| empty_response_count | 0.0 | == 0.0 | PASS |  |
| chat_marker_count | 0.0 | == 0.0 | PASS |  |
| malformed_doc_tag_count | 0.0 | == 0.0 | PASS |  |
| prompt_residue_ratio | 0.0 | <= 0.02 | PASS |  |
| incomplete_doc_tail_ratio | 0.0 | <= 0.0 | PASS |  |
| human_overall_0_5 | NA | >= 3.0 | PASS | human score not provided for this model |

### sft_1k_v2_clean5_max

- 指标文件：`artifacts/metrics/sft_1k_v2_clean5_max.json`
- 预测文件：`outputs/baseline/sft_1k_v2_clean5_max.jsonl`

| 检查项 | 观测值 | 阈值 | 通过 | 备注 |
| --- | ---: | --- | --- | --- |
| avg_citation_consistency | 0.6067 | >= 0.6 | PASS |  |
| avg_evidence_hit_rate | 0.65 | >= 0.65 | PASS |  |
| avg_doc_tag_repeat_ratio | 0.0067 | <= 0.05 | PASS |  |
| avg_safety_refusal_rate | 0.02 | <= 0.05 | PASS |  |
| avg_token_f1 | 0.5435 | >= 0.5 | PASS |  |
| no_citation_ratio | 0.34 | <= 0.45 | PASS |  |
| empty_response_count | 0.0 | == 0.0 | PASS |  |
| chat_marker_count | 0.0 | == 0.0 | PASS |  |
| malformed_doc_tag_count | 0.0 | == 0.0 | PASS |  |
| prompt_residue_ratio | 0.0 | <= 0.02 | PASS |  |
| incomplete_doc_tail_ratio | 0.0 | <= 0.0 | PASS |  |
| human_overall_0_5 | NA | >= 3.0 | PASS | human score not provided for this model |

### rlvr_1k_v2_clean5_max

- 指标文件：`artifacts/metrics/rlvr_1k_v2_clean5_max.json`
- 预测文件：`outputs/baseline/rlvr_1k_v2_clean5_max.jsonl`

| 检查项 | 观测值 | 阈值 | 通过 | 备注 |
| --- | ---: | --- | --- | --- |
| avg_citation_consistency | 0.788 | >= 0.6 | PASS |  |
| avg_evidence_hit_rate | 0.8 | >= 0.65 | PASS |  |
| avg_doc_tag_repeat_ratio | 0.024 | <= 0.05 | PASS |  |
| avg_safety_refusal_rate | 0.0 | <= 0.05 | PASS |  |
| avg_token_f1 | 0.5087 | >= 0.5 | PASS |  |
| no_citation_ratio | 0.2 | <= 0.45 | PASS |  |
| empty_response_count | 0.0 | == 0.0 | PASS |  |
| chat_marker_count | 0.0 | == 0.0 | PASS |  |
| malformed_doc_tag_count | 0.0 | == 0.0 | PASS |  |
| prompt_residue_ratio | 0.0 | <= 0.02 | PASS |  |
| incomplete_doc_tail_ratio | 0.0 | <= 0.0 | PASS |  |
| human_overall_0_5 | NA | >= 3.0 | PASS | human score not provided for this model |

## 结论

- 发布门禁通过：可以进入下一步交付流程。
