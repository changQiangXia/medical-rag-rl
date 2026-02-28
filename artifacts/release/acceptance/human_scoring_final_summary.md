# 人工打分汇总（Codex代评）

- 样本总数：40（每模型 20 条）
- 评分文件：`/root/autodl-tmp/medical-rag-rl/artifacts/release/acceptance/human_scoring_final.csv`

| 模型 | correctness | evidence_alignment | readability | safety | overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| biz_real_rlvr | 3.20 | 3.10 | 2.85 | 5.00 | 3.25 |
| biz_real_sft | 3.30 | 3.10 | 2.75 | 5.00 | 3.10 |

## 主要问题类型计数
- biz_real_rlvr:
  - none: 4
  - garbled_chars: 2
  - minor_language_noise: 2
  - focus_drift_hallucination: 1
  - format_noise: 1
  - light_hallucination: 1
  - mild_overclaim: 1
  - minor_fact_blur: 1
  - minor_term_error: 1
  - minor_verbosity: 1
  - possible_misinterpretation: 1
  - possible_wrong_claim: 1
  - prompt_residue: 1
  - prompt_residue_repetition: 1
  - truncated_output: 1
- biz_real_sft:
  - none: 9
  - prompt_residue: 3
  - truncated_output: 2
  - format_break: 1
  - light_hallucination: 1
  - minor_generalization: 1
  - minor_noise: 1
  - prompt_residue_repetition: 1
  - severe_truncation: 1
