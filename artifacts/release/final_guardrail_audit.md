# 护栏审计报告：final_guardrail_audit

- 生成时间：2026-02-28T07:56:30
- 总文件数：8
- 通过文件数：4

| 文件 | 样本数 | 空回复 | 对话残片 | 引用标签异常 | 全角标签 | 无引用 | 过长回复 | 通过 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| outputs/baseline/base_test.jsonl | 10 | 0 | 0 | 0 | 0 | 10 | 0 | PASS |
| outputs/baseline/sft_1k_v2_test.jsonl | 10 | 0 | 0 | 0 | 0 | 1 | 0 | PASS |
| outputs/baseline/dpo_1k_v2_clean3_test.jsonl | 10 | 0 | 0 | 0 | 0 | 3 | 0 | PASS |
| outputs/baseline/rlvr_1k_v2_clean3_test.jsonl | 10 | 0 | 0 | 0 | 0 | 2 | 0 | PASS |
| outputs/baseline/base_max.jsonl | 100 | 0 | 0 | 2 | 0 | 98 | 0 | FAIL |
| outputs/baseline/sft_1k_v2_max.jsonl | 100 | 0 | 6 | 3 | 0 | 34 | 0 | FAIL |
| outputs/baseline/dpo_1k_v2_clean3_max.jsonl | 100 | 0 | 0 | 1 | 0 | 27 | 1 | FAIL |
| outputs/baseline/rlvr_1k_v2_clean3_max.jsonl | 100 | 0 | 0 | 4 | 0 | 21 | 0 | FAIL |

## 判定规则

- `PASS` 条件：`空回复=0` 且 `对话残片=0` 且 `引用标签异常=0`。
- `无引用`、`过长回复` 为提示项，不单独判 FAIL，可按业务要求追加阈值。
