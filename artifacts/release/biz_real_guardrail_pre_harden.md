# 护栏审计报告：biz_real_guardrail_pre_harden

- 生成时间：2026-02-28T09:21:39
- 总文件数：2
- 通过文件数：2

| 文件 | 样本数 | 空回复 | 对话残片 | 引用标签异常 | 全角标签 | 无引用 | 过长回复 | 通过 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| outputs/baseline/biz_real_rlvr_pre_harden.jsonl | 20 | 0 | 0 | 0 | 0 | 3 | 0 | PASS |
| outputs/baseline/biz_real_sft_pre_harden.jsonl | 20 | 0 | 0 | 0 | 0 | 3 | 0 | PASS |

## 判定规则

- `PASS` 条件：`空回复=0` 且 `对话残片=0` 且 `引用标签异常=0`。
- `无引用`、`过长回复` 为提示项，不单独判 FAIL，可按业务要求追加阈值。
