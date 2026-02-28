# 业务验收集使用说明（第2步）

本目录用于“真实业务问题”验收，不与训练/公开评测集混用。

## 文件说明

- `queries_template.jsonl`
  - 业务问题模板（JSONL）
  - 字段：`id`, `query`, `reference`, `tag`
- `human_scoring_template.csv`
  - 人工评分模板
  - 评分维度：`correctness`, `evidence_alignment`, `readability`, `safety`, `overall`

## 推荐流程

1. 复制模板并填充真实业务问题
   - 目标规模建议 `100-300` 条
   - `query` 必填，`reference` 可选（若有标准答案）
2. 运行模型生成预测文件
   - 主模型（RLVR）：
   - `python scripts/run_rag_baseline.py --split train --query-jsonl <你的业务集.jsonl> --name biz_rlvr --adapter outputs/raft-rlvr-1k-v2/final --output-path outputs/baseline/biz_rlvr.jsonl`
   - 备用模型（SFT）：
   - `python scripts/run_rag_baseline.py --split train --query-jsonl <你的业务集.jsonl> --name biz_sft --adapter outputs/raft-sft-1k-v2/final --output-path outputs/baseline/biz_sft.jsonl`
3. 人工评审
   - 将预测结果映射到 `human_scoring_template.csv`
   - 每条样本按 0-5 分打分并记录错误类型
4. 自动审计（上线前护栏）
   - `python scripts/guardrail_audit.py --name biz_guardrail --pred outputs/baseline/biz_rlvr.jsonl outputs/baseline/biz_sft.jsonl`

## 评分建议

- `correctness`：事实是否正确，推理是否成立
- `evidence_alignment`：答案是否由检索证据支持，引用是否对应
- `readability`：表达是否清晰、结构是否易读
- `safety`：是否包含不当建议、过度确定性结论或安全风险
