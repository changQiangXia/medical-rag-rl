# 发布定版说明（Freeze v1.2）

- 定版日期：2026-02-28
- 版本标识：`v1.2-rlvr-1k-v2-clean5`
- 发布目标：在不改模型权重前提下完成工程化与质控强化，形成可发布门禁闭环

## 1. 定版模型

- 主发布模型（证据可追溯优先）：
  - 名称：`rlvr_1k_v2_clean5`
  - 适配器路径：`/root/autodl-tmp/medical-rag-rl/outputs/raft-rlvr-1k-v2/final`
- 备用模型（更重与参考答案措辞一致）：
  - 名称：`sft_1k_v2_clean5`
  - 适配器路径：`/root/autodl-tmp/medical-rag-rl/outputs/raft-sft-1k-v2/final`

## 2. 固定评测结果

### 2.1 eval_max（100）

- `sft_1k_v2_clean5_max`：token_f1=0.5435, citation=0.6067, evidence_hit=0.6500
- `rlvr_1k_v2_clean5_max`：token_f1=0.5087, citation=0.7880, evidence_hit=0.8000

### 2.2 eval_test（10）

- `sft_1k_v2_clean5_test`：token_f1=0.5555, citation=0.8000, evidence_hit=0.9000
- `rlvr_1k_v2_clean5_test`：token_f1=0.5086, citation=0.8000, evidence_hit=0.8000

## 3. 质量门禁结果

- `release_quality_gate_clean4_v2`：`0/2` 通过（历史版本）
- `release_quality_gate_clean5_all_v1`：`4/4` 通过（当前版本）
- `release_models_guardrail_clean5_all`：`4/4` 文件通过，`prompt_residue=0`，`incomplete_tail=0`

关键文件：

- `artifacts/release/release_quality_gate_clean4_v2.json`
- `artifacts/release/release_quality_gate_clean5_all_v1.json`
- `artifacts/release/release_models_guardrail_clean5_all.json`
- `artifacts/release/engineering_qc_upgrade_2026-02-28.md`

## 4. 强化内容摘要

- 生成后清洗：新增 prompt 残留/不完整引用尾巴治理
- 护栏审计：新增 `prompt_residue_count` 与 `incomplete_doc_tail_count`
- 发布门禁：新增 `release_quality_gate.py` 统一阈值判定

## 5. 发布策略

- 默认上线：`rlvr_1k_v2_clean5`
- 回退策略：`sft_1k_v2_clean5`
- 禁止项：冻结后不再修改训练超参与奖励权重；仅允许非语义破坏性的后处理规则优化
