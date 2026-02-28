# 发布定版说明（Freeze）

- 定版日期：2026-02-28
- 版本标识：`v1.1-rlvr-1k-v2-clean4`
- 发布目标：不再继续提分，进入可交付与上线前验收阶段

## 1. 定版模型

- 主发布模型（更重证据可追溯）：
  - 名称：`rlvr_1k_v2_clean4`
  - 适配器路径：`/root/autodl-tmp/medical-rag-rl/outputs/raft-rlvr-1k-v2/final`
- 备用模型（更重与参考答案措辞一致）：
  - 名称：`sft_1k_v2_clean4`
  - 适配器路径：`/root/autodl-tmp/medical-rag-rl/outputs/raft-sft-1k-v2/final`

## 2. 固定依赖与配置

- 基座模型：`/root/autodl-tmp/medical-rag-rl/models/Qwen/Qwen2-7B-Instruct`
- 检索索引：`/root/autodl-tmp/medical-rag-rl/artifacts/index/train`
- 检索嵌入：`/root/autodl-tmp/medical-rag-rl/models/BAAI/bge-large-en-v1___5`
- RLVR 关键配置：
  - `batch_size=1`
  - `gradient_accumulation_steps=16`
  - `max_new_tokens=128`
  - `kl_coef=0.01`

## 3. 固定评测结果

### 3.1 eval_max（100）

- `base_max`：token_f1=0.4506, citation=0.0200, evidence_hit=0.0200
- `sft_1k_v2_clean4_max`：token_f1=0.5419, citation=0.6017, evidence_hit=0.6500
- `dpo_1k_v2_clean4_max`：token_f1=0.5039, citation=0.7400, evidence_hit=0.7400
- `rlvr_1k_v2_clean4_max`：token_f1=0.5085, citation=0.7880, evidence_hit=0.8000

### 3.2 eval_test（10）

- `base_test`：token_f1=0.4828, citation=0.0000, evidence_hit=0.0000
- `sft_1k_v2_clean4_test`：token_f1=0.5523, citation=0.8000, evidence_hit=0.9000
- `dpo_1k_v2_clean4_test`：token_f1=0.5241, citation=0.7000, evidence_hit=0.7000
- `rlvr_1k_v2_clean4_test`：token_f1=0.5086, citation=0.8000, evidence_hit=0.8000

## 4. 发布策略

- 默认上线：`rlvr_1k_v2_clean4`（证据一致性与可追溯优先）
- 回退策略：若业务场景更强调“贴参考答案措辞”，可切到 `sft_1k_v2_clean4`
- 禁止项：冻结后不再修改训练超参、奖励权重、提示词模板
