# 工程化与质控强化报告（2026-02-28）

## 1. 本轮改造内容

### 1.1 生成后清洗强化（`src/rag/generator.py`）

新增/强化了以下规则：

- 规范化异常引用（全角/下标数字、链接形式引用）
- 清除提示词残留（如 `请总结...`、`Answer in Chinese...`）
- 清除尾部不完整引用标签（如 `...[Doc`）
- 清除异常签名噪声（如 `(... 科学校验)`）
- 去重重复段落并统一空白字符

### 1.2 护栏审计增强（`scripts/guardrail_audit.py`）

新增风险项检测：

- `prompt_residue_count`
- `incomplete_doc_tail_count`

并将其纳入 `PASS` 判定。

### 1.3 统一质量门禁（`scripts/release_quality_gate.py`）

新增“一键发布门禁”脚本，统一检查：

- 自动指标：citation / evidence / token_f1 / doc_repeat / safety
- 护栏指标：空回复、对话残片、异常引用、提示词残留、不完整尾巴
- 可选人工评分：`overall_0_5`

输出标准化 JSON + Markdown 报告，给出每项阈值与 PASS/FAIL。

### 1.4 回归测试补充

新增测试文件：

- `tests/test_generator_cleanup.py`
- `tests/test_quality_gate.py`

并修复 `tests/test_rag_pipeline.py` 的 DummyGenerator 兼容问题。

---

## 2. 实验与评估（前后对比）

## 2.1 业务验收集（20条）

- 对比对象：`pre_harden` vs `harden_v3`

### 2.1.1 护栏审计对比

- `pre_harden`: `passed_files=0/2`, `prompt_residue=5`, `incomplete_tail=3`
  - 文件：`artifacts/release/biz_real_guardrail_pre_harden_v2.json`
- `harden_v3`: `passed_files=2/2`, `prompt_residue=0`, `incomplete_tail=0`
  - 文件：`artifacts/release/biz_real_guardrail_harden_v3.json`

### 2.1.2 质量门禁对比

- `pre_harden`: `passed_models=0/2`
  - 文件：`artifacts/release/biz_quality_gate_pre_harden.json`
- `harden_v3`: `passed_models=2/2`
  - 文件：`artifacts/release/biz_quality_gate_harden_v3.json`

结论：业务场景在不改模型权重的前提下，通过工程化后处理+门禁规则，已从“不满足发布标准”提升到“全量通过”。

---

## 2.2 发布评测集（eval_max=100）

- 对比对象：`clean4` vs `clean5`（SFT / RLVR）

### 2.2.1 指标对比

- SFT：
  - token_f1: `0.5419 -> 0.5435`
  - citation: `0.6017 -> 0.6067`
  - evidence_hit: `0.6500 -> 0.6500`
  - doc_repeat: `0.0067 -> 0.0067`
- RLVR：
  - token_f1: `0.5085 -> 0.5087`
  - citation: `0.7880 -> 0.7880`
  - evidence_hit: `0.8000 -> 0.8000`
  - doc_repeat: `0.0240 -> 0.0240`

对应文件：

- `artifacts/metrics/sft_1k_v2_clean4_max.json`
- `artifacts/metrics/sft_1k_v2_clean5_max.json`
- `artifacts/metrics/rlvr_1k_v2_clean4_max.json`
- `artifacts/metrics/rlvr_1k_v2_clean5_max.json`

### 2.2.2 质量门禁对比

- `clean4`: `passed_models=0/2`
  - 文件：`artifacts/release/release_quality_gate_clean4_v2.json`
- `clean5`: `passed_models=4/4`
  - 文件：`artifacts/release/release_quality_gate_clean5_all_v1.json`
- `clean5` 护栏审计（test+max）：`passed_files=4/4`
  - 文件：`artifacts/release/release_models_guardrail_clean5_all.json`

结论：clean5 在基本指标不下降的情况下，显著提升了发布可用性（门禁由失败转为通过）。

---

## 3. 当前建议发布版本

- 主发布：`rlvr_1k_v2_clean5`（证据对齐优先）
- 回退发布：`sft_1k_v2_clean5`（语义贴近优先）

对应输出：

- `outputs/baseline/rlvr_1k_v2_clean5_max.jsonl`
- `outputs/baseline/sft_1k_v2_clean5_max.jsonl`

---

## 4. 仍需关注的风险

- `no_citation_ratio` 仍偏高（尤其 SFT），虽在当前门限内，但建议后续专项优化
- 个别答案仍存在中英混杂表达，可继续做风格统一后处理
- 若转正式生产，建议补充在线监控（实时失败告警、回退切换、异常请求审计）
