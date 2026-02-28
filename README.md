# medical-rag-rl（中文版总手册）

单卡 RTX 4090 的医疗 RAG + SFT + DPO + RLVR 训练与评测工程。

本文档已合并原 `DATASET.md` 与 `REPRODUCE.md` 的内容，并补齐交付清单、复现实操、质控门禁、问题复盘、结项报告。

---

## 1. 项目目标与最终状态

### 1.1 项目目标

- 构建医疗问答 RAG 系统
- 在 RAG 基础上完成 SFT、DPO、RLVR 三阶段训练
- 形成可复现、可验收、可发布的工程化流水线
- 给出完整质控门禁，支持发布决策

### 1.2 当前状态（2026-02-28）

- 训练链路：已完成（SFT / DPO / RLVR）
- 评测链路：已完成（`eval_test(10)` + `eval_max(100)`）
- 工程化强化：已完成（生成后清洗、护栏、门禁、一键检查）
- 业务验收：已完成（真实问题集 + 人工打分）
- 发布资料：已完成（Freeze v1.2、门禁报告、强化报告）

结论：项目达到交付标准，可进入仓库对外共享与复现阶段。

---

## 2. 环境与硬件要求

### 2.1 推荐硬件

- GPU：RTX 4090（24GB）
- CPU：8 核及以上
- 内存：32GB 及以上
- 硬盘：200GB 及以上可用空间

### 2.2 软件版本

- Python：3.10.x
- PyTorch：已在当前环境验证
- 依赖安装方式：`pip install -r requirements.lock`

### 2.3 环境准备命令

```bash
cd /root/autodl-tmp/medical-rag-rl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

### 2.4 环境检查命令

```bash
python scripts/check_env.py
python scripts/check_env.py --strict-train
```

---

## 3. 数据集分发与校验（合并自 DATASET.md）

### 3.1 必需数据包

根据 `MEDICAL_RAG_RLHF_RLVR_4090_PLAN.md` 的数据范围约束，以下三份数据包完整提供后即可覆盖复现所需训练与评测数据。

- `PubMed_20k_RCT.zip` -> `data/raw/PubMed_20k_RCT`
- `pubmed_20k_rct_processed.zip` -> `data/processed/pubmed/pubmed_20k_rct`
- `synthetic_bootstrap.zip` -> `data/synthetic_bootstrap/synthetic`

### 3.2 与计划文档的路径映射确认

计划文档中的 Windows 源路径：

- `D:\\pythonProjects\\NanoRAFT-RL\\data\\raw\\pubmed-rct-master\\PubMed_20k_RCT`
- `D:\\pythonProjects\\NanoRAFT-RL\\data\\processed\\pubmed\\pubmed_20k_rct`
- `D:\\pythonProjects\\NanoRAFT-RL\\data\\synthetic`

复现目标路径：

- `data/raw/PubMed_20k_RCT`
- `data/processed/pubmed/pubmed_20k_rct`
- `data/synthetic_bootstrap/synthetic`

结论：以上三份目录对应内容上传到百度网盘并按目标路径落盘后，配合仓库脚本与依赖锁文件，可完成完整复现流程。

### 3.3 百度网盘发布模板（示例占位符）

以下字段用于发布时替换为真实信息：

| 数据包 | 百度网盘链接（示例占位符） | 提取码（示例占位符） | SHA256（示例占位符） |
| --- | --- | --- | --- |
| `PubMed_20k_RCT.zip` | `<BAIDUPAN_URL_PUBMED_20K_RCT>` | `<PAN_CODE_PUBMED_20K_RCT>` | `<SHA256_PUBMED_20K_RCT>` |
| `pubmed_20k_rct_processed.zip` | `<BAIDUPAN_URL_PUBMED_20K_PROCESSED>` | `<PAN_CODE_PUBMED_20K_PROCESSED>` | `<SHA256_PUBMED_20K_PROCESSED>` |
| `synthetic_bootstrap.zip` | `<BAIDUPAN_URL_SYNTHETIC_BOOTSTRAP>` | `<PAN_CODE_SYNTHETIC_BOOTSTRAP>` | `<SHA256_SYNTHETIC_BOOTSTRAP>` |

### 3.4 数据目录要求

- `data/raw/PubMed_20k_RCT`
- `data/processed/pubmed/pubmed_20k_rct`
- `data/synthetic_bootstrap/synthetic`

### 3.5 数据校验命令

```bash
python scripts/verify_data.py
```

如需 SHA 校验：

```bash
python scripts/verify_data.py --sha-json artifacts/data_sha256.json
```

### 3.6 可复现条件确认

满足下列条件后，项目可完成完整复现：

1. 第 3.1 节三份数据包可下载并正确落盘  
2. Python 依赖可安装（`requirements.lock`）  
3. 基座模型与 embedding 模型可获取

模型获取可使用在线下载，或使用网盘镜像占位符：

- `<BAIDUPAN_URL_QWEN2_7B_INSTRUCT>` / `<PAN_CODE_QWEN2_7B_INSTRUCT>`
- `<BAIDUPAN_URL_BGE_LARGE_EN_V1_5>` / `<PAN_CODE_BGE_LARGE_EN_V1_5>`

上述条件全部满足时，仓库第三方复现流程可闭环执行。

---

## 4. 完整复现指南（合并自 REPRODUCE.md）

### 4.1 Smoke 复现

```bash
cd /root/autodl-tmp/medical-rag-rl
bash scripts/reproduce_smoke.sh
```

### 4.2 全流程复现

```bash
cd /root/autodl-tmp/medical-rag-rl
bash scripts/reproduce_full.sh
```

### 4.3 分阶段复现（手动）

```bash
cd /root/autodl-tmp/medical-rag-rl
python scripts/build_sft_data.py --split all
python scripts/train_sft.py
python scripts/build_preference_data.py
python scripts/train_dpo.py
python scripts/train_rlvr.py
```

### 4.4 复现注意事项

- 固定随机种子：`configs/system.yaml`
- 模型路径优先使用本地路径，避免训练时隐式联网下载
- 生成指标常见波动范围：`±1% ~ ±3%`

---

## 5. 项目主流程与关键脚本

1. 数据准备：`scripts/prepare_pubmed.py`
2. 构建索引：`scripts/build_index.py`
3. RAG 生成：`scripts/run_rag_baseline.py`
4. SFT 数据构建：`scripts/build_sft_data.py`
5. SFT 训练：`scripts/train_sft.py`
6. DPO 数据构建：`scripts/build_preference_data.py`
7. DPO 训练：`scripts/train_dpo.py`
8. RLVR 训练：`scripts/train_rlvr.py`
9. 统一评测：`scripts/run_eval.py`
10. 护栏审计：`scripts/guardrail_audit.py`
11. 质量门禁：`scripts/release_quality_gate.py`
12. 发布一键检查：`scripts/run_release_checks.sh`

---

## 6. 工程化与质控强化（本轮重点）

### 6.1 生成后清洗强化

文件：`src/rag/generator.py`

已实现：

- 清除对话残片：`Human/User/Assistant/System`
- 规范化引用标签：全角数字、下标数字、链接引用
- 清除提示词残留：`请总结...`、`Answer in Chinese...`
- 清除不完整引用尾巴：`...[Doc`
- 抑制重复引用与重复段落
- 仅在采样模式传 `temperature`，减少 warning 噪声

### 6.2 护栏审计增强

文件：`scripts/guardrail_audit.py`

新增检测项：

- `prompt_residue_count`
- `incomplete_doc_tail_count`

这两项已纳入 PASS 条件。

### 6.3 统一发布门禁

文件：`scripts/release_quality_gate.py`

门禁聚合项：

- 自动指标：`token_f1 / citation / evidence_hit / doc_repeat / safety_refusal`
- 护栏指标：`empty/chat_marker/malformed_tag/prompt_residue/incomplete_tail`
- 人工评分：`overall_0_5`（可选）

### 6.4 一键执行

```bash
cd /root/autodl-tmp/medical-rag-rl
bash scripts/run_release_checks.sh
```

---

## 7. 发布定版与结果

### 7.1 Freeze 文档

- `artifacts/release/release_freeze_2026-02-28.md`
- `artifacts/release/release_freeze_2026-02-28.json`
- `artifacts/release/release_freeze_2026-02-28_v1.2.md`
- `artifacts/release/release_freeze_2026-02-28_v1.2.json`

### 7.2 评测口径与可比性

- 主对比集：`eval_max`（100 条），用于阶段间主结论。
- 小样本验证集：`eval_test`（10 条），用于发布前快速回归。
- 全部指标来自 `artifacts/metrics/*.json`，门禁结果来自 `artifacts/release/*.json`。
- 业务验收集不提供标准答案，因此业务门禁中 `avg_token_f1` 设为可选（`require_token_f1=false`）。

### 7.3 分阶段结果总览（eval_max=100）

| 阶段/模型 | avg_token_f1 | avg_citation_consistency | avg_evidence_hit_rate | avg_doc_tag_repeat_ratio | avg_safety_refusal_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| base_max | 0.4506 | 0.0200 | 0.0200 | 0.0000 | 0.2400 |
| sft_1k_v2_clean4_max | 0.5419 | 0.6017 | 0.6500 | 0.0067 | 0.0200 |
| dpo_1k_v2_clean4_max | 0.5039 | 0.7400 | 0.7400 | 0.0000 | 0.0000 |
| rlvr_1k_v2_clean4_max | 0.5085 | 0.7880 | 0.8000 | 0.0240 | 0.0000 |
| sft_1k_v2_clean5_max | 0.5435 | 0.6067 | 0.6500 | 0.0067 | 0.0200 |
| rlvr_1k_v2_clean5_max | 0.5087 | 0.7880 | 0.8000 | 0.0240 | 0.0000 |

结论：

- SFT 阶段把语义贴近度（token_f1）从 `0.4506` 拉升到 `0.5419~0.5435`，提升最显著。
- DPO/RLVR 阶段继续提升证据维度，citation 与 evidence 分别提升到 `0.74/0.74` 与 `0.788/0.8`。
- RLVR 的证据对齐最佳，SFT 的 token_f1 最优，形成清晰的双模型互补结构。

### 7.4 相对 base 的增益拆解（eval_max=100）

| 模型 | Δtoken_f1 | Δcitation | Δevidence_hit | Δsafety_refusal |
| --- | ---: | ---: | ---: | ---: |
| sft_1k_v2_clean4_max | +0.0913 | +0.5817 | +0.6300 | -0.2200 |
| dpo_1k_v2_clean4_max | +0.0533 | +0.7200 | +0.7200 | -0.2400 |
| rlvr_1k_v2_clean4_max | +0.0579 | +0.7680 | +0.7800 | -0.2400 |
| sft_1k_v2_clean5_max | +0.0929 | +0.5867 | +0.6300 | -0.2200 |
| rlvr_1k_v2_clean5_max | +0.0581 | +0.7680 | +0.7800 | -0.2400 |

结果解释：

- SFT 的主要收益来自语义生成能力增强；RLVR 的主要收益来自证据引用与命中率增强。
- 安全拒答率在训练后显著下降（`0.24 -> 0.00~0.02`），可用性明显提升。
- `doc_tag_repeat_ratio` 在 RLVR 高于 SFT，但仍低于门禁阈值 `0.05`。

### 7.5 clean4 -> clean5 的增量影响

| 模型 | token_f1 增量 | citation 增量 | evidence 增量 | repeat 增量 | safety 增量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sft: clean4 -> clean5 | +0.0016 | +0.0050 | +0.0000 | +0.0000 | +0.0000 |
| rlvr: clean4 -> clean5 | +0.0002 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

结论：

- clean5 的核心价值不在“再拉高指标”，而在“清洗与护栏后的发布稳定性”。
- 指标基本持平说明 clean5 对主能力无明显副作用，可视为低风险工程化强化版本。

### 7.6 发布集与小样本回归（eval_test=10）

| 模型 | avg_token_f1 | avg_citation_consistency | avg_evidence_hit_rate | avg_doc_tag_repeat_ratio |
| --- | ---: | ---: | ---: | ---: |
| sft_1k_v2_clean5_test | 0.5555 | 0.8000 | 0.9000 | 0.0000 |
| rlvr_1k_v2_clean5_test | 0.5086 | 0.8000 | 0.8000 | 0.0000 |

说明：

- 小样本下两模型均满足门禁阈值。
- SFT 在小样本语义贴近度更高，RLVR 在大样本证据维度更稳定。

### 7.7 门禁与护栏：前后对照

#### 7.7.1 业务验收集（真实问题）

| 检查项 | 强化前 | 强化后 |
| --- | ---: | ---: |
| 护栏通过文件数 | 0/2 | 2/2 |
| 质量门禁通过模型数 | 0/2 | 2/2 |
| prompt residue 总数 | 5 | 0 |
| incomplete tail 总数 | 3 | 0 |

对应文件：

- `artifacts/release/biz_real_guardrail_pre_harden_v2.json`
- `artifacts/release/biz_real_guardrail_harden_v3.json`
- `artifacts/release/biz_quality_gate_pre_harden.json`
- `artifacts/release/biz_quality_gate_harden_v3.json`

#### 7.7.2 发布评测集（test+max）

| 检查项 | clean4 | clean5 |
| --- | ---: | ---: |
| 质量门禁通过模型数 | 0/2 | 4/4 |
| 护栏通过文件数 | 未形成最终全通过报告 | 4/4 |
| prompt residue 比例 | 出现超阈值样本 | 全部为 0 |
| incomplete tail 比例 | 出现超阈值样本 | 全部为 0 |

对应文件：

- `artifacts/release/release_quality_gate_clean4_v2.json`
- `artifacts/release/release_quality_gate_clean5_all_v1.json`
- `artifacts/release/release_models_guardrail_clean5_all.json`

### 7.8 人工评分分析（40 条，Codex 代评）

来源：`artifacts/release/acceptance/human_scoring_final_summary.md`

| 模型 | correctness | evidence_alignment | readability | safety | overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| biz_real_rlvr | 3.20 | 3.10 | 2.85 | 5.00 | 3.25 |
| biz_real_sft | 3.30 | 3.10 | 2.75 | 5.00 | 3.10 |

客观解读：

- RLVR 的综合分更高（`3.25`），优势来自结构化输出稳定性与整体可读性。
- SFT 的 correctness 略高（`3.30`），但在可读性与格式稳定性上略逊。
- 两模型 safety 均为 `5.00`，安全性风险可控。
- 主要误差类型集中在轻度幻觉、格式噪声、提示词残留与截断，已在 clean5 通过清洗与护栏显著收敛。

### 7.9 发布建议与选择策略

- 主发布：`rlvr_1k_v2_clean5`
- 回退发布：`sft_1k_v2_clean5`
- 适用策略：
  - 证据对齐、可追溯性优先时，选择 RLVR。
  - 语义贴近与问答自然性优先时，选择 SFT。
  - 面向生产上线时，保持双模型并行灰度，按业务指标动态路由。

### 7.10 剩余风险与后续优化方向

- 剩余风险：
  - 少量样本仍存在轻度术语漂移与可读性波动。
  - 业务场景缺少更大规模人工标注集，长期评估方差仍偏高。
- 优化方向：
  - 扩充高质量医学问答标注，按病种分层抽样。
  - 引入规则化术语表与后处理对齐器，压缩术语漂移。
  - 持续化门禁运行（按周/按版本）并累计趋势报表。

---

## 8. 产物清单（可交付目录树）

```text
medical-rag-rl/
├── README.md
├── pyproject.toml
├── requirements.lock
├── configs/
│   ├── system.yaml
│   ├── rag.yaml
│   ├── sft.yaml
│   ├── dpo.yaml
│   ├── rlvr.yaml
│   └── eval.yaml
├── scripts/
│   ├── check_env.py
│   ├── verify_data.py
│   ├── prepare_pubmed.py
│   ├── build_index.py
│   ├── run_rag_baseline.py
│   ├── build_sft_data.py
│   ├── train_sft.py
│   ├── build_preference_data.py
│   ├── train_dpo.py
│   ├── train_rlvr.py
│   ├── run_eval.py
│   ├── guardrail_audit.py
│   ├── release_quality_gate.py
│   └── run_release_checks.sh
├── src/
│   ├── rag/
│   │   ├── generator.py
│   │   └── pipeline.py
│   ├── retrieval/
│   ├── train/
│   └── eval/
├── tests/
│   ├── test_generator_cleanup.py
│   ├── test_quality_gate.py
│   ├── test_rag_pipeline.py
│   ├── test_retrieval.py
│   └── test_rewards.py
├── artifacts/
│   └── release/
│       ├── release_freeze_2026-02-28_v1.2.md
│       ├── release_freeze_2026-02-28_v1.2.json
│       ├── engineering_qc_upgrade_2026-02-28.md
│       ├── release_quality_gate_clean5_all_v1.json
│       ├── release_quality_gate_clean5_all_v1.md
│       ├── release_models_guardrail_clean5_all.json
│       ├── release_models_guardrail_clean5_all.md
│       └── acceptance/
│           ├── README_CN.md
│           ├── queries_template.jsonl
│           ├── human_scoring_template.csv
│           ├── human_scoring_final.csv
│           └── human_scoring_final_summary.md
└── outputs/
    └── baseline/
        ├── sft_1k_v2_clean5_test.jsonl
        ├── sft_1k_v2_clean5_max.jsonl
        ├── rlvr_1k_v2_clean5_test.jsonl
        └── rlvr_1k_v2_clean5_max.jsonl
```

---

## 9. 最终命令清单（从验收到门禁一键复现）

### 9.1 业务验收生成

```bash
cd /root/autodl-tmp/medical-rag-rl

python scripts/run_rag_baseline.py \
  --split train \
  --query-jsonl artifacts/release/acceptance/queries_template.jsonl \
  --name biz_real_rlvr \
  --adapter outputs/raft-rlvr-1k-v2/final \
  --output-path outputs/baseline/biz_real_rlvr.jsonl

python scripts/run_rag_baseline.py \
  --split train \
  --query-jsonl artifacts/release/acceptance/queries_template.jsonl \
  --name biz_real_sft \
  --adapter outputs/raft-sft-1k-v2/final \
  --output-path outputs/baseline/biz_real_sft.jsonl
```

### 9.2 业务验收护栏

```bash
python scripts/guardrail_audit.py \
  --name biz_real_guardrail \
  --pred outputs/baseline/biz_real_rlvr.jsonl outputs/baseline/biz_real_sft.jsonl
```

### 9.3 发布评测集生成（test + max）

```bash
python scripts/run_rag_baseline.py \
  --split train \
  --query-jsonl data/eval/eval_test.jsonl \
  --name sft_1k_v2_clean5_test \
  --adapter outputs/raft-sft-1k-v2/final \
  --output-path outputs/baseline/sft_1k_v2_clean5_test.jsonl

python scripts/run_rag_baseline.py \
  --split train \
  --query-jsonl data/eval/eval_test.jsonl \
  --name rlvr_1k_v2_clean5_test \
  --adapter outputs/raft-rlvr-1k-v2/final \
  --output-path outputs/baseline/rlvr_1k_v2_clean5_test.jsonl

python scripts/run_rag_baseline.py \
  --split train \
  --query-jsonl data/eval/eval_max.jsonl \
  --name sft_1k_v2_clean5_max \
  --adapter outputs/raft-sft-1k-v2/final \
  --output-path outputs/baseline/sft_1k_v2_clean5_max.jsonl

python scripts/run_rag_baseline.py \
  --split train \
  --query-jsonl data/eval/eval_max.jsonl \
  --name rlvr_1k_v2_clean5_max \
  --adapter outputs/raft-rlvr-1k-v2/final \
  --output-path outputs/baseline/rlvr_1k_v2_clean5_max.jsonl
```

### 9.4 自动评测

```bash
python scripts/run_eval.py --name sft_1k_v2_clean5_test --pred-path outputs/baseline/sft_1k_v2_clean5_test.jsonl
python scripts/run_eval.py --name rlvr_1k_v2_clean5_test --pred-path outputs/baseline/rlvr_1k_v2_clean5_test.jsonl
python scripts/run_eval.py --name sft_1k_v2_clean5_max --pred-path outputs/baseline/sft_1k_v2_clean5_max.jsonl
python scripts/run_eval.py --name rlvr_1k_v2_clean5_max --pred-path outputs/baseline/rlvr_1k_v2_clean5_max.jsonl
```

### 9.5 发布护栏 + 质量门禁（推荐一键）

```bash
bash scripts/run_release_checks.sh
```

### 9.6 人工评分（可选）

- 模板：`artifacts/release/acceptance/human_scoring_template.csv`
- 已完成样例：`artifacts/release/acceptance/human_scoring_final.csv`

---

## 10. 简版结项报告（1页）

### 10.1 项目背景

项目目标是构建医疗 RAG 系统并完成 SFT/DPO/RLVR 三阶段训练，最终输出可发布版本与可复现流程。

### 10.2 执行范围

- 数据准备、索引构建、RAG 推理
- SFT、DPO、RLVR 训练
- 自动评测、业务验收、人工评分
- 工程化质控强化（清洗、护栏、门禁）

### 10.3 关键成果

- `clean5` 版本在指标保持稳定前提下，通过了完整发布门禁
- 业务验收集由 `0/2` 通过提升到 `2/2` 通过
- 发布评测集（test+max）达到护栏 `4/4` 与门禁 `4/4` 通过

### 10.4 量化结果摘要

- `sft_1k_v2_clean5_max`: token_f1=0.5435, citation=0.6067, evidence=0.6500
- `rlvr_1k_v2_clean5_max`: token_f1=0.5087, citation=0.7880, evidence=0.8000

### 10.5 主要风险与控制

- 风险：生成残留、引用尾巴、格式噪声
- 控制：生成后清洗规则 + 护栏审计 + 质量门禁
- 结果：残留与尾巴问题在发布集归零，门禁全通过

### 10.6 交付结论

项目已达到交付标准，可用于仓库发布与完整复现。

---

## 11. 全量问题复盘（项目过程）

| 问题 | 现象 | 根因 | 处理方案 | 验证结果 |
| --- | --- | --- | --- | --- |
| 多语嵌入模型下载失败 | HuggingFace 拉取 `bge-m3` 报网络不可达 | 外网限制 | 固定本地 `bge-large-en`，增加中文翻译检索 | 训练评测持续可跑 |
| RLVR 显存 OOM | 第 1 个 update 崩溃 | `kl_coef>0` 需参考模型，峰值显存高 | `batch=1`, `grad_accum=16`, `max_new_tokens=128`，加 OOM 跳过保护 | RLVR 完整跑通 |
| evidence_hit 偏低 | 中文场景几乎全 0 | 词面重叠对跨语言不友好 | 增加跨语言分支 | 指标恢复可解释 |
| 对话残片污染 | 输出含 `Human:/Assistant:` | 训练分布残留 | 生成后清洗截断 | clean5 发布集为 0 |
| 引用标签异常 | `[Doc ２]`、`[Doc 1](...)` | 格式噪声 | 统一正规化 | 护栏通过 |
| 引用循环 | 重复 `[Doc x]` | 偏好训练诱导 | 压缩重复引用 | repeat 指标可控 |
| 提示词残留/尾巴 | `请总结...` / `...[Doc` | 残留 + 截断 | 清洗规则升级 + 审计入门禁 | 发布集归零 |
| 验收标准分散 | 指标与人工评分分离 | 缺统一门禁 | 新增 `release_quality_gate.py` | clean5 门禁全通过 |
| warning 噪声 | `do_sample=False` 仍报 sampling 参数 warning | generation_config 继承默认 | do_sample=False 时不传采样参数 | warning 显著减少 |

---

## 12. 关键路径文件

- 主文档：`README.md`
- 工程强化报告：`artifacts/release/engineering_qc_upgrade_2026-02-28.md`
- 最新 Freeze：`artifacts/release/release_freeze_2026-02-28_v1.2.md`
- 最新门禁：`artifacts/release/release_quality_gate_clean5_all_v1.json`
- 最新护栏：`artifacts/release/release_models_guardrail_clean5_all.json`
- 最终人工评分：`artifacts/release/acceptance/human_scoring_final.csv`

---

## 13. 对外共享建议

- 仓库中保留：代码、配置、脚本、文档、验收模板、门禁规则
- 仓库中忽略：本地模型权重、运行日志、中间输出、大体量原始数据
- 共享时补充：数据下载链接与 SHA256

---

## 14. 附：常用命令速查

```bash
# 1) 环境检查
python scripts/check_env.py --strict-train

# 2) 数据校验
python scripts/verify_data.py

# 3) 一键烟雾复现
bash scripts/reproduce_smoke.sh

# 4) 一键全流程复现
bash scripts/reproduce_full.sh

# 5) 一键发布检查
bash scripts/run_release_checks.sh
```
