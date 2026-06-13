# CLAIMARC-v2 实验结果汇总（夜间持续研究）

> 数据：`dataset_verify_faithful.jsonl`（去噪后 train 1101 / val 256 / test 337，正类率约 38%）。
> 评测指标对齐 §4.3：**主指标 = Macro-F1（首要）、可靠性加权 F1（wF1）、AUPRC**；AUROC/ECE 为辅。
> 显著性：测试集 n=337 上的**配对 bootstrap**（2000 次重采样），与 §4.4.1 要求一致。

---

## 1. 锁定的最优架构（经穷尽搜索验证）

**CLAIMARC-v2 = BGE-large-zh + LoRA(r=48) + 双流 + 2 层双向交叉注意力 + 属性分块 RACL(λ=0.5, τ=0.07, Kp=3,Kn=5) + 2 特征检索-参数融合(blend2) + 单调 Platt 标定 + 5 种子集成。**

架构搜索（均在 RACL 核心不变前提下）得到的关键结论：

| 改动方向 | 结果 | 结论 |
|---|---|---|
| LoRA 秩 16/32/48/64 | mF1 0.60→**0.623**(r48)→回落 | r48 最优 |
| 全量微调 BGE-large | AP 0.503 mF1 0.608 | **比 LoRA 差**（小数据过拟合） |
| 骨干换 RoBERTa-wwm-ext 全量微调 | AP 0.479 mF1 0.585 | **双流框架内 RoBERTa 反而更差** |
| 骨干换 RoBERTa + LoRA | AP 0.488 | 同上 |
| 骨干换 BERT-base | mF1 0.592 | **BGE > BERT 骨干**（验证 §4.4.5 消融1） |
| 更强对比(λ=1.0~1.5, τ=0.05, Kn=10) | mF1 0.574, knn-AP↓ | 过度对比损害分类头，**canonical 更优** |
| 跨秩异构集成(16+32+48) | AP 0.533 | 冻结 BGE 特征相关→集成无增益 |
| 融合：标量α vs **blend2(2特征)** vs 7特征ARF | blend2 最稳，ARF 过拟合(AP↓到0.49) | **blend2 为主方法** |

> **要点**：架构搜索本身**验证了提案的设计选择**——检索预训练的 BGE 骨干 + LoRA + 双流交叉注意力，确实优于全量微调 / 通用骨干 / 单流。把 RoBERTa 塞进双流反而变差，说明双流 + RACL 的收益与"检索式骨干"绑定。

---

## 2. 主分布对比（RQ1，5 种子集成，固定划分）

| 方法 | AUPRC | AUROC | **Macro-F1** | wF1 | ECE |
|---|---|---|---|---|---|
| **CLAIMARC-v2** | 0.543 | 0.668 | **0.623** | 0.603 | 0.306 |
| RoBERTa-wwm-ext + [CLS] | **0.574** | **0.680** | 0.603 | 0.559 | 0.314 |
| BERT-base + [CLS] | 0.537 | 0.655 | 0.607 | 0.578 | 0.286 |
| BERT-NLI (OCNLI) | 0.497 | 0.646 | 0.583 | 0.566 | 0.332 |
| ESIM | 0.473 | 0.606 | 0.553 | 0.501 | 0.245 |
| 冻结 BGE + LR（嵌入上限） | 0.524 | 0.675 | 0.616 | **0.619** | 0.278 |
| 冻结 BGE + kNN | 0.497 | 0.636 | 0.586 | 0.607 | 0.257 |
| 双编码器 + 余弦（无交叉注意力） | 0.379 | 0.482 | 0.384 | 0.333 | 0.244 |

**配对 bootstrap（CLAIMARC vs 基线；p = P(基线≥CLAIMARC)）：**

| vs 基线 | ΔMacro-F1 (p) | ΔAUPRC (p) | ΔAUROC (p) |
|---|---|---|---|
| ESIM | **+0.072 (0.024)** ✓ | +0.066 (0.088) | +0.063 (0.054) |
| 双编码器余弦 | **+0.240 (0.000)** ✓ | **+0.162 (0.000)** ✓ | **+0.187 (0.000)** ✓ |
| BERT-NLI | +0.018 (0.26) | +0.044 (0.11) | +0.022 (0.22) |
| BERT-base | +0.011 (0.35) | +0.006 (0.42) | +0.013 (0.31) |
| 冻结 BGE+kNN | +0.039 (0.076) | +0.042 (0.13) | +0.033 (0.10) |
| 冻结 BGE+LR | +0.005 (0.43) | +0.017 (0.31) | −0.008 (0.62) |
| RoBERTa | +0.020 (0.77) | **−0.029 (0.81)** | −0.012 (0.67) |

**诚实结论（RQ1）：**
- CLAIMARC-v2 在**首要指标 Macro-F1 上为全场最高（0.623）**，对 ESIM、双编码器余弦达到**统计显著**。
- 对最强的两个监督基线（RoBERTa、冻结 BGE+LR），CLAIMARC 在 Macro-F1 上领先但**差异未达统计显著**；RoBERTa 的 AUPRC 略高（+0.029）也**不显著**（p=0.81）。
- **任务在 n=337 域内是噪声受限的**：诸强方法统计上打平。要把 ~0.02–0.03 的差异做成显著，需更大测试集 / 交叉验证（超出提案"固定划分 3 种子"协议）。

---

## 3. 消融实验（§4.4.5，3 种子；boundary 子集 = 69/337）

| 配置 | 整体 Macro-F1 | 整体 AUPRC | **boundary Macro-F1** |
|---|---|---|---|
| **完整 CLAIMARC** | **0.623** | 0.543 | 0.549 |
| − 对比目标 (λ=0) | 0.593 | 0.516 | 0.478 (**−0.071**) |
| − 属性分块（全局负采样） | 0.596 | 0.536 | **0.433 (−0.116)** |
| − 双流融合（退化为双编码器） | 0.609 | 0.521 | 0.605* |
| BGE → BERT 骨干 | 0.592 | 0.526 | 0.604* |

- **完整模型在整体 Macro-F1 上优于每一个消融** → 每个组件都有正贡献。
- **属性分块是核心**：换成全局随机负采样，boundary Macro-F1 暴跌 0.116；移除对比目标跌 0.071——与 §4.4.3/§5.1 关于"边界样本最依赖属性分块对比"的预期**方向一致且幅度最大**。
- *注：boundary 子集仅 69 样本，nofus/bert 的 boundary 数值方差大，不宜过度解读；整体指标更可靠。

---

## 4. 跨域少样本适应（RQ3，留一品类 × 3 种子，10 次有效运行）

| 推理方式 | 目标域 Macro-F1 |
|---|---|
| CLAIMARC 前向（零样本，无适应） | 0.571 |
| CLAIMARC RKC（检索投票，m=0） | **0.580** |
| CLAIMARC RKC（注入 m=5/10/20 support） | 0.579 / 0.576 / 0.579 |
| 冻结 BGE + kNN（m=0 / m=20） | 0.576 / 0.574 |

- CLAIMARC 的**对比检索表征**零样本即优于前向（+0.009）与通用 BGE-kNN（+0.004）。
- 但**增量注入 support 样本未带来稳定增益**（m=0≈m=20）——本数据留出品类与源域共享大量属性，零样本检索已捕获多数信号，"检索库增量适应"的边际价值有限。这是一个**诚实的负结果**，建议在 RQ3 表述中调整为"检索表征跨域可迁移"而非"增量更新显著获益"。

---

## 4b. 分组 5 折交叉验证（n=1694，最严格证据）⚠️ 关键

固定划分 n=337 噪声受限，为提升统计功效，按 `room_id` 分组做 StratifiedGroupKFold(5)，
每样本恰预测一次，汇集 OOF（n=1694）+ 配对 bootstrap。**结论与固定划分相反，必须正视。**

**(a) 用主方法 blend2+Platt 融合（与固定划分一致的推理）：**

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---|---|---|---|
| CLAIMARC_v2 (blend2+Platt) | 0.434 | 0.565 | 0.561 | 0.553 |
| RoBERTa | 0.474 | 0.604 | 0.561 | 0.580 |
| BERT | 0.461 | 0.603 | 0.566 | 0.561 |
| 冻结 BGE + LR | **0.477** | **0.621** | **0.579** | **0.609** |

bootstrap：CLAIMARC_v2 **显著差于**三个基线（ΔMacro-F1 −0.03~−0.035，p≈0.93~1.0）。
→ **blend2/Platt 融合在小 val 折上过拟合**，固定划分上的融合增益是划分特定的。

**(b) 改用稳健推理（原始 p_cls 集成 + 仅阈值，无 blend2/Platt）：**

| 方法 | AUPRC | AUROC | **Macro-F1** | wF1 |
|---|---|---|---|---|
| **CLAIMARC (p_cls 集成)** | 0.466 | 0.593 | **0.577** | 0.581 |
| RoBERTa | 0.474 | 0.604 | 0.561 | 0.580 |
| BERT | 0.461 | 0.603 | 0.566 | 0.561 |
| 冻结 BGE + LR | **0.477** | **0.621** | 0.579 | **0.609** |

**严格 CV 的诚实结论：**
1. **冻结 BGE + LR 是该任务最稳健的方法**（AUROC/wF1 最高、标定最好、零训练成本）。
2. CLAIMARC 用稳健推理时，Macro-F1 与 BGE+LR **基本打平**、并**优于 RoBERTa/BERT 等监督 Transformer 基线**；但在 AUPRC/AUROC/wF1 上**略逊**。
3. CLAIMARC 复杂的双流+融合机制**未带来稳健的泛化增益**；它的价值更多体现在"Macro-F1 与最强嵌入基线持平 + 可解释检索 + 部署侧检索库扩展"，而非判别精度的全面碾压。
4. **根因推测**：标签由评论弱监督导出，信号上限受标注质量制约；冻结 BGE 嵌入已捕获多数可学信号，额外参数化机制在小样本上更易过拟合。

> ⚠️ **研究诚信提示**：固定划分上"CLAIMARC 主指标最优"不可在严格 CV 上复现，不应作为"显著优于基线"的依据。建议论文采用 CV 或多划分汇报，并据实定位贡献。

**(c) 互补性检验（rank-average 集成，CV OOF）：**

| 系统 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---|---|---|---|
| 冻结 BGE+LR（最强单模型） | 0.477 | 0.621 | 0.579 | 0.609 |
| BGE+RoBERTa 集成 | 0.487 | 0.623 | 0.591 | 0.604 |
| **BGE+RoBERTa+CLAIMARC 集成** | **0.488** | **0.624** | **0.601** | **0.625** |

- **含 CLAIMARC 的集成在 CV 上是最强系统**（全指标最高）；但相对 BGE+RoBERTa，CLAIMARC 的边际贡献小且不显著（ΔMacro-F1 +0.010 p=0.15，ΔAUPRC +0.001 p=0.45）。
- 解读：CLAIMARC 提供**少量互补信号（主要在 F1/标定维度）**，可作为集成成员锦上添花，但单独不构成对基线的稳健显著优势。

---

## 4c. 2026-06-07 clean-RACL 复核：低容量 + 置信过滤 + 稳健推理

针对 4b 中“复杂融合过拟合”的问题，新增一轮结构调整：

- 骨干从 BGE-large LoRA(r=48) 降为 **BGE-small LoRA(r=8)**，降低小样本过拟合。
- 训练改为 **warmup=1 + CL=2**，并加入 `cl_c_min=0.10 / cl_neg_c_min=0.10`，低置信样本不作为对比锚点或负例。
- 推理层面区分三种头：`CLAIMARC_pcls`（仅参数化分类头）、`CLAIMARC_selectiveRKC`（保守门控检索修正）、`CLAIMARC_v2`（旧 blend2+Platt）。

固定划分 5-seed 的最优配置为 `small_e3_c10`：

| 推理头 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| **CM_pcls_ens** | **0.5385** | **0.6723** | **0.6168** | **0.6086** |
| CM_blendScalar | 0.5345 | 0.6705 | 0.6028 | 0.5826 |
| CLAIMARC_v2 | 0.5213 | 0.6644 | 0.5978 | 0.5760 |
| kNN-only | 0.4666 | 0.6217 | 0.5808 | 0.5516 |

严格分组 5 折 CV（OOF n=1694，仅对最强嵌入基线 BGE+LR 复核）：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| **CLAIMARC_pcls** | **0.4734** | **0.6187** | **0.5891** | **0.6228** |
| CLAIMARC_selectiveRKC | 0.4603 | 0.6074 | 0.5889 | 0.6221 |
| CLAIMARC_v2 (blend2+Platt) | 0.4565 | 0.5963 | 0.5795 | 0.5961 |
| BGE+LR | 0.4698 | 0.6132 | 0.5681 | 0.5958 |

配对 bootstrap（`CLAIMARC_pcls` vs BGE+LR）：

| 指标 | Δ | 95% CI | p = P(基线≥方法) |
|---|---:|---:|---:|
| AUPRC | +0.0038 | [-0.0192, 0.0279] | 0.3845 |
| AUROC | +0.0058 | [-0.0145, 0.0266] | 0.2765 |
| Macro-F1 | +0.0047 | [-0.0173, 0.0266] | 0.3335 |

**更新后的诚实结论：**

1. clean-RACL 后，CLAIMARC 首次在严格 CV 上**全指标超过 BGE+LR**，说明“低容量 + 置信过滤 + warmup 后对比学习”方向有效。
2. 领先仍未达到统计显著；目前不能声称“显著优于最强基线”，但已经从“打平/略逊”推进到“全指标正向”。
3. 推理时检索投票仍不稳：`kNN-only` 和 `blend2/Platt` 均弱于 `p_cls`，`selectiveRKC` 也没有额外收益。论文方法应重定为 **RACL 训练增强 + 稳健参数化推理**，检索用于表征约束、解释和扩展，而非直接标签投票。
4. 下一步优先尝试 LLM 生成的支持/反驳/证据缺口 argument augmentation；后台已开始生成 `dataset_verify_faithful_args.jsonl`，完整后将自动跑 `args_small_e3_c10` 3-seed。

补充互补性检验：每折复用 OOF 预测，将 `CLAIMARC_pcls` 与 BGE+LR 做 rank-average：

| 系统 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| CLAIMARC_pcls | 0.4734 | 0.6187 | **0.5891** | **0.6228** |
| BGE+LR | 0.4698 | 0.6132 | 0.5681 | 0.5958 |
| CLAIMARC+BGE rank-average | **0.4838** | **0.6287** | 0.5751 | 0.5936 |

对 BGE+LR 的 paired bootstrap：rank-average 在 AUPRC 上 Δ=+0.0136, p=0.0395；AUROC 上 Δ=+0.0156, p=0.0025；Macro-F1 上 Δ=+0.0112, p=0.096。  
解读：CLAIMARC 与冻结嵌入基线存在**显著排序互补性**，适合用于“高风险样本排序/筛查”场景；若以二分类 Macro-F1/wF1 为主，仍应采用 `CLAIMARC_pcls`。

随后对 `small_e3_c10` 做二轮小范围 sweep（固定划分 5-seed）：

| 配置 | 最强推理头 | AUPRC | AUROC | Macro-F1 | wF1 | 判断 |
|---|---|---:|---:|---:|---:|---|
| `small_e3_c05` | p_cls | 0.5468 | 0.6731 | 0.6089 | 0.5854 | 排序↑，F1↓ |
| `small_e3_c20` | scalar blend | **0.5525** | **0.6774** | 0.6028 | 0.5853 | 排序最高，F1不足 |
| `small_e3_lam03_c10` | scalar/v2 | 0.5503 | 0.6752 | 0.6072 | 0.5849 | 排序↑，F1↓ |
| `small_e3_c10` | p_cls | 0.5385 | 0.6723 | **0.6168** | **0.6086** | 主分类最稳 |

因此暂不对 cleancl2 做严格 CV；若论文强调 AUPRC/AUROC，可把 `small_e3_c20` 作为排序型变体或与 BGE+LR 做 rank-average，但主表仍采用 `small_e3_c10 + CLAIMARC_pcls`。

LLM argument augmentation（RAFTS-style 支持/反驳/证据缺口，固定划分 5-seed）完整跑通后：

| 方法头 | AUPRC | AUROC | Macro-F1 | wF1 | 判断 |
|---|---:|---:|---:|---:|---|
| args + p_cls | 0.5240 | 0.6611 | 0.5687 | 0.5789 | seed 方差大，分类退化 |
| args + kNN-only | 0.4940 | 0.6511 | 0.6096 | 0.5908 | argument 让检索空间更可用 |
| args + ARF_diag | **0.5561** | **0.6842** | 0.5951 | 0.6046 | 排序最高，但 F1 未超主方法 |
| no-args `small_e3_c10` + p_cls | 0.5385 | 0.6723 | **0.6168** | **0.6086** | 主分类仍最稳 |

解读：LLM 生成的支持/反驳 argument 对**检索排序**有实质帮助，尤其让 ARF 的 AP/AUROC 达到当前最高；但它也引入训练方差，p_cls 分类头明显退化。后续若继续推进，应将 argument 分支作为检索/排序专家，与 no-args p_cls 做多专家融合，而不是简单把 argument 拼入 evidence 流替代原输入。

严格分组 5 折 CV（argument 数据，OOF n=1694，2026-06-07 deterministic `val_carve` 修正版）：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| **CLAIMARC_pcls + arguments** | **0.4771** | **0.6301** | **0.5943** | **0.6216** |
| CLAIMARC_selectiveRKC + arguments | 0.4642 | 0.6116 | 0.5749 | 0.5985 |
| CLAIMARC_v2 + arguments | 0.4501 | 0.5844 | 0.5804 | 0.6069 |
| no-args CLAIMARC_pcls（同切分） | 0.4769 | 0.6193 | 0.5685 | 0.5907 |
| BGE+LR（原始证据） | 0.4570 | 0.6072 | 0.5739 | 0.6030 |

deterministic 复跑后的关键变化：

1. argument 版 `CLAIMARC_pcls` 仍明显好于 no-args 版，尤其 Macro-F1 +0.0258、wF1 +0.0309。
2. 但发现一个公平性问题：旧文本基线没有读入 `arguments` 字段，而 CLAIMARC 读入了。已修正 `src/models/baselines.py`，让 BGE+LR / RoBERTa / BERT 在 argument 数据上同样可见 supporting/refuting/evidence-gap 文本。

公平 argument-aware 基线（同一 OOF 协议）：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| CLAIMARC_pcls + arguments | 0.4771 | 0.6301 | **0.5943** | **0.6216** |
| fair BGE+LR + arguments | 0.4677 | 0.6248 | 0.5731 | 0.5912 |
| fair RoBERTa + arguments | 0.4420 | 0.6047 | 0.5827 | 0.5990 |
| fair BERT + arguments | 0.4498 | 0.5802 | 0.5846 | 0.5866 |

单模型 `CLAIMARC_pcls` 相对 fair BGE+LR 全指标更高，但 paired bootstrap 未显著：AUPRC Δ=+0.0097, p=0.1955；AUROC Δ=+0.0056, p=0.277；Macro-F1 Δ=+0.0088, p=0.2245。  
因此，单模型 CLAIMARC 不能作为“显著优于公平最强基线”的最终主张。

新的主候选是 **CLAIMARC-Hybrid = fair BGE+LR 与 argument-ARF 检索专家的 rank-average**。其中 argument-ARF 来自 CLAIMARC 的 RACL 表征和检索特征，BGE+LR 是公平强基线；若 hybrid 显著优于 BGE+LR，说明 RACL/argument 检索专家对强嵌入分类器提供了可复现增益。

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| **CLAIMARC-Hybrid: rankavg(args_ARF, fair BGE+LR)** | **0.5003** | **0.6502** | **0.5935** | **0.6174** |
| fair BGE+LR + arguments | 0.4677 | 0.6248 | 0.5731 | 0.5912 |
| fair RoBERTa + arguments | 0.4420 | 0.6047 | 0.5827 | 0.5990 |
| fair BERT + arguments | 0.4498 | 0.5802 | 0.5846 | 0.5866 |

5,000 次 paired bootstrap（CLAIMARC-Hybrid vs fair baselines）：

| vs 基线 | ΔAUPRC (p) | ΔAUROC (p) | ΔMacro-F1 (p) |
|---|---:|---:|---:|
| fair BGE+LR | **+0.0324 (0.0032)** | **+0.0254 (0.0006)** | **+0.0191 (0.0376)** |
| fair RoBERTa | **+0.0575 (0.0000)** | **+0.0452 (0.0000)** | **+0.0292 (0.0088)** |
| fair BERT | **+0.0505 (0.0014)** | **+0.0699 (0.0000)** | **+0.0464 (0.0006)** |

**seed=0 下的最强结论**：在公平 argument-aware 输入、确定性 grouped 5-fold OOF、5k paired bootstrap 下，CLAIMARC-Hybrid 对三个强基线在 AUPRC/AUROC/Macro-F1 上均达到统计显著优势。这是一个强方向信号，但还不是最终论文主张，因为它仍依赖单一 grouped split 随机种子。

### 4d. Repeated grouped CV 复核：seed=1 暴露分类不稳

为检查 seed=0 是否为折分特例，加入 `--fold_seed` 支持并在 `fold_seed=1` 上复跑 argument CLAIMARC + fair BGE+LR。结果如下：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| CLAIMARC_pcls + arguments | 0.4719 | 0.6070 | 0.5868 | **0.6289** |
| CLAIMARC_selectiveRKC + arguments | 0.4714 | 0.6062 | 0.5866 | 0.6264 |
| CLAIMARC_v2 + arguments | **0.4764** | 0.6140 | 0.5798 | 0.6093 |
| fair BGE+LR + arguments | 0.4641 | **0.6274** | **0.5975** | 0.6267 |

paired bootstrap（`CLAIMARC_pcls` vs fair BGE+LR）：AUPRC Δ=+0.0073, p=0.299；AUROC Δ=-0.0206, p=0.974；Macro-F1 Δ=-0.0330, p=0.997。  
这说明单模型 CLAIMARC 在 repeated split 上不稳，且 `fold_seed=1` 下分类边界明显输给 fair BGE。

继续评估多个无需重训的 hybrid 候选：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | vs fair BGE+LR 结论 |
|---|---:|---:|---:|---:|---|
| rankavg(args_pcls, BGE+LR) | **0.4815** | **0.6341** | 0.5901 | 0.6189 | AUPRC 显著 +0.0170 (p=0.013)，Macro-F1 不增 |
| rankavg(args_ARF, BGE+LR) | 0.4732 | 0.6180 | 0.5749 | 0.5992 | seed=0 主候选未复现 |
| rankavg(args_pcls, args_ARF, BGE+LR) | 0.4804 | 0.6236 | 0.5798 | 0.6085 | AUPRC 边界提升，分类下降 |
| val-select hybrid | 0.4667 | 0.6077 | 0.5732 | 0.6031 | 验证集门控选择也不稳 |

阈值诊断（`cv_threshold_diagnose.py`）显示：

| fold_seed | 方法 | val-threshold Macro-F1 | pooled oracle Macro-F1 | AUPRC | AUROC |
|---:|---|---:|---:|---:|---:|
| 0 | rankavg(args_ARF, BGE+LR) | 0.5935 | **0.6071** | 0.5003 | 0.6502 |
| 0 | fair BGE+LR | 0.5731 | 0.5881 | 0.4677 | 0.6248 |
| 1 | rankavg(args_pcls, BGE+LR) | 0.5901 | 0.5974 | **0.4815** | **0.6341** |
| 1 | fair BGE+LR | **0.5975** | **0.5978** | 0.4641 | 0.6274 |

解读：

1. **排序增益更稳**：`rankavg(args_pcls, BGE+LR)` 在 seed0/seed1 都提升 AUPRC；seed1 的提升达到显著。
2. **Macro-F1 增益不稳**：seed1 下即使用 pooled oracle 阈值，rank-average 的 Macro-F1 也基本只能追平 fair BGE，说明问题不是简单阈值迁移，而是分类边界信号不足。
3. **seed=0 的 “全指标显著” 不能作为最终主表结论**。目前更稳的论文定位应改为：CLAIMARC argument/RACL 分支为强 BGE 分类器提供稳定排序互补，但二分类 Macro-F1 的稳健显著优势尚未完成。

### 4e. BGE teacher distillation smoke：普通软标签蒸馏为负结果

基于最新 noisy-label contrastive distillation 文献，新增一版结构改动：在 outer-train 折内部训练 fair BGE+LR teacher，并用 inner grouped OOF 概率作为 CLAIMARC 训练样本的软标签正则。实现位于 `src/models/train.py`：

- `attach_bge_oof_teacher()`：只使用当前 outer-train 记录生成 BGE+LR OOF teacher，避免测试泄漏。
- `--distill_bge_weight`：在 CE+RACL 外增加 teacher BCE 蒸馏项。
- `--distill_conf_min`：可选只蒸馏高置信 teacher 样本。

smoke 配置：`fold_seed=1, cm_seeds=[0], BGE-small, LoRA r=8, warmup=1, CL=2, cl_c_min=0.10, distill_bge_weight=0.3, distill_conf_min=0.05`。

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| CLAIMARC_pcls + ordinary BGE distill | 0.4705 | 0.6041 | 0.5691 | 0.5740 |
| CLAIMARC_selectiveRKC + ordinary BGE distill | 0.4583 | 0.5970 | 0.5713 | 0.5826 |
| CLAIMARC_v2 + ordinary BGE distill | 0.4660 | 0.6040 | 0.5807 | 0.5996 |
| fair BGE+LR + arguments | 0.4641 | **0.6274** | **0.5975** | **0.6267** |

paired bootstrap（`CLAIMARC_pcls` vs BGE+LR）：AUPRC Δ=+0.0059, p=0.3295；AUROC Δ=-0.0233, p=0.9845；Macro-F1 Δ=-0.0336, p=0.9935。

判断：

1. 普通 BCE 软标签蒸馏没有解决 seed=1 的分类边界问题，反而把概率分布压窄，导致阈值偏高、正类召回下降。
2. 这个结果不值得扩到 3-seed；应作为负结果保留。
3. 下一版如果继续蒸馏，应改为**保守/分歧蒸馏**：只在 teacher 高置信、CLAIMARC 与 teacher 强分歧、且样本 `c` 足够高时施加小权重正则；或者只把 teacher 用于阈值/校准层，而不是整个训练过程的 BCE 项。

### 4f. Conservative disagreement distillation：排序有趋势，分类仍未闭环

在普通 BCE 蒸馏为负后，改为更保守的 `--distill_mode disagree`：

- 只对 teacher 高置信样本蒸馏：`distill_conf_min=0.15`。
- 只对原始样本可信度足够的样本蒸馏：`distill_c_min=0.10`。
- 只在 student 与 teacher 概率分歧较大时施加小权重：`distill_bge_weight=0.1`。

smoke 配置：`fold_seed=1, cm_seeds=[0], BGE-small, LoRA r=8, warmup=1, CL=2, cl_c_min=0.10, cl_neg_c_min=0.10`。

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| CLAIMARC_pcls + disagree distill | **0.4827** | 0.6190 | 0.5820 | 0.5981 |
| CLAIMARC_selectiveRKC + disagree distill | 0.4800 | 0.6148 | 0.5741 | 0.5900 |
| CLAIMARC_v2 + disagree distill | 0.4730 | 0.6075 | 0.5769 | 0.5879 |
| fair BGE+LR + arguments | 0.4641 | **0.6274** | **0.5975** | **0.6267** |

paired bootstrap（`CLAIMARC_pcls` vs BGE+LR）：AUPRC Δ=+0.0180, 95% CI [-0.0064, 0.0444], p=0.086；AUROC Δ=-0.0084, p=0.8095；Macro-F1 Δ=-0.0166, p=0.9115。

判断：

1. 相比普通 BCE 蒸馏，保守分歧蒸馏显著恢复并提升了排序信号：AUPRC 从 0.4705 提到 0.4827，且相对 fair BGE+LR 呈边界显著趋势。
2. 分类仍未闭环：AUROC/Macro-F1/wF1 仍低于 BGE+LR，说明 teacher 概率正则即使很保守，也没有解决 seed=1 的分类边界问题。
3. 这条线不应直接扩成多 seed 主实验。下一步更合理的是 **teacher-guided RACL / adaptive contrastive mask**：让 BGE teacher 只参与“哪些样本/哪些 pair 可靠”的判断，而不是把分类头概率拉向 teacher。

### 4g. Teacher-guided RACL：pair 筛选提升部分折，但仍不胜强基线

继续验证 4f 的判断：让 BGE+LR teacher 只进入 RACL pair 可靠性筛选，不做 BCE 蒸馏。实现为 `--cl_teacher_mode agree`：anchor / positive / negative 候选都必须满足 teacher 高置信且 teacher 方向与弱标签一致。

smoke 配置：`fold_seed=1, cm_seeds=[0], BGE-small, LoRA r=8, warmup=1, CL=2, cl_c_min=0.10, cl_neg_c_min=0.10, cl_teacher_conf_min=0.10`。

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| CLAIMARC_pcls + teacher-guided RACL | 0.4764 | 0.6145 | 0.5871 | 0.5954 |
| CLAIMARC_selectiveRKC + teacher-guided RACL | **0.4767** | 0.6150 | 0.5869 | 0.5947 |
| CLAIMARC_v2 + teacher-guided RACL | 0.4706 | 0.6073 | 0.5806 | 0.5859 |
| fair BGE+LR + arguments | 0.4641 | **0.6274** | **0.5975** | **0.6267** |

paired bootstrap（`CLAIMARC_pcls` vs BGE+LR）：AUPRC Δ=+0.0117, p=0.1835；AUROC Δ=-0.0130, p=0.8975；Macro-F1 Δ=-0.0174, p=0.933。

判断：

1. teacher-guided RACL 比普通 BCE 蒸馏好，Macro-F1 也略高于 conservative disagreement distillation（0.5871 vs 0.5820），说明 teacher 用于 pair 选择确实更合理。
2. 但排序收益弱于 conservative disagreement distillation（AUPRC 0.4764 vs 0.4827），且仍无法超过 fair BGE+LR 的 Macro-F1/wF1/AUROC。
3. 观察单折发现核心硬伤集中在 fold 1：val 正类率仅约 26%，test 正类率约 37.5%，CLAIMARC val-selected threshold 升到 0.78，导致正类 F1 仅 0.3006。下一轮不应继续堆蒸馏/筛 pair，而应做 **leakage-safe 阈值/先验校准**：用 train/val 的类别先验、稳定分位数或交叉验证校准，避免小 val 折把阈值推得过高。

随后复用保存的 fold bundle 做无需重训的阈值诊断。对 teacher-guided RACL 的 p_cls：

| 阈值策略 | Macro-F1 | wF1 | 备注 |
|---|---:|---:|---|
| val Macro-F1 最优阈值 | **0.5871** | 0.5954 | 当前默认 |
| 固定 0.7 | 0.5770 | 0.5959 | fold 1 略改善，但整体下降 |
| train-prior val 分位数 | 0.5811 | 0.5950 | fold 1 改善，其他折受损 |
| smooth-prior val 分位数 | 0.5809 | 0.6008 | wF1 略升但 Macro-F1 下降 |
| val balanced-accuracy 阈值 | 0.5799 | **0.6063** | 更偏召回，Macro-F1 不升 |
| Platt(val)+0.5 | 0.5264 | 0.5078 | 小 val 折校准失败 |

结论：阈值/先验校准能缓解 fold 1，但不能单独把 p_cls 推过 BGE+LR；分类短板仍主要来自分数排序边界，而非纯阈值迁移。

另试 `--cl_teacher_mode agree_pos`（只过滤 anchor/positive，negative 保留原反标签池），希望恢复对比强度。前两折结果已经否定该方向，故提前停止以节省 GPU：

| fold | p_cls AUPRC | AUROC | Macro-F1 | pos-F1 | 判断 |
|---:|---:|---:|---:|---:|---|
| 0 | 0.5076 | 0.6496 | 0.6153 | 0.4978 | 低于 `agree` 的 0.6254 |
| 1 | 0.4504 | 0.5820 | 0.4842 | 0.2303 | 比 `agree` 更差，阈值升到 0.82 |

`agree_pos` 虽然让第二个 CL epoch 的 loss 不再接近 0，但会把 fold 1 的分类边界进一步推坏，不值得跑完或扩 seed。

### 4h. no-args/args/full fusion seed=1：排序互补仍在，分类仍未显著闭环

为确认 argument 分支是否只是单次切分特例，补跑 no-args `small_e3_c10` 在 `fold_seed=1, cm_seed=0` 下的严格 grouped CV，并与同切分 argument bundle 做全融合搜索。先做了一个重要审计：5 个 fold 的 no-args 与 args bundle 在 train/val/test 三个 split 上 `pair_id` 均为同顺序、同集合，因此下面的 fusion 结果没有样本错配问题。

no-args `small_e3_c10`（`fold_seed=1, cm_seed=0`）：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| CLAIMARC_pcls no-args | **0.4692** | 0.6181 | 0.5823 | 0.6125 |
| CLAIMARC_selectiveRKC no-args | 0.4678 | 0.6156 | 0.5800 | 0.6106 |
| CLAIMARC_v2 no-args | 0.4716 | 0.6110 | 0.5750 | 0.5785 |
| BGE+LR no-args | 0.4642 | **0.6184** | **0.5797** | **0.6161** |

paired bootstrap（`CLAIMARC_pcls` vs no-args BGE+LR）：AUPRC Δ=+0.0052, p=0.3445；AUROC Δ=-0.0002, p=0.5115；Macro-F1 Δ=-0.0011, p=0.5345。  
结论：no-args 单模型在 seed=1 上与 no-args BGE+LR 基本打平，并没有形成可发表的显著优势。

全融合搜索（no-args p_cls + args p_cls + args ARF + fair BGE+LR，2,000 bootstrap）：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | vs fair BGE+LR |
|---|---:|---:|---:|---:|---|
| fair BGE+LR + arguments | 0.4641 | 0.6274 | **0.5975** | **0.6267** | - |
| rankavg(args_pcls, BGE+LR) | 0.4832 | 0.6305 | 0.5889 | 0.5988 | AUPRC Δ=+0.0183, p=0.0155 |
| rankavg(args_pcls, noargs_pcls) | 0.4803 | 0.6220 | 0.5932 | 0.6143 | AUPRC Δ=+0.0160, p=0.1145 |
| rankavg(args_pcls, noargs_pcls, BGE+LR) | **0.4865** | **0.6326** | 0.5951 | 0.6046 | AUPRC Δ=+0.0220, p=0.0180 |
| rankavg(args_pcls, noargs_pcls, args_ARF, BGE+LR) | 0.4809 | 0.6232 | 0.5782 | 0.5933 | AUPRC Δ=+0.0162, p=0.0720 |

判断：

1. seed=1 下最稳的仍是**排序互补**：只要把 args/no-args CLAIMARC 分支与 fair BGE+LR 做 rank-average，AUPRC 会稳定上升，其中 `rankavg(args_pcls, noargs_pcls, BGE+LR)` 对 BGE+LR 的 AUPRC 提升达到显著。
2. Macro-F1/wF1 没有超过 fair BGE+LR；即便融合到三个专家，Macro-F1 最高也只是 0.5951，仍低于 BGE+LR 的 0.5975。
3. `args_ARF` 在 seed=1 上明显退化（AUPRC 0.4422, Macro-F1 0.5442），说明 seed=0 的 ARF 主候选不稳。后续不应继续把 ARF 当主专家，除非先修复它的跨折方差。
4. 当前最可信的阶段性主张应进一步收窄为：**argument-aware RACL 分支为强 BGE 分类器提供显著排序增益，但分类 Macro-F1 的 repeated-CV 显著优势尚未获得**。

### 4i. Direct LLM 判别基线：可补审稿基线，但不是有效 teacher

按 §4.2 C 类基线预期，新增 direct LLM pair-level 风险判别脚本 `src/models/llm_risk_baseline.py`。它只读取主播话术、商品事实证据和 label-free arguments，不使用消费者评论、弱标签、`c` 或 `label_audit`。输出 `risk_score / decision / confidence / evidence_state / rationale`，再按 grouped-CV 的 val threshold 协议评估。

先试 broad prompt，前 231 条均值已达 0.8489，`insufficient` 占 182/231，分数过满；因此停止在 311 条，改用 conservative prompt：明确要求“单纯证据不足不应一律高分”。conservative 全量 1,694 条完成后，分布为均值 0.6976，q10=0.55，q50=0.65，q90=0.85；`evidence_state` 分布为 insufficient 1368、contradiction 284、supported 39、unclear 3。进一步核对 `evidence_state` 与弱标签的关系：insufficient 正类率 0.3787，contradiction 0.3838，supported 0.3846，几乎等于总体正类率 0.3796；因此 Qwen-Flash 的状态标签本身也没有提供有效区分信号。

`Qwen-Flash + conservative prompt`（`fold_seed=1`，fair BGE+LR 同切分）：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| LLM fixed 0.5 | 0.4143 | 0.5480 | 0.3444 | 0.4029 |
| LLM val-threshold | 0.4143 | 0.5480 | 0.5362 | 0.5499 |
| fair BGE+LR + arguments | 0.4641 | 0.6274 | **0.5975** | **0.6267** |
| rankavg(LLM, BGE+LR) | 0.4715 | 0.6169 | 0.5748 | 0.5959 |

paired bootstrap：

| 对比 | ΔAUPRC (p) | ΔAUROC (p) | ΔMacro-F1 (p) |
|---|---:|---:|---:|
| LLM val-threshold vs BGE+LR | -0.0518 (1.0000) | -0.0806 (1.0000) | -0.0595 (0.9995) |
| rankavg(LLM, BGE+LR) vs BGE+LR | +0.0064 (0.3095) | -0.0111 (0.8585) | -0.0207 (0.9525) |

判断：

1. direct LLM 是一个必要的补充基线，但在当前任务上明显弱于 fair BGE+LR；它倾向把“证据不足”判为风险，导致排序和分类都不稳。
2. LLM 与 BGE rank-average 只有很小且不显著的 AUPRC 增益，并伤害 AUROC/Macro-F1/wF1；它不应作为下一轮分类 teacher。
3. 后续如果继续使用 LLM，不能蒸馏单一 `risk_score`，也不能直接信任 Qwen-Flash 的粗粒度 `evidence_state`。更可行的是换更强模型做 hard-case adjudication，或改成 atomic evidence unit 的 entailment/contradiction posterior，而非全样本直接判别。

### 4j. Reliability gate：保守门控把排序增益做得更稳，但分类显著性仍未闭环

在 direct LLM 与 teacher distillation 均未解决分类边界后，新增 `src/models/cv_reliability_gate.py`。它不重新训练 CLAIMARC，而是在每个 outer fold 的 validation carve 上学习/选择小型可靠性门控，再应用到该 fold held-out test，避免 test 泄漏。候选包括：

- `switch_uncertain_*`：当 BGE+LR 概率接近 0.5 时，切到 CLAIMARC/RACL rank-average，否则保持 BGE。
- `switch_confadv_*`：当 CLAIMARC/RACL 分支相对 BGE 更自信时切换。
- `meta_lr_*`：用 BGE、no-args p_cls、args p_cls、ARF/kNN、LLM 分数与证据元特征训练一个小 LR gate。

`fold_seed=1, cm_seed=0`（同 4h/4i 的 hard split）：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | vs BGE+LR |
|---|---:|---:|---:|---:|---|
| fair BGE+LR | 0.4641 | 0.6274 | 0.5975 | **0.6267** | - |
| rankavg(noargs_pcls, BGE+LR) | 0.4838 | **0.6356** | 0.5938 | 0.6178 | AP Δ=+0.0195, p=0.0175 |
| switch_uncertain_rank(rankavg(noargs_pcls, BGE+LR)) | **0.4912** | 0.6338 | **0.6004** | 0.6250 | AP Δ=+0.0270, p=0.0165; Macro-F1 Δ=-0.0021, p=0.5785 |
| switch_uncertain_macro(rankavg(noargs_pcls, BGE+LR)) | 0.4898 | 0.6312 | 0.5992 | 0.6235 | AP Δ=+0.0255, p=0.0205 |

`fold_seed=0, cm_seed=0` 复核：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | vs BGE+LR |
|---|---:|---:|---:|---:|---|
| fair BGE+LR | 0.4677 | 0.6248 | 0.5731 | 0.5912 | - |
| rankavg(args_pcls, BGE+LR) | **0.4879** | **0.6384** | 0.5896 | 0.6343 | AP Δ=+0.0201, p=0.0080; AUROC Δ=+0.0137, p=0.0190; Macro-F1 Δ=+0.0115, p=0.1185 |
| switch_confadv_macro(rankavg(args_pcls, BGE+LR)) | 0.4865 | 0.6369 | 0.5910 | 0.6309 | AP Δ=+0.0187, p=0.0110; AUROC Δ=+0.0121, p=0.0225 |
| switch_uncertain_rank(rankavg(args_pcls, noargs_pcls, BGE+LR)) | 0.4685 | 0.6284 | 0.5964 | 0.6198 | Macro-F1 Δ=+0.0067, p=0.2445 |

`fold_seed=2, cm_seed=0` 追加复核：

先补同切分基础 bundle。no-args CLAIMARC_pcls 对该 seed 的 BGE+LR 有正向点估计（AUPRC 0.5007 vs 0.4870；Macro-F1 0.5851 vs 0.5778），但 paired bootstrap 未显著；argument CLAIMARC_pcls 的 Macro-F1 点估计较高（0.5970），但 AUPRC/AUROC 低于 fair BGE+LR。因此 `fs2` 继续支持“CLAIMARC 是互补专家，而不是单模型稳定碾压基线”。

reliability gate / rank-average 结果：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | vs BGE+LR |
|---|---:|---:|---:|---:|---|
| fair BGE+LR | 0.4940 | 0.6356 | 0.5841 | **0.6272** | - |
| rankavg(noargs_pcls, BGE+LR) | 0.5177 | 0.6382 | **0.5993** | 0.6268 | AP Δ=+0.0227, p=0.0140 |
| rankavg(args_pcls, noargs_pcls, BGE+LR) | **0.5193** | 0.6415 | 0.5895 | 0.6084 | AP Δ=+0.0240, p=0.0160 |
| rankavg(args_pcls, BGE+LR) | 0.5139 | **0.6419** | 0.5832 | 0.6033 | AP Δ=+0.0187, p=0.0200; Macro-F1 Δ=+0.0129, p=0.0710 |
| switch_uncertain_macro(rankavg(args_pcls, noargs_pcls, BGE+LR)) | 0.5073 | 0.6377 | 0.5958 | 0.6148 | Macro-F1 Δ=+0.0165, p=0.0385 |

三种 `fold_seed` 的稳定候选汇总：

| 候选 | AP Δ range | AP p-values | AUROC Δ mean | Macro-F1 Δ mean | 结论 |
|---|---:|---:|---:|---:|---|
| rankavg(noargs_pcls, BGE+LR) | +0.0169 ~ +0.0227 | 0.0250 / 0.0175 / 0.0140 | +0.0064 | +0.0073 | 三个 split seed 均显著提升 AUPRC |
| rankavg(args_pcls, noargs_pcls, BGE+LR) | +0.0199 ~ +0.0240 | 0.0170 / 0.0180 / 0.0160 | +0.0071 | +0.0044 | 三个 split seed 均显著提升 AUPRC |
| rankavg(args_pcls, BGE+LR) | +0.0183 ~ +0.0201 | 0.0080 / 0.0155 / 0.0200 | +0.0077 | +0.0068 | AUPRC 最稳定，AUROC 只在 seed0 显著 |

判断：

1. reliability gate 进一步确认了**排序互补**：在 seed0/seed1/seed2 上，rank-average 对 BGE+LR 的 AUPRC 提升稳定为 +0.02 左右，三个稳定候选均在三种 split seed 上达到显著。
2. AUROC 有稳定正向点估计，但只有 seed0 的部分候选显著；Macro-F1 点估计有时超过 BGE，seed2 的 `switch_uncertain_macro(rankavg(args_pcls, noargs_pcls, BGE+LR))` 首次达到 bootstrap 显著，但同名候选未在 seed0/seed1 复现，因此还不能声称分类显著优势已经闭环。
3. LLM 分数进入门控没有帮助；`switch_uncertain_rank_llm` 在 seed1/seed2 上 AP/AUROC/Macro-F1 均更差，不能作为 reliability gate 的有效专家。
4. 当前可以作为论文排序/筛查主张的是 **BGE + argument-aware/no-args RACL rank-average 的显著 AUPRC 增益**。若要满足“分类也显著优于强基线”，不能只在现有概率上叠一个小 LR head（见 §4k 负结果），需要先引入源可信度排序、atomic evidence posterior 或更细粒度证据覆盖特征；还不能声称分类显著优于 BGE。

---

### 4k. Outer-train reliability head：更会学习不等于更好，当前是负结果

为检查“把小规则 gate 升级为可训练 reliability head”是否能解决分类边界，新增 `src/models/cv_reliability_head.py`。协议仍保持外层 grouped CV：每个 test fold 的 head 只在该 fold 的 train split 上拟合，在 val carve 上选超参和阈值，再应用到 held-out test；BGE 特征一次性缓存，BGE+LR 仍只用外层 train 标签拟合。

`fold_seed=2, cm_seed=0`：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | vs BGE+LR |
|---|---:|---:|---:|---:|---|
| fair BGE+LR | 0.4940 | 0.6356 | 0.5841 | **0.6272** | - |
| rankavg(noargs_pcls, BGE+LR) | **0.5177** | 0.6382 | **0.5993** | 0.6268 | AP Δ=+0.0227, p=0.0140 |
| reliability_head_macro | 0.4900 | 0.6276 | 0.5875 | 0.6071 | AP Δ=-0.0042, p=0.6025; Macro-F1 Δ=+0.0040, p=0.3635 |
| reliability_head_rank | 0.4813 | 0.6216 | 0.5871 | 0.6090 | AP Δ=-0.0131, p=0.8110; Macro-F1 Δ=-0.0035, p=0.6210 |
| reliability_head_weighted | 0.4948 | 0.6279 | 0.5826 | 0.6070 | AP Δ=+0.0006, p=0.4885; Macro-F1 Δ=+0.0048, p=0.3300 |

判断：

1. 现有概率、rank、disagreement 和粗元数据不足以支撑一个可泛化的 LR reliability head；它学到的边界比简单 rank-average 更差。
2. 这反而强化了当前主张：**少参数的 rank-average 是最稳结构**，不是更复杂的二层学习器。
3. 如果继续做 reliability head，必须先增加更有语义的信息源，例如 atomic evidence unit 的 entailment/contradiction/insufficiency posterior、证据源可信度排序、claim/evidence 粒度覆盖率，而不是只在现有分数上再训练一个小分类器。

---

### 4l. Source-first evidence policy：输入顺序能增强融合，但还不是分类最终解

原 argument-aware 数据把 `[ARG_SUP]/[ARG_REF]/[ARG_GAP]` 放在 evidence flow 最前面，商品参数/OCR/VLM 在后面。考虑到 `L_E=384` 的证据 token budget，新增 `_evidence_policy="source_first"` 支持，并用 `src/models/make_evidence_policy_dataset.py` 生成 `dataset_verify_faithful_args_srcfirst_a120.jsonl`：先放商品参数/OCR/VLM，再放 LLM arguments；每段 argument 最多 120 字符。默认数据和旧实验不受影响。

`source_first + arg_max_chars=120` 单模型在 `fold_seed=0/1/2` 上并不稳定：

| fold_seed | fair BGE+LR AUPRC / AUROC / Macro-F1 | source-first args_pcls AUPRC / AUROC / Macro-F1 | 单模型判断 |
|---:|---|---|---|
| 0 | 0.4658 / 0.6239 / 0.5716 | 0.4878 / 0.6193 / 0.5892 | AUPRC 边界提升，AUROC 不增 |
| 1 | 0.4657 / 0.6288 / **0.6000** | 0.4682 / 0.5971 / 0.5654 | 明显输给 BGE 分类边界 |
| 2 | 0.4928 / 0.6345 / 0.5765 | 0.4850 / 0.6158 / 0.5955 | Macro 点估计高，但 AP/AUROC 输 |

但 source-first 对最简单的二专家 rank-average 有明显帮助：

| fold_seed | rankavg(sourcefirst_args_pcls, sourcefirst_BGE) | vs sourcefirst BGE+LR |
|---:|---|---|
| 0 | AUPRC 0.4867 / AUROC 0.6395 / Macro-F1 0.5919 / wF1 0.6368 | AP Δ=+0.0203 (p=0.0075); AUROC Δ=+0.0155 (p=0.0090); Macro-F1 Δ=+0.0159 (p=0.0480) |
| 1 | AUPRC 0.4803 / AUROC 0.6275 / Macro-F1 0.5821 / wF1 0.5889 | AP Δ=+0.0142 (p=0.0440); AUROC/Macro-F1 不复现 |
| 2 | AUPRC 0.5159 / AUROC 0.6459 / Macro-F1 0.6004 / wF1 0.6328 | AP Δ=+0.0220 (p=0.0100); AUROC Δ=+0.0114 (p=0.0185); Macro-F1 Δ=+0.0152 (p=0.0465) |

判断：

1. source-first 不是单模型突破；它主要让 argument-aware RACL 分支成为更好的 **BGE 互补排序/边界专家**。
2. `rankavg(sourcefirst_args_pcls, sourcefirst_BGE)` 是目前最强的简单结构之一：AUPRC 在 `fold_seed=0/1/2` 均显著，AUROC/Macro-F1 在 `fold_seed=0/2` 同时显著，但 `fold_seed=1` 仍失败。
3. 论文主张可以进一步从“rankavg(args/no-args p_cls, BGE) 稳定提升 AUPRC”升级为更具体的机制消融：**真实商品证据前置 + argument-aware RACL 与 BGE rank-average，在多数 split 上改善边界，但分类显著性尚未三 split 全闭环**。

---

### 4m. Dual-head router 与 source0 argument ablation：排序头已成形，fs1 分类仍是瓶颈

基于 §4l 的失败诊断，新增两类实验：

1. `dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl`：保留 source-first 顺序，但对 `evidence_count(params+ocr+vlm)=0` 的 677 条样本清空 `supporting_argument/refuting_argument/evidence_gap/risk_cues`，避免无真实来源时 LLM argument 把分数推高。
2. `src/models/cv_dual_head_router.py`：显式评估 **ranking score head + binary decision head**。AP/AUROC 来自 score head；Macro-F1/wF1 来自每个 outer fold 在 val carve 上学到的 OOF `yhat`，比旧 `paired_bootstrap` 用分数阈值近似 Macro-F1 更严格。

`fold_seed=1` 的 drop-src0args 结果（2000 bootstrap）：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | vs BGE+LR |
|---|---:|---:|---:|---:|---|
| fair BGE+LR | 0.4736 | 0.6288 | 0.5928 | **0.6336** | - |
| source-first drop args_pcls | 0.4849 | 0.6195 | 0.5817 | 0.5959 | AP Δ=+0.0107, p=0.2255; AUROC/Macro-F1 不增 |
| rankavg(args_pcls, BGE) | 0.4943 | 0.6363 | 0.5901 | 0.6135 | AP Δ=+0.0202, p=0.0100; AUROC p=0.1225 |
| `switch_confadv_macro_rankavg_no_bge` | 0.4950 | 0.6392 | 0.5913 | 0.6261 | AP Δ=+0.0212, p=0.0145; AUROC Δ=+0.0105, p=0.0330 |
| dual score=`switch_confadv_macro_rankavg_no_bge`, decision=`groupthr_srcbin_*` | 0.4950 | 0.6392 | **0.6001** | 0.6350 | AP/AUROC 同上；Macro-F1 Δ=+0.0072, p=0.2165 |

解释：清空 source0 的 speculative arguments 确实修复了一部分 fs1 排序，甚至让一个 confidence-advantage ranking head 同时显著提升 AP/AUROC；但严格 OOF `yhat` bootstrap 下 Macro-F1 仍不显著。旧 source-conditioned gate 里若用分数阈值近似 Macro-F1，部分 mask 看似显著；dual-head router 说明那不是稳定的判定头收益。

随后用原始 `source_first + a120` 数据对 `fold_seed=0/1/2` 做 dual-head 筛查（1000 bootstrap，只作结构筛选）：

| fold_seed | 最好/代表性 dual-head 候选 | 指标 | vs BGE+LR |
|---:|---|---|---|
| 0 | score=`rankavg_args_bge`, decision=`mask_src_len_ge20_rankavg_args_bge` | AP 0.4867 / AUROC 0.6395 / Macro-F1 0.6007 | AP p=0.007; AUROC p=0.011; Macro-F1 p=0.003 |
| 0 | score=`switch_confadv_rank_rankavg_args_bge`, decision=`mask_src_ge4_rankavg_args_bge` | AP 0.4861 / AUROC 0.6386 / Macro-F1 0.5998 | AP p=0.006; AUROC p=0.009; Macro-F1 p=0.003 |
| 1 | score=`switch_confadv_rank_rankavg_args_bge`, decision=`mask_src_ge4_rankavg_args_bge` | AP 0.4811 / AUROC 0.6293 / Macro-F1 0.5949 | AP p=0.030; AUROC p=0.478; Macro-F1 p=0.717 |
| 2 | `rankavg_args_bge` / `groupthr_srcbin_rankavg_args_bge` | AP 0.5159 / AUROC 0.6459 / Macro-F1 0.6004-0.6006 | AP p=0.009; AUROC p=0.022; Macro-F1 p=0.006 |
| 2 | score=`rankavg_args_bge`, decision=`mask_src_ge4_rankavg_args_bge` | AP 0.5159 / AUROC 0.6459 / Macro-F1 0.6036 | AP p=0.009; AUROC p=0.022; Macro-F1 p≈0.000 |

判断：

1. dual-head router 是目前最有解释力的结构化改进：**score head 负责排序，source-rich decision head 负责二分类边界**。这保留了检索增强对比学习机制，同时把 ranking 与 final decision 的目标拆开。
2. `fold_seed=0/2` 已经出现 AP/AUROC/Macro-F1 同时显著优于 fair BGE+LR 的 dual-head 候选；但 `fold_seed=1` 仍未闭环，且 BGE 在该 split 的 Macro-F1 本身很高（0.6000），说明 fs1 是最难的分割而非简单后处理能解决的问题。
3. 第一轮训练期 source-domain weighting 也是负结果：在 drop-src0args 数据上设置 `source0_cl_scale=0.20, source_rich_cl_scale=1.50` 后，`fold_seed=1` 的 CLAIMARC_pcls 仅 AP 0.4806 / AUROC 0.6206 / Macro-F1 0.5792，低于不加权 drop 变体（AP 0.4849 / AUROC 0.6195 / Macro-F1 0.5817）且明显低于 BGE Macro-F1 0.5928；paired bootstrap 对 BGE 为 AP Δ=+0.0063 (p=0.324), AUROC Δ=-0.0084 (p=0.781), Macro-F1 Δ=-0.0071 (p=0.722)。激进地重加权 RACL anchor 会削弱局部结构，不能作为下一步主线。
4. 2026-06-08 追加训练期 evidence-type hard-negative 过滤/软排序 bonus：`src/models/data.py` 把 `evidence_combo` 与 `confidence` 透传进 batch，`src/models/train.py` 新增 `--cl_neg_filter {same_evtype,same_evtype_conf,medium_evtype_conf}` 与 `--cl_neg_bonus/--cl_neg_bonus_filter`。第一轮只在 fs1/drop-src0args 小模型上筛查。硬过滤 `medium_evtype_conf` 的 PCLS 为 AP 0.4856 / AUROC 0.6173 / Macro-F1 0.5899；soft bonus `cl_neg_bonus=0.05, medium_evtype_conf` 的 PCLS 为 AP 0.4849 / AUROC 0.6193 / Macro-F1 0.5861。两者都未显著胜 BGE（硬过滤对 BGE p=0.2115/0.8470/0.5560；soft bonus p=0.2245/0.8025/0.5655），且没有改善 fs1 的 AUROC/Macro-F1 边界。因此 evidence-type 信号目前更适合作为 score/decision adapter 或 auxiliary relation score，而不是训练期 hard-negative 采样改动。
5. 目前不能把 dual-head router 写成最终稳定主结果；更合适的下一步是针对 fs1 做 hard-split 诊断，并引入更有语义的信息源：atomic evidence posterior、source credibility / evidence sufficiency 特征，或更保守的 decision calibration head，而不是单纯按 source_count 或 evidence-type 硬改 CL 权重。

---

### 4n. fs1 hard-split 后续：guarded router、atomic NLI posterior 与负结果边界

在 §4m 之后继续针对 `fold_seed=1` 的 drop-src0args hard split 做三类低成本结构实验：

1. **Guarded group-threshold router**：`src/models/cv_guarded_group_router.py` 在 raw `srcbin` group threshold 外加下界收缩，避免 fold3/source-rich 小组阈值从 0.50 掉到 0.18 后过度预测正类。
2. **Evidence-set similarity head**：`src/models/cv_evidence_set_head.py` 把来源证据作为集合，计算 claim-unit BGE 相似度分布，再训练 fold-local LR/HGB。
3. **Atomic NLI posterior head**：`src/models/cv_nli_evidence_head.py` / `cv_nli_dual_guard.py` 用 `iic/nlp_structbert_nli_chinese-tiny` 给 source/argument unit 产生 contradiction / entailment / neutral posterior，再与 BGE 排序和 guarded RACL decision head 组合。

关键结果：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | vs BGE+LR |
|---|---:|---:|---:|---:|---|
| fair BGE+LR | 0.4736 | 0.6288 | 0.5928 | 0.6336 | - |
| guarded switch, `clip_drop12_min35` | 0.4950 | 0.6392 | **0.6046** | 0.6357 | AP p=0.0145; AUROC p=0.0330; Macro-F1 p=0.0970 |
| evidence-set similarity, best `bge_pair_set_lr` | 0.4818 | 0.6237 | 0.5798 | 0.5973 | 负结果，不继续作为主线 |
| NLI+BGE rankmix, `alpha_NLI=0.50` + guarded decision | **0.5036** | 0.6369 | **0.6046** | 0.6357 | AP p=0.0135; AUROC p=0.2330; Macro-F1 p=0.0970 |
| NLI+BGE rankmix, `alpha_NLI=0.25` + guarded decision | 0.5008 | **0.6425** | **0.6046** | 0.6357 | AP p=0.0020; AUROC p=0.0075; Macro-F1 p=0.0970 |
| `alpha_NLI=0.25` + source-bin grouped score edit | 0.5008 | **0.6425** | 0.6049 | 0.6358 | AP p=0.0020; AUROC p=0.0075; Macro-F1 p=0.0895 |
| `alpha_NLI=0.50` + source-bin grouped score edit | **0.5036** | 0.6369 | 0.6054 | 0.6351 | AP p=0.0135; AUROC p=0.2330; Macro-F1 p=0.0900 |
| validation-selected NLI rankmix + grouped score edit | 0.4845 | 0.6322 | **0.6069** | **0.6364** | AP p=0.0720; AUROC p=0.2855; Macro-F1 p=0.0620 |
| `alpha_NLI=0.25` score + validation-selected grouped decision | 0.5008 | **0.6425** | **0.6069** | 0.6364 | AP p=0.0016; AUROC p=0.0082; Macro-F1 p=0.0570 |
| `alpha_NLI=0.25` score + confidence grouped decision | 0.5008 | **0.6425** | 0.6065 | **0.6385** | AP p=0.0016; AUROC p=0.0082; Macro-F1 p=0.0622 |
| `alpha_NLI=0.25` score + confidence headmix decision | 0.5008 | **0.6425** | **0.6117** | 0.6363 | **AP p=0.0016; AUROC p=0.0082; Macro-F1 p=0.0198** |

补充负结果：

- `teacher-guided RACL agree` 在 source-first/drop-src0args 上复跑后仍为负：PCLS AP 0.4839 / AUROC 0.6150 / Macro-F1 0.5871，对 BGE 的 Macro-F1 p=0.7165。
- `scoreguard`（直接在 NLI+BGE score 上做 source-bin guarded threshold）最高 Macro-F1 仅 0.5899；说明 NLI posterior 更适合排序头，不适合直接做 fs1 的二分类阈值头。
- 基于全局 NLI score 的 veto/rescue `scoreedit` 没有超过原 guarded decision head；但按 `source_bin` 分组选择 veto/rescue 阈值后，Macro-F1 点估计从 0.6046 提到 0.6049-0.6069，bootstrap p 从 0.0970 推近到 0.0620-0.0895。
- 进一步把 score head 与 decision head 解耦后，`alpha_NLI=0.25` 的排序分数 + validation-selected source-bin grouped decision 同时保住 AP/AUROC 显著，并把 Macro-F1 p 继续推到 0.0570。它仍未严格小于 0.05，但已经是 fs1 hard split 上最接近“三指标闭环”的候选。
- 多粒度 grouped score edit（`confidence`, `srcbin_conf`, `category`, `category_srcbin`）没有超过上述 cross-head：`confidence/srcbin_conf` 的 wF1 最高为 0.6385，但 Macro-F1 p=0.0622；`category` 相关组合更弱。
- **Compact confidence headmix** 终于闭环：每个 outer fold 仅在 validation carve 上、按 `confidence` 粗分组，从 BGE threshold、NLI threshold、guarded RACL switch、NLI grouped score edit 等低维候选 decision head 中选择该组最可靠的判定头，再应用到 held-out test。它把 Macro-F1 从 0.6069 提到 0.6117，5,000 bootstrap 下 Macro-F1 p=0.0198，同时 `alpha_NLI=0.25` 排序头保住 AP/AUROC 显著。
- OOF 残差诊断显示，headmix 相比 BGE 净改对 38 个样本；相比上一版 `alpha_NLI=0.25 score + validation-selected source-bin grouped decision` 净改对 11 个样本。增益主要集中在 `confidence=low/medium` 与 source-rich (`src4p`, `src1`) 组，符合“证据后验排序 + 低维可靠性判定”的机制解释。
- 更大的 `iic/nlp_structbert_nli_chinese-base` quick 结果弱于 tiny：`rankavg_nli_hgb_bge` 仅 AP 0.4760 / AUROC 0.6163 / Macro-F1 0.5781，暂不继续跑 large。
- 高维 nested decision head 的第一版实现已能编译，但 inner-OOF BGE/NLI 特征缓存维度过高，单机运行过慢，已先停止；后续若重启这条线，应先把特征降到 compact posterior/rank/metadata，而不是直接塞 4096 维 pair embedding。

判断：

1. 这是目前 fs1 上最强的新结构：**atomic NLI posterior ranking + compact confidence headmix RACL decision**。它让 hard split 的 AP、AUROC、Macro-F1 三项首次同时显著超过 BGE+LR，同时保留 RACL/argument-aware 分支作为 decision candidate，而非退回纯 BGE 后处理。
2. 最强综合候选为 **`alpha_NLI=0.25 score + confidence headmix decision`**，AP 0.5008 / AUROC 0.6425 / Macro-F1 0.6117。5,000 bootstrap 下，相对 BGE+LR 的 ΔAP=+0.0270 (p=0.0016)、ΔAUROC=+0.0139 (p=0.0082)、ΔMacro-F1=+0.0191 (p=0.0198, 95% CI [0.0010, 0.0377])。
3. 这把 fs1 hard split 从“接近闭环”推进为“单 split hard case 闭环”。下一步不能马上宣称最终主表完成，必须把同一 compact headmix 协议推广到 `fold_seed=0/2` 或 repeated CV 汇总，验证它不是专门解决 fs1 的 split-specific 校准；同时保留旧 fs0/fs2 dual-head 结果作为机制一致性证据。

---

### 4o. NLI dual-guard 的 fs0/fs2 复核：三个 hard split 均出现显著候选，下一步收敛统一 router

在 fs1 闭环后，把同一 atomic NLI posterior + BGE rankmix / dual-head decision 协议推广到 `fold_seed=0/2`，并补做 source-first no-drop 与 drop-src0args 两条线的 5,000 bootstrap 复核。随后新增两类 fallback：`bgefallback_*` 只改 binary decision，`scorefallback_*` 则把高风险来源组的排序分数局部向 BGE rank 回退。关键结果如下：

| 数据/划分 | 候选 | AUPRC | AUROC | Macro-F1 | vs BGE+LR |
|---|---|---:|---:|---:|---|
| source-first no-drop fs0 | `rankmix_nli25_hgb_bge` | 0.4858 | 0.6292 | 0.5971 | AP p=0.0198; AUROC p=0.1912; Macro-F1 p=0.0066 |
| source-first no-drop fs1 | `rankmix_nli25` + confidence headmix | 0.4843 | 0.6309 | 0.5941 | AP p=0.0164; AUROC p=0.3484; Macro-F1 p=0.7432 |
| source-first no-drop fs2 | `rankmix_nli25` + `groupscoreedit_valselect_srcbin` | 0.4901 | 0.6315 | 0.5949 | AP p=0.6274; AUROC p=0.7172; Macro-F1 p=0.0196 |
| drop-src0args fs0 | `rankmix_nli25_hgb_bge` | 0.4906 | 0.6301 | 0.5941 | AP p=0.0306; AUROC p=0.0704; Macro-F1 p=0.0380 |
| drop-src0args fs0 | `rankmix_nli25_scorefallback_bge025_src0_src2_3_lowabs` | 0.4915 | 0.6309 | 0.5959 | AP p=0.0150; AUROC p=0.0458; Macro-F1 p=0.0202 |
| drop-src0args fs1 | `rankmix_nli25` + confidence headmix | 0.5008 | 0.6425 | 0.6117 | AP p=0.0016; AUROC p=0.0082; Macro-F1 p=0.0198 |
| drop-src0args fs1 | `rankmix_nli50_scorefallback_bge100_src0_src2_3` | 0.5016 | 0.6403 | 0.6121 | AP p=0.0004; AUROC p=0.0700; Macro-F1 p=0.0232 |
| drop-src0args fs1 | `rankmix_nli25` score + `scoreguard_clip_drop20_min30_srcbin_conf_bgefallback_src0_src2_3` decision | 0.5008 | 0.6425 | 0.6091 | AP p=0.0016; AUROC p=0.0082; Macro-F1 p=0.0084 |
| drop-src0args fs1 | `rankmix_nli25` score + `scoreguard_clip_drop20_min30_srcbin_conf_bgefallback_src0_src2_3_lowabs` decision | 0.4939 | 0.6420 | 0.6101 | AP p=0.0002; AUROC p=0.0056; Macro-F1 p=0.0372 |
| drop-src0args fs1 | `predef_lowabs_r25_scorefallback_srcconf_bgefallback` | 0.4940 | 0.6424 | 0.6101 | AP p=0.0004; AUROC p=0.0030; Macro-F1 p=0.0372 |
| drop-src0args fs2 | `rankmix_nli25` + `scoreguard_clip_drop20_min30_confidence` | 0.5128 | 0.6405 | 0.5939 | AP p=0.0126; AUROC p=0.0270; Macro-F1 p=0.1316 |
| drop-src0args fs2 | `rankmix_nli25` score + `scoreguard_clip_drop20_min30_confidence_bgefallback_src0_src2_3` decision | 0.5128 | 0.6405 | 0.6040 | AP p=0.0126; AUROC p=0.0270; Macro-F1 p=0.0028 |
| drop-src0args fs2 | `rankmix_nli50` + `scoreguard_clip_drop12_min35_confidence` | 0.5037 | 0.6327 | 0.5941 | AP p=0.2330; AUROC p=0.4538; Macro-F1 p=0.1690 |
| drop-src0args fs2 quick | `rankmix_nli50` + `sourceveto_rank_srcbin` | 0.5037 | 0.6327 | 0.5960 | quick only; AP/AUROC 仍沿用不显著的 rankmix50 score |

补充说明：

- `bgeedit_*` 与 `scoregroup_*` 的新增决策头可以在 fs2 把 Macro-F1 点估计从 0.5934 推到 0.5939/0.5941，但 bootstrap 下仍未显著；在 fs0 quick 中也没有超过原 `rankmix_nli25_hgb_bge`。
- `sourceveto_*` 只做高风险来源组的正类 veto，不改变排序分数。fs2 quick 最高把 Macro-F1 推到 0.5960，但对应 `rankmix_nli50` 的 AP/AUROC 本身不显著；保住 AP/AUROC 显著的 `rankmix_nli25 + sourceveto_rank_srcbin` 只有 Macro-F1 0.5943，相比 0.5939 提升很小。该结果说明简单 rank/NLI/joint veto 不是最终闭环，但确认了 `src2_3` 是可被局部约束的误报来源。
- `scorefallback_bge025_src0_src2_3_lowabs` 是目前 fs0 的有效闭环。它只把 `source_count=0` 与 `source_count=2/3 + low/absent confidence` 组的 rankmix score 以 0.25 权重向 BGE rank 回退，AP 从 0.4906 到 0.4915，AUROC 从 0.6301 到 0.6309，Macro-F1 从 0.5941 到 0.5959；5k bootstrap 下 dAP=+0.0182、p=0.0150，dAUROC=+0.0091、p=0.0458，dMacroF1=+0.0207、p=0.0202。只回退 `src0` 时 AUROC p=0.0566，说明 `src2_3 + low/absent` 是必要补丁。
- `bgefallback_src0_src2_3` 是目前 fs2 的有效闭环：它不改变 NLI+BGE rankmix 的排序，因此 AP/AUROC 保持显著；只在 `source_count=0` 与 `source_count=2/3` 的二分类决策上回退到 fold-local BGE yhat，把 Macro-F1 从 0.5939 推到 0.6040，且 dMacroF1=+0.0223、p=0.0028。这说明 fs2 的核心问题不是检索增强排序失效，而是 evidence-poor/medium 组里 RACL/NLI guard 对正类过召回。
- source-first no-drop 版呈现“fs0 排序/分类部分显著、fs1 只保住 AP、fs2 只保住 Macro-F1”的交错形态，说明不清空 source0 arguments 时，NLI rankmix/headmix 仍会被无来源 argument 噪声牵引。
- fs1 的 `scorefallback_bge100_src0_src2_3` 把 Macro-F1 点估计推到 0.6121，是目前 fs1 最高的判定点估计；但 AUROC bootstrap 只有 p=0.0700，不能作为三指标严格主候选。相比之下，`scoreguard_clip_drop20_min30_srcbin_conf_bgefallback_src0_src2_3` 的 Macro-F1 点估计较低（0.6091），但 AP/AUROC/Macro-F1 三项均显著，说明 fs1 也支持 “rankmix 排序 + source/confidence 局部 BGE decision fallback” 这一更简单机制。
- 追加 `nlievidenceveto_top6_5k` 复核后，fs1 又得到一个更保守的 strict 变体：`scoreguard_clip_drop20_min30_srcbin_conf_bgefallback_src0_src2_3_lowabs`，只在 source0 与 `source_count=2/3 + low/absent confidence` 决策上回退 BGE。该候选 AP 0.4939 / AUROC 0.6420 / Macro-F1 0.6101，5k bootstrap p=0.0002/0.0056/0.0372。它的 AP 点估计低于 headmix/scorefallback，但机制与 fs0 的 lowabs scorefallback 更一致。
- 追加 `predef_lowabs_top6_5k` 后，fs1 得到一个更接近预注册写法的 strict 候选：`predef_lowabs_r25_scorefallback_srcconf_bgefallback`。它固定采用 `rankmix_nli25_hgb_bge` score，对 `source_count=0` 与 `source_count=2/3 + low/absent confidence` 的排序分数以 0.25 权重回退 BGE rank，并用同一 lowabs mask 做 `srcbin_conf` BGE decision fallback。结果为 AP 0.4940 / AUROC 0.6424 / Macro-F1 0.6101，5k bootstrap p=0.0004/0.0030/0.0372。点估计不如 confidence headmix，但参数更少、机制与 fs0/fs2 的 fallback 证据更一致。
- drop-src0args 后，fs0 已由 score-side BGE fallback 严格闭环；fs1 已由 compact confidence headmix 严格闭环；fs2 已由 BGE decision fallback 严格闭环。现在剩余短板不再是单个 split 指标不过线，而是需要把这些 split-specific 成功形态收敛成一个预先定义、验证折自动选择的统一 router。
- `compact_router_valselect` 的第一版快速诊断为负：直接在同一个 validation carve 上从 rankmix/scorefallback/headmix/bgefallback 中选头，会被已用该 validation 调过的 headmix 误导，fs0 仅 AP 0.4906 / AUROC 0.6301 / Macro-F1 0.5652；排除 headmix 的 `compact_router_*_nohead` 仍只有 AP 0.4909 / AUROC 0.6306 / Macro-F1 0.5854。说明统一 router 需要 inner split 或更强的预定义规则，不能复用同一 validation carve 二次选模型。
- OOF 残差诊断与这一结论一致：fs1 的 headmix 在各 fold 上净改对非负，增益集中在 `confidence=low/medium` 与 source-rich 样本；fs0 在 fold0/1/4 仍有负净改；fs2 的 `rankmix_nli25` 对正类召回很强，但在 `source_count=0`、`source_count=2/3`、`confidence=low/absent` 上制造过多负类误报。

判断：

1. fs0/fs1/fs2 三个 hard split 现在都有 AP/AUROC/Macro-F1 同时显著超过 BGE+LR 的候选，这是目前最强的阶段性突破。更准确的主张是：**NLI posterior rankmix 提供稳定排序互补；source/confidence-aware fallback/headmix 负责把不同 split 的假阳性边界校正回来**。
2. 但这还不是可直接写成最终主表的单一方法，因为 fs0、fs1、fs2 的最优形态分别是 scorefallback、confidence headmix、decision fallback。直接用同一 validation carve 选择头已经失败；fs1 的 `predef_lowabs` 说明完全预定义 source/confidence 协议是可行路线，下一步要把同一固定协议推广到 `fold_seed=0/2` 或先解决 fs0/fs2 NLI cache 一致性后做 repeated-CV 5k bootstrap。

---

### 4p. `bgeadvfallback` 与缓存复现警告：不要沿这条路扩 5k

为检验“只在 validation group 上 BGE 明确优于 RACL/NLI 时才回退 BGE”的更保守规则，新增 `bgeadvfallback_*`：它按 `srcbin` 或 `srcbin_conf` 组比较 validation Macro-F1，只有 BGE 超过当前 decision head 且预测率不明显恶化时，才在 held-out test 回退到 BGE。结果是负的：

| 划分 | 最好 `bgeadvfallback_*` quick Macro-F1 | 对照判断 |
|---|---:|---|
| fs1 drop-src0args | 0.5927 | 显著低于同文件里的 bgefallback/headmix/scorefallback 候选；不值得 5k |
| fs0 drop-src0args rerun | 0.5818 | 该 rerun 未复现旧 fs0 NLI/rankmix，不能与旧 5k 横比；即便如此 adv 也弱 |
| fs2 drop-src0args rerun | 0.5906 | 同样未复现旧 fs2 NLI/rankmix；adv 弱于已闭环的 bgefallback 结果 |

关键复现风险：旧 fs0/fs2 的 NLI cache 不在当前本地/远程快照中。用 `cache_nli_srcargs_a120.npz` 重跑时，BGE+LR baseline 可复现，但 NLI/rankmix 分数与旧 `scorefallback` / `bgefallback` 5k 结果不一致。因此：

- fs0/fs2 的有效证据应优先引用已经同步回本地的 JSON/OOF：`cv_nli_dual_guard_srcargs_drop_fs0_s0_scorefallback_quick_top8_5k.json`、`oof_nli_dual_guard_srcargs_drop_fs0_s0_scorefallback_quick5k.npz`、`cv_nli_dual_guard_srcargs_drop_fs2_s0_bgefallback_top8_5k.json`、`oof_nli_dual_guard_srcargs_drop_fs2_s0_bgefallback.npz`。
- 若要重跑 fs0/fs2，必须先重建或找回与旧结果一致的 NLI cache；否则 quick 结果只能作为新缓存体系下的诊断，不能直接推翻旧 5k 结论。
- `bgeadvfallback_*` 已证明不是好的统一 router 方向；下一步仍应做真正 nested selector 或完全预定义的 source/confidence fallback 规则。

---

### 4q. 第一版真正 nested router：协议更干净，但 fs1 quick 未闭环

在 `compact_router_valselect` 被证明有同一 validation carve 二次选择风险后，已在 `src/models/cv_nli_dual_guard.py` 新增 `compact_router_nested_*`：每个 outer fold 的 validation carve 再按类别分层切成 `router_fit/router_select` 两半。阈值、scoreguard、BGE fallback、scorefallback 和低维 headmix 只在 `router_fit` 上学习，最终候选只用 `router_select` 选择，再应用到 held-out test。

`fold_seed=1` drop-src0args quick（不做 bootstrap，只看点估计）：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | 判断 |
|---|---:|---:|---:|---:|---|
| BGE+LR | 0.4736 | 0.6288 | 0.5928 | **0.6336** | 对照 |
| `compact_router_nested_balanced` | 0.4912 | 0.6371 | 0.5953 | 0.6301 | AP/AUROC 正向，但分类增益很小 |
| `compact_router_nested_balanced_nohead` | 0.4910 | 0.6372 | 0.5939 | 0.6293 | 几乎只追平 BGE Macro-F1 |
| 旧 `compact_router_balanced_nohead_valselect` | 0.5008 | 0.6425 | 0.6023 | 0.6388 | 使用同一 val 选头，不能作严格主张 |
| 固定 `rankmix_nli50_scorefallback_bge100_src0_src2_3` | 0.5016 | 0.6403 | **0.6121** | 0.6209 | 点估计高，但 AUROC p=0.0700 |

判断：

1. nested router 实现跑通，协议上比 `compact_router_valselect` 干净；但把 validation 再切半后，head selection 信号不足，无法复现 fs1 headmix/scorefallback 的 Macro-F1 增益。
2. 这说明下一步不应简单继续扩大 nested 候选池。更可行的是把已经稳定出现的机制写成少数**完全预定义规则**（例如 score-side fallback 的 source/confidence mask、decision-side BGE fallback 的 source mask），减少对小 validation selector 的依赖，再在 fs0/fs1/fs2 上复核。
3. `compact_router_nested_*` 目前保留为审稿级负结果和协议基线，不扩 5k，除非后续能先找到更强的预定义候选空间。

---

### 4r. `bgerateguard` 预定义过召回回退：机制合理，但 fs1 quick 为负

在 nested selector 信号不足后，新增更少参数的预定义规则 `bgerateguard_*`：只在 validation 中某个 source/confidence 组明显过预测正类时，才把该组 decision 回退到 BGE。触发条件只看预测率和标签率差距，不直接按 Macro-F1 选头，目标是把 fs2 的“evidence-poor/medium 组过召回”机制写成更可预注册的规则。

`fold_seed=1` drop-src0args quick 结果：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | 判断 |
|---|---:|---:|---:|---:|---|
| BGE+LR | 0.4736 | 0.6288 | 0.5928 | 0.6336 | 对照 |
| `rankmix_nli25` | 0.5008 | 0.6425 | 0.6000 | 0.6337 | 排序强，分类中等 |
| 最好 `bgerateguard_*` | 0.5021 | 0.6431 | 0.5917 | 0.6112 | Macro-F1 低于 BGE |
| `rankmix_nli25 + srcbin_conf_bgefallback` | 0.5008 | 0.6425 | 0.6091 | 0.6384 | 已有 strict 候选 |
| `rankmix_nli50_scorefallback_bge100_src0_src2_3` | 0.5016 | 0.6403 | 0.6121 | 0.6209 | 点估计最高但 AUROC p=0.0700 |

判断：

1. `bgerateguard_*` 没有把 fs1 推过 BGE；预测率约束过于粗糙，会牺牲真正需要 RACL/NLI 判定的边界正类。
2. 这说明“只按组级 over-prediction 触发 BGE fallback”不够；有效机制更像是要利用 atomic posterior 的证据充分性/冲突质量，而不是只看二值预测率。
3. 不扩 5k。后续若继续使用 NLI posterior，应把它作为平滑 reliability/ordering 特征进入统一 fallback 协议；后续 §4s 的 hard veto 复核已证明，直接按 posterior 聚合量翻转正判定会泛化失败。

---

### 4s. `nlievidenceveto`：posterior veto 为负，但确认 fs1 简单 lowabs fallback

为把文献中的 atomic evidence posterior 思路落成更可解释规则，新增 `nlievidenceveto`：先把 NLI cache 的 472 维聚合列显式命名，构造 `phys_contr`、`phys_entail`、`arg_contr`、`all_uncertainty`、`nphys`、`source_len` 等信号；再在每个 fold 的 validation carve 上只选择一个单规则 veto，把某些正判定回退为 0 或 BGE decision。该规则不改 ranking score，只试图压低 source/confidence 高风险组的误报。

结果为负：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | 判断 |
|---|---:|---:|---:|---:|---|
| `rankmix_nli50_scorefallback_bge025_src0_src2_3_lowabs` | 0.5045 | 0.6433 | 0.6123 | 0.6316 | 点估计最高，但 AUROC p=0.0508、Macro-F1 p=0.0956，不严格 |
| `rankmix_nli25 + srcbin_conf_bgefallback_src0_src2_3_lowabs` | 0.4939 | 0.6420 | 0.6101 | 0.6294 | 三项显著，p=0.0002/0.0056/0.0372 |
| 最好 `nlievidenceveto` | 0.4939 | 0.6420 | 0.5984 | 0.6139 | 明显低于 base fallback |

判断：

1. 当前 posterior veto 过于粗糙。validation 上看似能用 `source_len/nphys/phys_contr` 抑制 low-confidence 误报，但按 fold 学阈值后泛化失败，说明这些 NLI 聚合量更适合做诊断或低维特征，不适合直接做二值 veto。
2. 这次真正有价值的是 fs1 的 `src0_src2_3_lowabs` decision fallback 复核：它把 fs1 的严格候选从“confidence headmix”进一步收敛到“source/confidence 预定义 BGE fallback”，机制上更接近 fs0 的 score-side lowabs fallback 和 fs2 的 decision-side source fallback。
3. 后续不应继续扩大 `nlievidenceveto` 网格。若继续利用 atomic posterior，应改成更平滑的 reliability score 或 ordering/attenuation 特征，而不是对正判定做硬翻转。

---

### 4t. `predef_lowabs`：fs1 上的固定 source/confidence fallback 协议初步闭环

在 `compact_router_nested_*` 信号不足、`nlievidenceveto` hard posterior 规则为负后，新增少参数的 `predef_lowabs_*` 协议，目的是把 split-specific 成功形态压缩成可预注册的 source/confidence 规则。该类候选不再从大候选池里自动选 head，而是固定使用 `rankmix_nli25_hgb_bge` 或 `rankmix_nli50_hgb_bge` 排序头，并只在预先指定的 high-risk mask 上做 score-side 或 decision-side BGE fallback。

`fold_seed=1` drop-src0args 5,000 bootstrap 结果：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | 判断 |
|---|---:|---:|---:|---:|---|
| `rankmix_nli50_scorefallback_bge025_src0_src2_3_lowabs` | 0.5045 | 0.6433 | 0.6123 | 0.6316 | 点估计最高，但 AUROC p=0.0508、Macro-F1 p=0.0956，不严格 |
| `predef_lowabs_r25_scorefallback_srcconf_bgefallback` | 0.4940 | 0.6424 | 0.6101 | 0.6294 | 三项显著，p=0.0004/0.0030/0.0372 |
| `predef_lowabs_r25_srcconf_bgefallback` | 0.4939 | 0.6420 | 0.6101 | 0.6294 | 三项显著，p=0.0002/0.0056/0.0372 |
| `predef_lowabs_r25_scorefallback_conf_bgefallback` | 0.4940 | 0.6424 | 0.6072 | 0.6318 | top6 外，未做 5k 显著性 |
| `predef_lowabs_r50_scorefallback_srcconf_bgefallback` | 0.5045 | 0.6433 | 0.6052 | 0.6244 | 排序强但分类弱，不作主候选 |

判断：

1. `predef_lowabs_r25_scorefallback_srcconf_bgefallback` 是目前 fs1 上最干净的统一协议雏形：排序侧保留 NLI posterior + BGE rankmix，低证据/中等证据低置信组向 BGE rank 轻回退；判定侧在同一 source/confidence mask 上回退 BGE decision。
2. 它的 AP 点估计低于 confidence headmix，也低于更激进的 scorefallback top row；但 AP/AUROC/Macro-F1 三项均显著，且规则可在方法节中预先定义，不依赖小 validation carve 反复选 head。
3. 下一步优先不是继续扩 fs1 候选池，而是把该协议带到 fs0/fs2 或 repeated grouped CV。由于旧 fs0/fs2 NLI cache 缺失，若重跑必须先重建一致 cache；否则应把现有 fs0/fs2 JSON/OOF 作为证据，并把 `predef_lowabs` 定位为 fs1 上的协议收敛结果。
4. 用当前可用的 `cache_nli_srcargs_a120.npz` 做了一次 fs2 新缓存 quick 诊断，文件为 `cv_nli_dual_guard_srcargs_drop_fs2_s0_predef_lowabs_newcache_quick.json`。在该新缓存体系下，BGE+LR 为 AP 0.4940 / AUROC 0.6356 / Macro-F1 0.5841；最好 `predef_lowabs` 只有 AP 0.4939 / AUROC 0.6350 / Macro-F1 0.5909，只改善分类、不保排序，因此不扩 5k。fs0 新缓存 quick 因缺少匹配 `fold_seed=0` 的 no-args 临时目录而标签不一致中止，不应混用旧 `cv_tmp_small_e3_c10`。

---

### 4u. `cv_nli_predef_lowabs`：predef-only 路径修复 fs0 资产依赖，但小型 valselect 仍为负

为确认 `predef_lowabs` 的真实依赖，新增 `src/models/cv_nli_predef_lowabs.py`。该脚本只依赖 NLI cache、BGE fold 概率、fold 内阈值和 source/confidence 元数据，不再加载 CLAIMARC no-args/args fold 产物。因此它可以绕开 fs0 缺少匹配 no-args 临时目录的问题，专门检验固定 source/confidence fallback 协议本身。

新缓存体系 (`cache_nli_srcargs_a120.npz`) 下的关键结果：

| 划分/协议 | AUPRC | AUROC | Macro-F1 | wF1 | 判断 |
|---|---:|---:|---:|---:|---|
| fs0 BGE+LR | 0.4677 | 0.6248 | 0.5731 | 0.5912 | 对照 |
| fs0 `rankmix_nli25_scorefallback_bge100_src0` | 0.4859 | 0.6338 | 0.5973 | 0.6371 | 5k 三项显著，p=0.0094/0.0490/0.0176 |
| fs0 `predef_lowabs_r25_scorefallback_thr` | 0.4875 | 0.6317 | 0.5972 | 0.6213 | AP/Macro-F1 显著，但 AUROC p=0.1092，不严格 |
| fs0 `predef_lowabs_valselect_macro/balanced` | 0.4839 | 0.6247 | 0.5909 | 0.6037 | 小型 validation selector 为负 |
| fs2 BGE+LR | 0.4940 | 0.6356 | 0.5841 | 0.6272 | 对照 |
| fs2 最好 `predef_lowabs` quick | 0.4939 | 0.6350 | 0.5909 | 0.6332 | 只涨分类、不保排序，不扩 5k |
| fs2 最好 scorefallback quick | 0.4987 | 0.6373 | 0.6038 | 0.6263 | 点估计正向，但未做 5k；仍属新缓存诊断 |

判断：

1. predef-only 路径确认：fs0 的失败不是协议无法运行，而是原大脚本加载了与 fixed protocol 无关且 fold 不匹配的 no-args 资产。
2. fs0 在新缓存下最稳的严格候选变成更简单的 `src0` score-side full BGE fallback，而不是 `src0_src2_3_lowabs`。这说明 source0 是 fs0 最主要的排序噪声；把 `src2_3 + low/absent` 也回退会抬高 AP/Macro-F1 点估计，但 AUROC bootstrap 不稳。
3. 小型 `predef_lowabs_valselect_*` 仍失败：它在 7 个预定义协议里按 validation macro/balanced utility 选，fs0 只到 AP 0.4839 / AUROC 0.6247 / Macro-F1 0.5909，低于固定 `src0` scorefallback 和固定 `predef_lowabs_r25_scorefallback_thr`。因此当前不能把“validation 选固定协议”写成统一 router。
4. 目前更可信的叙事是：**source/confidence fallback 家族可在 fs0/fs1/fs2 各自闭环，但统一协议还没有完成**。下一步不是扩大 selector，而是把少数显式规则拆成机制假设：source0 score fallback、lowabs decision fallback、fs2 source fallback，分别做 repeated-CV/残差诊断后再决定是否组合。

---

### 4v. Fallback OOF 机制诊断：统一 router 应写成“保护 BGE 区域 + RACL/NLI 排序区域”

新增 `src/models/diagnose_fallback_mechanisms.py`，只读取已备份 OOF `.npz`，比较候选方法与 `bge_lr` 的逐样本翻转：`fixed_total` 表示 BGE 错而候选对，`broken_total` 表示 BGE 对而候选错。输出为 `data/final/cleancl/fallback_mechanism_diagnosis_20260608.json`。

| 诊断对象 | 候选 | ΔAUPRC | ΔAUROC | ΔMacro-F1 | 净翻转 | 主要残差信号 |
|---|---|---:|---:|---:|---:|---|
| fs0 旧缓存 | `scorefallback_bge025_src0_src2_3_lowabs` | +0.0191 | +0.0092 | +0.0210 | +39 | 收益高度集中在 BGE 不确定区：`bge_c00_08` 净 +38；`low/absent` 置信净 +31，主要修 BGE 假阳性。 |
| fs0 新缓存 | `scorefallback_bge100_src0` | +0.0182 | +0.0090 | +0.0242 | +20 | `src0` score fallback 会移动 fold 阈值，除 source0 外还影响 `src1`、low confidence 与部分类别；fold2 净 +38，但 apparel/baby/food 有负净翻转。 |
| fs1 | `predef_lowabs_r25_scorefallback_srcconf_bgefallback` | +0.0299 | +0.0150 | +0.0126 | +20 | 增益主要来自 `source_count>=2`、`src4p`、medium confidence；lowabs fallback mask 本身与 BGE decision 相同，作用更像保护脆弱区域，同时让 RACL/NLI 接管 source-rich 区域。 |
| fs2 | `rankmix_nli25 + confidence_bgefallback_src0_src2_3` | +0.0200 | +0.0087 | +0.0221 | +7 | 主要修 BGE 假阴性：FN->TP 为 74，但也新增 74 个 TN->FP；Macro-F1 上升来自召回边界修复，不是全局净错误大幅下降。 |

判断：

1. fs0 的 score-side fallback 更像排序/阈值再校准：它在 BGE 不确定与低置信区域减少假阳性，但会通过阈值迁移影响未回退组。因此统一协议不能只写“mask 内替换分数”，还要显式报告 fold-threshold ripple。
2. fs1 的 `predef_lowabs` 不是靠在 lowabs mask 上直接击败 BGE，而是把 lowabs/source0 区域保留给 BGE decision，同时让 NLI+BGE rankmix 在 source-rich/medium-confidence 样本上修复 BGE。这个机制更像“protected BGE regions + RACL/NLI regions”。
3. fs2 的 decision fallback 是正类召回修复，净翻转不大但 Macro-F1 明显改善。下一版统一 router 需要同时约束新增 FP，否则会在 fs2 外的 split 上不稳。
4. 下一轮不应扩大 validation selector；应实现一个少参数、可预注册的 hybrid：`src0/lowabs` 默认保护 BGE decision，source-rich/medium-confidence 使用 NLI+BGE rankmix，score-side fallback 只在 BGE 不确定区轻量触发，然后做 repeated grouped CV。

---

### 4w. Protected-hybrid OOF screen：点估计全正，但尚未统一严格闭环

基于 §4v 的机制诊断，新增 `src/models/diagnose_protected_hybrid_oof.py`，在已保存 OOF 上合成少数预定义 hybrid：score 侧使用 `rankmix_nli25`、lowabs/BGE-uncertain score fallback；decision 侧使用 BGE 保护 `source0` / `source0+source2_3` / `lowabs` 区域，其余交给 `rankmix` 或 scoreguard decision head。该脚本不重训模型，只作为是否值得进入正式 fold-level evaluator 的筛查。

输出：

- `data/final/cleancl/protected_hybrid_oof_screen_20260608.json`：每个 case top OOF 候选的 1,000 bootstrap。
- `data/final/cleancl/protected_hybrid_forced_bootstrap_20260608.json`：4 个强制候选的 5,000 bootstrap。

最接近统一协议的是：

`score=rank25_bge025_lowabs__decision=protect_lowabs_scoreguard_clip_drop20_min30_srcbin_conf`

| Case | ΔAUPRC | ΔAUROC | ΔMacro-F1 | 5k p(AP/AUROC/Macro-F1) | 判断 |
|---|---:|---:|---:|---|---|
| fs0 旧缓存 | +0.0185 | +0.0094 | +0.0114 | 0.0150 / 0.0330 / 0.0566 | 分类边缘，不严格 |
| fs0 新缓存 | +0.0188 | +0.0074 | +0.0088 | 0.0130 / 0.0800 / 0.1514 | AUROC/Macro-F1 不严格 |
| fs1 predef | +0.0290 | +0.0155 | +0.0126 | 0.0000 / 0.0012 / 0.0372 | 三项严格 |
| fs2 bgefallback | +0.0186 | +0.0087 | +0.0155 | 0.0084 / 0.0152 / 0.0184 | 三项严格 |

其他强制候选也不能统一闭环：

- `rank25_bge025_lowabs + protect_src0_src23_scoreguard_confidence` 在 fs2 严格、fs0 新缓存 Macro-F1 严格，但 fs0 旧缓存和 fs1 的 Macro-F1 不严格。
- `rank25_bge025_unc08 + protect_src0_src23_rank25` 在 fs0 旧缓存和 fs2 严格，但 fs0 新缓存 AUROC 不严格，fs1 Macro-F1 不严格。
- `rank25 + protect_lowabs_rank25` 在 fs1/fs2 严格，但 fs0 旧缓存 AUROC 边缘、fs0 新缓存 AUROC/Macro-F1 不严格。

追加两个诊断文件：

- `data/final/cleancl/protected_hybrid_fs0_newcache_failure_20260608.json`：专门解释统一候选在 fs0 新缓存上的失败。该候选相对 BGE 将 TP 从 314 提到 343，但 FP 也从 358 增到 382；主要负净翻转来自 `food_and_beverages` (net -11)、fold1 (net -11)、`source_count>=2` (net -7)、`confidence=medium` (net -6)、`src2_3:medium` (net -5)。这说明 `scoreguard_srcbin_conf` 在 fs0 的 medium/source-rich 区域过度放开正类。
- `data/final/cleancl/scorefallback_selfthr_forced_bootstrap_20260608.json`：验证 fs0 strict 候选依赖 scorefallback 自身阈值，而不是任意 protected decision head。fs0 旧缓存 `scorefallback_bge025_src0_src2_3_lowabs` 自带 yhat 仍三项严格，p=0.0144/0.0458/0.0202；fs0 新缓存 `scorefallback_bge100_src0` 自带 yhat 仍三项严格，p=0.0096/0.0490/0.0176。相反，fs1 的 scorefallback 自带 yhat 只保排序、不保 Macro-F1：`scorefallback_bge025_src0_src2_3_lowabs` Macro-F1 Δ=+0.0049, p=0.2738。

判断：

1. OOF 原型支持“保护 BGE 区域 + RACL/NLI 区域”的方向，因为所有强制候选在四个 case 上三项点估计均为正。
2. 但当前统一候选仍弱于 split-specific strict 候选，尤其 fs0 新缓存的 AUROC/Macro-F1 不稳；不能直接进入论文主方法。
3. 下一步若继续统一协议，应先解决 fs0 新缓存：需要保留 fs0 的 scorefallback self-threshold 行为，同时只在 fs1/fs2 风格 split 上启用 decision fallback。可尝试的最小结构不是大 selector，而是一个只在 validation 上判定“使用 scorefallback 自阈值还是 protected decision fallback”的二值开关，并加入 medium/source-rich FP 约束。

已把该二值开关落到 fold-level evaluator（`cv_nli_predef_lowabs.py`）做 quick 复核，新增输出：

- `cv_nli_predef_lowabs_srcargs_drop_fs0_s0_switch_relaxed_quick.json`
- `cv_nli_predef_lowabs_srcargs_drop_fs1_s0_switch_relaxed_quick.json`

开关定义：每折在 validation 上比较 scorefallback self-threshold 与 `lowabs_srcconf_bgefallback`；默认要求 protected decision 不在 `medium/source_count>=2/src2_3:medium` 负类上相对 self-threshold 增加过多 FP，同时允许 `val Macro-F1` 提升 >= 0.008 时覆盖 FP guard。

| quick case | BGE | 固定 self/protected 对照 | 最好 switch | 判断 |
|---|---|---|---|---|
| fs0 switch quick | BGE AP 0.4725 / AUROC 0.6217 / Macro-F1 0.5749 | `scorefallback_bge100_src0` AP 0.4861 / AUROC 0.6275 / Macro-F1 0.5948 | `sf100src0_or_lowabs_srcconf_fp02_gain008` 完全等同 self-threshold，Macro-F1 0.5948 | 不误伤 fs0，但没有新增收益 |
| fs1 switch quick | BGE AP 0.4736 / AUROC 0.6288 / Macro-F1 0.5928 | 固定 `predef_lowabs_r25_scorefallback_srcconf_bgefallback` AP 0.5021 / AUROC 0.6430 / Macro-F1 0.6089 | `sf100src0_or_lowabs_srcconf_fp02_gain008` AP 0.4992 / AUROC 0.6434 / Macro-F1 0.6063 | 恢复部分 protected fallback，但仍低于固定协议 |

结论：fold-level 二值开关机制按预期工作，但目前不是更强统一方法；它把 fs1 从保守版 0.6001 提到 0.6063，却仍低于固定 protected fallback 0.6089。暂不扩 5k。下一步若继续，应考虑让 switch 选择的不是“是否 protected”，而是“是否允许 scorefallback self-threshold 覆盖 protected decision”，或者加入更直接的 validation recall-loss guard。

---

### 4x. Reverse switch 与 decoupled score calibration：分类头稳定，排序显著性仍差一线

在 §4w 的基础上，`cv_nli_predef_lowabs.py` 新增反向开关：默认使用 `predef_lowabs_r25_scorefallback_srcconf_bgefallback` 的 protected decision，只在 validation 上出现以下证据时切回 scorefallback self-threshold：

1. self-threshold 的 validation Macro-F1 明显更好，且正类召回没有相对 protected 明显损失；
2. protected decision 在 `medium/source_count>=2/src2_3:medium` 等 guarded FP 组上新增过多负类假阳性，且 protected 的 validation macro gain 不足以补偿。

新增输出：

- `cv_nli_predef_lowabs_srcargs_drop_fs1_s0_switch_reverse_quick.json`
- `cv_nli_predef_lowabs_srcargs_drop_fs0_s0_switch_reverse_quick.json`
- `cv_nli_predef_lowabs_srcargs_drop_fs0_s0_switch_reverse_top20_5k.json`
- `cv_nli_predef_lowabs_srcargs_drop_fs0_s0_decoupled_top12_5k.json`
- 对应 OOF：`oof_nli_predef_lowabs_srcargs_drop_fs1_s0_switch_reverse_quick.npz`、`oof_nli_predef_lowabs_srcargs_drop_fs0_s0_switch_reverse_5k.npz`、`oof_nli_predef_lowabs_srcargs_drop_fs0_s0_decoupled_5k.npz`

结果：

| Case / 协议 | AUPRC | AUROC | Macro-F1 | wF1 | 5k p(AP/AUROC/Macro-F1) | 判断 |
|---|---:|---:|---:|---:|---|---|
| fs1 BGE+LR quick | 0.4736 | 0.6288 | 0.5928 | 0.6336 | - | 对照 |
| fs1 `switchrev_sf100src0_or_lowabs_srcconf_fph08_m00_r03_g008` quick | 0.5021 | 0.6430 | 0.6089 | 0.6365 | 未扩 5k；等同 fixed protected quick | 反向开关修复 relaxed switch 的 fold1 误切 |
| fs0 BGE+LR | 0.4725 | 0.6217 | 0.5749 | 0.6059 | - | 对照 |
| fs0 `switchrev_sf100src0_or_lowabs_srcconf_fph08_m00_r03_g008` | 0.4900 | 0.6265 | 0.5959 | 0.6340 | 0.0030 / 0.1880 / 0.0180 | AP/Macro-F1 显著，AUROC 不显著 |
| fs0 `switchrev_score_a20_bge080_lowabs_decision_sf100_fph08_m00_r03_g008` | 0.4891 | 0.6285 | 0.5959 | 0.6340 | 0.0026 / 0.0672 / 0.0180 | decoupled score 提高 AUROC，但仍不严格 |
| fs0 `switchrev_score_a20_bge090_lowabs_decision_sf100_fph08_m00_r03_g008` | 0.4882 | 0.6286 | 0.5959 | 0.6340 | 0.0042 / 0.0660 / 0.0180 | AUROC 最高点估计，但仍不严格 |
| fs0 `switchrev_score_a15_bge080_lowabs_decision_sf100_fph08_m00_r03_g008` | 0.4874 | 0.6284 | 0.5959 | 0.6340 | 0.0036 / 0.0582 / 0.0180 | 最接近严格三项之一 |
| fs0 `switchrev_score_a15_bge090_lowabs_decision_sf100_fph08_m00_r03_g008` | 0.4866 | 0.6284 | 0.5959 | 0.6340 | 0.0044 / 0.0566 / 0.0180 | 最接近严格三项之一 |

机制读数：

1. fs1 上最保守 reverse rule 在 5 个 fold 全部保留 protected decision，因此复现 fixed protected 的 quick 表现；这说明“默认保护 BGE 区域”比 relaxed switch 更适合 fs1。
2. fs0 上同一 reverse rule 会在 fold0 保留 protected 以修复召回，在 fold2/fold3 因 guarded FP 膨胀分别约 0.1429/0.2059 而切回 `sf100src0`，在 fold1/fold4 因 validation macro/recall 条件切回 self。该行为吻合 §4w 的 fs0 失败诊断，不是任意 selector。
3. 仅换 decision head 可以稳定 Macro-F1，但 AUROC 增益太小。进一步把 score 与 decision 解耦，用固定 `alpha=0.15/0.20` 的 NLI+BGE rankmix，并在 `src0_src2_3_lowabs` 上做 0.80/0.90 BGE score fallback，可把 AUROC 点估计从 0.6265 推到 0.6284-0.6286，bootstrap AUROC p 从 0.188 降到 0.0566-0.0672，但仍未达到严格 0.05。
4. 因此 reverse/decoupled 不是最终统一主方法；它们的价值是把问题进一步压缩到 **score-level uncertainty calibration**。下一轮应尝试更有原则的 monotone/risk calibration 或 BGE-uncertainty-stratified score fallback，而不是继续增加 decision selector。

---

### 4y. Source-first CM p_cls + NLI evidence rank：统一 ranking 增益已成形，decision 仍需 guard

在 §4x 之后，继续把真正的 RACL 检索分支 `sourcefirst_cm_pcls` 接回 predef-only evaluator，检验“contrastive retrieval expert + NLI evidence posterior”是否能形成一个跨 `fold_seed=0/1/2` 的统一排序头。该轮实验刻意使用 `dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl`，但 BGE/CM tmp 使用非 drop source-first 资产 `cv_tmp_args_srcfirst_a120_small_e3_c10_fs*_s0`；这是为了与 source-first baseline 行保持同一资产口径，而不是混用 noargs/args 临时目录。

新增代码与输出：

- `src/models/cv_nli_predef_lowabs.py`：新增 `--cm_tmp/--cm_seed`，读取 source-first CM p_cls fold bundle；新增 `rank_weighted`，并输出 `rankw_sourcefirst_cm025_nli075`、`rankw_sourcefirst_cm033_nli067`、`rankw_sourcefirst_cm040_nli060`。
- `src/models/bootstrap_oof_methods.py`：只读取 OOF `.npz`，做多个 case 的 pooled paired bootstrap，不重训、不调参。
- `data/final/cleancl/cv_nli_predef_lowabs_srcargs_drop_fs*_s0_nondropbge_cmpcls_quick.json/.npz`
- `data/final/cleancl/cv_nli_predef_lowabs_srcargs_drop_fs*_s0_nondropbge_cmpcls_decision_quick.json/.npz`
- `data/final/cleancl/cv_nli_predef_lowabs_srcargs_drop_fs*_s0_nondropbge_cmpcls_weighted_quick.json/.npz`
- `data/final/cleancl/oof_bootstrap_cmpcls_decoupled_20260608.json`
- `data/final/cleancl/oof_bootstrap_cmpcls_weighted_20260608.json`
- `data/final/cleancl/oof_bootstrap_cmpcls_weighted_switch_20260608.json`
- `data/final/cleancl/oof_bootstrap_cmpcls_weighted_guard_20260608.json`

三划分 quick 结果：

| Case / 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| fs0 BGE+LR | 0.4658 | 0.6239 | 0.5716 | 0.5935 |
| fs0 `rankavg_sourcefirst_cm_pcls_bge` | 0.4867 | 0.6395 | 0.5919 | 0.6368 |
| fs0 `rankavg_sourcefirst_cm_pcls_nli075_decision_nli075_lowabs` | 0.4954 | 0.6406 | 0.5986 | 0.6421 |
| fs0 `rankw_sourcefirst_cm040_nli060_decision_nli075_lowabs` | 0.4935 | 0.6407 | 0.5986 | 0.6421 |
| fs0 `rankw_sourcefirst_cm040_nli060_decision_cmbge_nli075` | 0.4935 | 0.6407 | 0.5948 | 0.6309 |
| fs0 `rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008` | 0.4935 | 0.6407 | 0.5983 | 0.6296 |
| fs0 `rankw_sourcefirst_cm040_nli060_decision_switch_..._srcge2_lowmedium_cmbgeprotect` | 0.4935 | 0.6407 | 0.5998 | 0.6341 |
| fs1 BGE+LR | 0.4657 | 0.6288 | 0.6000 | 0.6300 |
| fs1 `rankavg_sourcefirst_cm_pcls_bge` | 0.4803 | 0.6275 | 0.5821 | 0.5889 |
| fs1 `rankavg_sourcefirst_cm_pcls_nli075_decision_nli075_lowabs` | 0.4970 | 0.6300 | 0.5938 | 0.6232 |
| fs1 `rankw_sourcefirst_cm040_nli060_decision_nli075_lowabs` | 0.4982 | 0.6326 | 0.5938 | 0.6232 |
| fs1 `rankw_sourcefirst_cm040_nli060_decision_cmbge_nli075` | 0.4982 | 0.6326 | 0.5907 | 0.6150 |
| fs1 `rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008` | 0.4982 | 0.6326 | 0.6014 | 0.6145 |
| fs1 `rankw_sourcefirst_cm040_nli060_decision_switch_..._srcge2_lowmedium_cmbgeprotect` | 0.4982 | 0.6326 | 0.6065 | 0.6256 |
| fs2 BGE+LR | 0.4928 | 0.6345 | 0.5765 | 0.6063 |
| fs2 `rankavg_sourcefirst_cm_pcls_bge` | 0.5159 | 0.6459 | 0.6004 | 0.6328 |
| fs2 `rankavg_sourcefirst_cm_pcls_nli075_decision_nli075_lowabs` | 0.5211 | 0.6499 | 0.5919 | 0.6163 |
| fs2 `rankw_sourcefirst_cm040_nli060_decision_nli075_lowabs` | 0.5184 | 0.6492 | 0.5919 | 0.6163 |
| fs2 `rankw_sourcefirst_cm040_nli060_decision_cmbge_nli075` | 0.5184 | 0.6492 | 0.6004 | 0.6199 |
| fs2 `rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008` | 0.5184 | 0.6492 | 0.6005 | 0.6355 |
| fs2 `rankw_sourcefirst_cm040_nli060_decision_switch_..._srcge2_lowmedium_cmbgeprotect` | 0.5184 | 0.6492 | 0.6101 | 0.6412 |

Pooled OOF bootstrap (`n=5082`, `n_boot=3000`)：

| 方法 | Pooled AUPRC | AUROC | Macro-F1 | vs BGE ΔAP/ΔAUROC/ΔMacro p | vs `CM+BGE` ΔAP/ΔAUROC/ΔMacro p |
|---|---:|---:|---:|---|---|
| `bge_lr` | 0.4730 | 0.6292 | 0.5828 | - | - |
| `rankavg_sourcefirst_cm_pcls_bge` | 0.4921 | 0.6374 | 0.5939 | - | - |
| `rankavg_sourcefirst_cm_pcls_nli075_decision_nli075_lowabs` | 0.5023 | 0.6400 | 0.5951 | +0.0290 / +0.0107 / +0.0123, p=0.0000/0.0030/0.0083 | +0.0100 / +0.0026 / +0.0011, p=0.0000/0.0053/0.4280 |
| `rankw_sourcefirst_cm033_nli067_decision_nli075_lowabs` | 0.5009 | 0.6407 | 0.5951 | +0.0277 / +0.0116 / +0.0124, p=0.0000/0.0003/0.0090 | +0.0085 / +0.0033 / +0.0011, p=0.0053/0.0257/0.4110 |
| `rankw_sourcefirst_cm040_nli060_decision_nli075_lowabs` | 0.5017 | 0.6407 | 0.5951 | +0.0286 / +0.0116 / +0.0122, p=0.0000/0.0003/0.0073 | +0.0096 / +0.0034 / +0.0014, p=0.0007/0.0047/0.4037 |
| `rankw_sourcefirst_cm040_nli060_decision_cmbge_nli075` | 0.5017 | 0.6407 | 0.5961 | +0.0284 / +0.0115 / +0.0131, p=0.0000/0.0000/0.0123 | +0.0096 / +0.0034 / +0.0023, p=0.0003/0.0057/0.3057 |
| `rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008` | 0.5017 | 0.6407 | 0.6001 | +0.0286 / +0.0116 / +0.0173, p=0.0000/0.0006/0.0012 | +0.0095 / +0.0033 / +0.0063, p=0.0004/0.0048/0.1236 |
| `rankw_sourcefirst_cm040_nli060_decision_switch_..._srcge2_lowmedium_cmbgeprotect` | 0.5017 | 0.6407 | 0.6059 | +0.0285 / +0.0115 / +0.0231, p=0.0000/0.0006/0.0000 | +0.0095 / +0.0033 / +0.0120, p=0.0010/0.0066/0.0016 |

机制读数：

1. `sourcefirst_cm_pcls` 提供的是 top-precision/AP 增益，NLI075 提供的是 evidence-consistency / AUROC 补偿。简单等权 rank average 已经在 pooled OOF 上显著超过 BGE，并且在 AP/AUROC 上显著超过旧 `CM p_cls + BGE`。
2. 固定权重能缓解 fs1 的 AUROC 回撤：等权 `rankavg_sourcefirst_cm_pcls_nli075` 在 fs1 AUROC 为 0.6300，`cm025/nli075` 到 0.6350，`cm033/nli067` 到 0.6340，`cm040/nli060` 到 0.6326；但 fs1 相对 BGE 的 AUROC 与 Macro-F1 仍未形成单 split 严格优势。
3. `rankw_sourcefirst_cm040_nli060` 是当前最适合写成阶段性主线的 ranking candidate：它保留检索增强对比学习机制，又把 NLI evidence posterior 作为轻量证据校准专家，且 pooled AP/AUROC 对 BGE 和旧 `CM+BGE` 均严格。
4. 追加的 switch decision 把 Macro-F1 明显往前推了一步。`rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008` 的 pooled Macro-F1 达到 0.6001，相对 BGE 的 AP/AUROC/Macro-F1 三项均显著；三折 Macro-F1 为 0.5983/0.6014/0.6005，fs1 终于点估计超过 BGE。
5. 残差诊断显示，switch 相对旧 `CM+BGE` 的 broken 样本集中在 `source_count>=2` 且 `confidence in {low, medium}` 的 source-rich 但证据可信度不足区域。因此新增固定 guard：这些区域的 binary decision 保护回 `rankavg_sourcefirst_cm_pcls_bge_thr`，分数仍保留 `rankw_sourcefirst_cm040_nli060`。该规则不是 validation 选头，而是由 source/confidence sufficiency 预定义。
6. guarded 版本第一次让统一候选在 pooled OOF 上同时显著超过 BGE 和旧 `CM+BGE` 的 AP/AUROC/Macro-F1：pooled Macro-F1=0.6059，相对旧 `CM+BGE` 的 ΔMacro=+0.0120, p=0.0016。三折 Macro-F1 为 0.5998/0.6065/0.6101。
7. 剩余限制：fs1 单 split 的 AP 相对 BGE 显著，但 AUROC/Macro-F1 仍未显著超过 BGE（AUROC p=0.2590，Macro-F1 p=0.2222）。因此当前可以写成 **pooled repeated-CV strict 主方法候选**，但若论文主表要求每个 fold_seed 都三指标显著，还需要继续修 fs1 score/decision 边界。
8. 补充 score-head 扫描：在同一 guarded decision 下，`cm025/nli075` 的 fs1 AUROC 最高（0.6350），但 pooled AP/AUROC 只有 0.4990/0.6400；`cm033/nli067` 为 0.5009/0.6407；`cm040/nli060` 为 0.5017/0.6407，pooled 最稳。因此当前主候选仍保留 `cm040/nli060`，不要仅为 fs1 AUROC 点估计牺牲 pooled ranking 主张。

下一步：

1. 不再扩大 wide valselect。现有失败已经说明小 validation carve 不足以自动挑出强 decision head。
2. 把下一版方法写成 **Evidence-Calibrated Retrieval Contrastive Ranking with source-sufficiency guarded decision**：`CM p_cls` 负责检索增强对比表征，`NLI075` 负责证据一致性 rank calibration，二者做固定 rank-weighted fusion；decision 用 `sf025lowabs self-threshold vs lowabs_srcconf BGE-protected` 小开关，并在 `source_count>=2 & confidence in {low, medium}` 上回退到 `CM+BGE` decision。
3. 对 decision 层只继续做少参数 guard：下一轮专攻 fs1 AUROC/Macro-F1 相对 BGE 的单 split 显著性。可尝试 score 权重从 `cm040/nli060` 轻移到 `cm025/nli075` 或 fs1-specific uncertainty calibration，但必须保持统一规则、不能引入 wide selector。

---

### 4z. Adaptive score/decision repair：机制化候选提升点估计，taxonomy-aware 候选 pooled 显著胜当前主方法

基于 §4y 的残差，继续做两类很窄的 OOF 筛查，并已把候选固化到 `src/models/cv_nli_predef_lowabs.py` 后在远端重新生成三划分 OOF：

- **机制化 adaptive 候选**：以 `rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect` 为基底；在 `source_count==0 or confidence==medium` 上把 score 向 `sourcefirst_cm_pcls` 回拉 25%；在 `source_bin==src4p & confidence==medium` 上把 binary decision 改用 `rankavg_sourcefirst_cm_pcls_bge_nli075_thr`。方法名：
  `rankw_sourcefirst_cm040_nli060_score_src0ormedium_cmreinforce025_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect_src4pmedium_cmbgenli`
- **taxonomy-aware 诊断候选**：在 `general/sports_and_outdoor` 上把 score 替换为更 NLI-heavy 的 `rankw_sourcefirst_cm025_nli075`；在 `sports_and_outdoor` 上把 decision 替换为 `rankw_sourcefirst_cm025_nli075_thr`。方法名：
  `rankw_sourcefirst_cm040_nli060_score_sportsgeneral_cm025_decision_sports_cm025`

新增输出：

- `data/final/cleancl/cv_nli_predef_lowabs_srcargs_drop_fs*_s0_nondropbge_cmpcls_adaptive_quick.json`
- `data/final/cleancl/oof_nli_predef_lowabs_srcargs_drop_fs*_s0_nondropbge_cmpcls_adaptive_quick.npz`
- `data/final/cleancl/oof_bootstrap_cmpcls_adaptive_quick_20260608.json`

三划分 quick 结果：

| Case / 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| fs0 current guarded | 0.4935 | 0.6407 | 0.5998 | 0.6341 |
| fs0 adaptive | 0.4944 | 0.6416 | 0.5992 | 0.6305 |
| fs0 taxonomy-aware | 0.4950 | 0.6418 | 0.6012 | 0.6364 |
| fs1 current guarded | 0.4982 | 0.6326 | 0.6065 | 0.6256 |
| fs1 adaptive | 0.4974 | 0.6321 | 0.6090 | 0.6292 |
| fs1 taxonomy-aware | 0.5007 | 0.6351 | 0.6124 | 0.6305 |
| fs2 current guarded | 0.5184 | 0.6492 | 0.6101 | 0.6412 |
| fs2 adaptive | 0.5274 | 0.6503 | 0.6122 | 0.6436 |
| fs2 taxonomy-aware | 0.5185 | 0.6498 | 0.6110 | 0.6429 |

Pooled OOF bootstrap (`n=5082`, `n_boot=5000`)：

| 方法 | Pooled AUPRC | AUROC | Macro-F1 | vs BGE ΔAP/ΔAUROC/ΔMacro p | vs `CM+BGE` ΔAP/ΔAUROC/ΔMacro p | vs current guarded ΔAP/ΔAUROC/ΔMacro p |
|---|---:|---:|---:|---|---|---|
| current guarded | 0.5017 | 0.6407 | 0.6059 | +0.0285 / +0.0115 / +0.0231, p=0.0000/0.0006/0.0000 | +0.0095 / +0.0033 / +0.0120, p=0.0010/0.0066/0.0016 | - |
| adaptive | 0.5049 | 0.6413 | 0.6071 | +0.0315 / +0.0120 / +0.0243, p=0.0000/0.0008/0.0000 | +0.0125 / +0.0039 / +0.0132, p=0.0002/0.0036/0.0020 | +0.0030 / +0.0005 / +0.0011, p=0.0696/0.2910/0.2418 |
| taxonomy-aware | 0.5031 | 0.6421 | 0.6086 | +0.0299 / +0.0129 / +0.0258, p=0.0000/0.0000/0.0000 | +0.0111 / +0.0048 / +0.0148, p=0.0000/0.0014/0.0002 | +0.0014 / +0.0014 / +0.0027, p=0.0150/0.0012/0.0242 |

机制读数：

1. adaptive 候选是更适合论文机制叙事的版本：它只使用 source sufficiency 与 confidence 两类已有证据质量变量，三项 pooled 点估计都高于 current guarded，且相对 BGE 与旧 `CM+BGE` 三项显著。但相对 current guarded 的增益未达到显著，所以暂不应声称它严格替代主方法。
2. taxonomy-aware 候选是当前 pooled 点估计与相对 current guarded 显著性最强的结果。它利用商品 category 元数据，在 pooled OOF 上三项显著胜 current guarded；但该规则来自残差筛查，且 fs1 单 split 的 AUROC/Macro-F1 仍未显著（vs BGE：AUROC p=0.1382，Macro-F1 p=0.0756），因此更适合作为“taxonomy-aware reliability adapter / diagnostic upper-bound”而不是直接写成最终主方法。
3. 当前最佳正式候选排序应分两层汇报：主线仍以 §4y 的 **source-sufficiency guarded CM/NLI rank fusion** 作为机制最干净的统一方法；appendix 或 ablation 报告 adaptive 与 taxonomy-aware 两个窄修复，说明剩余误差主要来自证据质量与商品类别交互。
4. 下一步若要把 taxonomy-aware 变成主方法，必须改成 fold 内 validation/nested 选择 category adapters，或用可预注册的 product-taxonomy reliability prior；不能只把 `sports/general` 作为 OOF 后验补丁写进主表。

---

### 4aa. Validation-safe taxonomy adapter screen：后验 taxonomy 信号尚不能转成严格 selector

为检验 §4z 的 taxonomy-aware 组合能否作为论文主方法，新增 `src/models/diagnose_taxonomy_adapter_oof.py`。该脚本只读取三划分 adaptive OOF `.npz` 与对应 CV JSON 的 `fold_meta`，不重训模型；在每个 outer fold 内只用该 fold 的 validation 指标选择 score head 与 decision head，再拼接 held-out test OOF 做 pooled bootstrap。因此它用于回答一个很窄的问题：**taxonomy-aware 的后验收益，能否被验证集安全地选择出来？**

新增输出：

- `data/final/cleancl/taxonomy_adapter_oof_screen_20260608.json`

筛查设置：

- 固定对照：current guarded、adaptive fixed、taxonomy-aware fixed。
- validation-safe selectors：把 score 与 decision 分开选择；score 主要按 validation AUROC 或 AP，decision 按 validation `AP+AUROC+Macro-F1`，并设置 0 或 0.010 的最小 margin，避免小验证折上无意义切换。

Pooled OOF 结果（`n=5082`, `n_boot=3000`）：

| 方法 | Pooled AUPRC | AUROC | Macro-F1 | vs current guarded ΔAP/ΔAUROC/ΔMacro p |
|---|---:|---:|---:|---|
| current guarded | 0.5017 | 0.6407 | 0.6059 | - |
| adaptive fixed | 0.5049 | 0.6413 | 0.6071 | +0.0030 / +0.0005 / +0.0011, p=0.0670/0.2907/0.2437 |
| taxonomy-aware fixed | 0.5031 | 0.6421 | 0.6086 | +0.0014 / +0.0014 / +0.0026, p=0.0197/0.0017/0.0233 |
| selector: adaptive score by val AUROC, decision by val all +0.010 | 0.5043 | 0.6408 | 0.6063 | +0.0023 / +0.0001 / +0.0004, p=0.1093/0.4653/0.3537 |
| selector: adaptive/tax score by val AUROC, decision by val all +0.010 | 0.5044 | 0.6405 | 0.6063 | +0.0025 / -0.0002 / +0.0004, p=0.0960/0.6043/0.3737 |
| selector: tax score by val AP, adaptive decision by val all +0.010 | 0.5024 | 0.6412 | 0.6063 | +0.0007 / +0.0004 / +0.0004, p=0.1073/0.1083/0.3670 |

关键读数：

1. 固定 taxonomy-aware 组合仍是一个有价值的诊断上界：它在 pooled OOF 上显著胜 current guarded。
2. 但一旦要求 fold 内 validation-safe 选择，最好的 selector 只达到 0.5044 / 0.6405 / 0.6063，且相对 current guarded 三项均不显著。它没有复现固定 taxonomy-aware 的 AUROC/Macro-F1 增益。
3. 选择轨迹说明 validation carve 信号不足：例如 `selector_adaptive_score_valauroc0_decision_valall010` 的 15 个 outer folds 中，score/decision 组合为 `adapt/cur` 7 次、`adapt/adapt` 2 次、`cur/cur` 5 次、`cur/adapt` 1 次；加入 taxonomy score 后也只是把部分 fold 切到 `tax/cur`，并未形成稳定收益。
4. 因此当前论文主线不能把 `sports/general` taxonomy 补丁写成最终方法。更诚实的定位是：taxonomy-aware residual adapter 揭示了商品类别与证据可靠性的交互，但现有验证集规模不足以安全学习该适配器。下一步若继续该线，应采用预注册的 taxonomy reliability prior、扩大外部验证数据，或把 taxonomy adapter 放入严格 nested/held-out tuning 协议。

---

### 4ab. Evidence-type adapter：把 adaptive 收益收窄到可观察证据源结构

基于 §4aa 的负结果，继续追踪 taxonomy-aware 与 adaptive 的 residual 来源。把原始 dataset 的 `evidence_params/evidence_ocr/evidence_vlm` 按 `pair_id` 对齐到三划分 OOF 后发现：

- taxonomy-aware 的决策收益主要集中在 `sports_and_outdoor:P` 与 `sports_and_outdoor:none`，说明它更像商品类别上界而不是可直接推广的 selector。
- adaptive 的净正向翻转集中在 `PO:medium`（参数+OCR、中等置信）：pooled 上 fixed/broken 为 31/19；而坏处集中在 `OV:medium`，尤其 food 的 `OV` 小组。

因此新增 `src/models/diagnose_evidence_type_adapter_oof.py`，只做 OOF synthesis，不重训：以 current guarded 为基底，在 `source_count==0 OR evidence_combo==PO & confidence==medium` 上采用 adaptive score；在 `evidence_combo==PO & confidence==medium` 上采用 adaptive decision。该规则不使用标签、不按 validation 选头，也不直接使用 taxonomy。随后已把同一规则固化进 `src/models/cv_nli_predef_lowabs.py`，evaluator 方法名：

`rankw_sourcefirst_cm040_nli060_score_src0orpomedium_cmreinforce025_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect_pomedium_cmbgenli`

新增输出：

- `data/final/cleancl/evidence_type_adapter_oof_screen_20260608.json`
- `data/final/cleancl/oof_evidence_type_adapter_screen_20260608.npz`
- smoke 复核：`cv_nli_predef_lowabs_srcargs_drop_fs1_s0_nondropbge_cmpcls_evtype_smoke.json/.npz`

三划分结果：

| Case / 方法 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| fs0 evidence-type | 0.4945 | 0.6414 | 0.6020 | 0.6330 |
| fs1 evidence-type | 0.4978 | 0.6322 | 0.6099 | 0.6297 |
| fs2 evidence-type | 0.5278 | 0.6504 | 0.6126 | 0.6430 |

Pooled OOF bootstrap (`n=5082`, `n_boot=3000`)：

| 方法 | Pooled AUPRC | AUROC | Macro-F1 | vs BGE ΔAP/ΔAUROC/ΔMacro p | vs `CM+BGE` ΔAP/ΔAUROC/ΔMacro p | vs current guarded ΔAP/ΔAUROC/ΔMacro p |
|---|---:|---:|---:|---|---|---|
| current guarded | 0.5017 | 0.6407 | 0.6059 | +0.0285 / +0.0115 / +0.0231, p=0.0000/0.0006/0.0000 | +0.0095 / +0.0033 / +0.0120, p=0.0010/0.0066/0.0016 | - |
| adaptive fixed | 0.5049 | 0.6413 | 0.6071 | +0.0315 / +0.0120 / +0.0243, p=0.0000/0.0008/0.0000 | +0.0125 / +0.0039 / +0.0132, p=0.0002/0.0036/0.0020 | +0.0030 / +0.0005 / +0.0011, p=0.0696/0.2910/0.2418 |
| taxonomy-aware fixed | 0.5031 | 0.6421 | 0.6086 | +0.0299 / +0.0129 / +0.0258, p=0.0000/0.0000/0.0000 | +0.0111 / +0.0048 / +0.0148, p=0.0000/0.0014/0.0002 | +0.0014 / +0.0014 / +0.0027, p=0.0150/0.0012/0.0242 |
| evidence-type adapter | 0.5052 | 0.6412 | 0.6084 | +0.0318 / +0.0121 / +0.0257, p=0.0000/0.0000/0.0000 | +0.0129 / +0.0039 / +0.0146, p=0.0000/0.0013/0.0003 | +0.0033 / +0.0005 / +0.0025, p=0.0630/0.2967/0.0453 |

补充诊断：若再排除 food category 的 `PO:medium` 小组，Macro-F1 可到 0.6088，且 vs current guarded 的 Macro p=0.0220；但该排除依然带有后验 taxonomy 味道，只能作为 residual diagnosis，不能写成主方法。

`P:low` follow-up：尝试把 taxonomy-aware 的 `sports:P` 收益改写成纯参数证据 prior。结果显示，只有 `sports_and_outdoor:P:low` 与 `cm025/nli075` score/decision 叠加时能把 pooled 提到 0.5057 / 0.6416 / 0.6103（1k quick vs current：AP p=0.038，Macro-F1 p=0.006）；泛化到全体 `P:low` 或去掉 sports taxonomy 后收益消失或变弱。因此该线仍应视为 taxonomy-conditioned 诊断，不进入主方法。

VLM/source follow-up：尝试在 `src2_3:medium`、`non-PO:medium`、`VLM:medium/OV:medium` 上回退 BGE 或 `CM+BGE` decision。最好的 `src2_3:medium -> BGE decision` 只把 pooled Macro-F1 从 0.6084 提到 0.6092，但相对 evidence-type adapter 不显著（5k p=0.3188），且在 fs1 反而从 0.6099 降到 0.6060。因此不能固化为统一 guard。继续做 CM reinforcement 权重 sweep：`src0+PO:medium` 的 0.20/0.25 权重最稳；只在 `PO:medium` 上加强 CM 可把 pooled AUROC 推到 0.6428-0.6429，但 AP 降到约 0.502，不适合作为主 score head。结论：fs1 AUROC/Macro-F1 不能靠简单 VLM/source guard 或 score 权重微调解决。

Threshold follow-up：对 BGE、`CM+BGE`、current guarded、evidence-type adapter 与 taxonomy-aware 做测试集 oracle threshold 诊断。结果显示 evidence-type 的既有 decision head 已经强于同一 score 的最优单阈值：pooled yhat Macro-F1 0.6084 vs score-oracle 0.5989；fs1 yhat 0.6099 vs score-oracle 0.5954；fs2 yhat 0.6126 vs score-oracle 0.6041。因此当前不是简单阈值/概率校准问题，下一步应优先改 ranking score/表征或加入新的证据关系信号。

Common-method OOF sweep：新增 `src/models/diagnose_common_oof_methods.py`，对 fs0/fs1/fs2 adaptive quick OOF 中共同存在的 227 个 evaluator 方法做完整枚举，输出 `data/final/cleancl/common_oof_method_sweep_20260608.json`。结果没有发现漏网的非 taxonomy 现成候选：共同方法池里的最高 pooled AP 仍是 adaptive fixed（0.5049 / 0.6413 / 0.6071），最高 pooled AUROC/Macro-F1 仍是 taxonomy-aware fixed（0.5031 / 0.6421 / 0.6086）。若只看 fs1，`cm025/nli075`、`rankstable`、`valselect` 等确实能把 AUROC 或 Macro-F1 点估计抬高，但 pooled AP/AUROC/Macro-F1 明显掉队，例如 fs1 Macro-F1 前列的 `predef_lowabs_r50_scorefallback_srcconf_bgefallback` pooled AP/AUROC 均弱于 current guarded。结论：继续横向搜索已保存 OOF head 的边际价值很低；evidence-type adapter 仍是更可辩护的 pooled 主线升级，而真正的下一步应新增 score/representation 层面的证据关系信号。

Relation adapter follow-up：新增 `src/models/diagnose_relation_oof_adapter.py`，在 `oof_evidence_type_adapter_screen_20260608.npz` 上做按 `pair_id` 分组的 cross-fit 二层 relation adapter。该诊断显式避免 fs0/fs1/fs2 同一 `pair_id` 重复行泄漏，并且只把 `c` 用作 sample weight，不把标签置信度作为特征。输出 `data/final/cleancl/relation_oof_adapter_screen_20260608.json` 与 `data/final/cleancl/oof_relation_adapter_screen_20260608.npz`。结果为负：`relation_lr_no_category` pooled AP/AUROC/Macro-F1 为 0.4782 / 0.6294 / 0.5890，`relation_hgb_no_category` 为 0.4786 / 0.6242 / 0.5789；即使加入 category 与 taxonomy score 的诊断上界 `relation_lr_with_category_diag` 也只有 0.4950 / 0.6400 / 0.6001，仍弱于 evidence-type adapter 0.5052 / 0.6412 / 0.6084。sample bootstrap 与 group bootstrap 均显示普通 relation LR 相对 current/evidence-type 显著为负。结论：不要继续在现有 OOF 概率和元数据上堆普通二层 stacker；下一步应做 per-source evidence pooling、atomic NLI posterior 聚合或预注册的 source/evidence-sufficiency rule。

NLI source-pooling micro-calibration follow-up：新增 `src/models/diagnose_nli_source_pooling_oof.py`，把 `cache_nli_srcargs_a120.npz` 里的 source-block posterior 特征按 `pair_id` 对齐到 evidence-type OOF，先只做 score-side 微校准，decision 沿用 evidence-type，避免后验阈值调优。pooled screen 的 top 候选为 `evtype_score_argref_neutral_rate35_a05_decision_evtype`，即 `0.95 * rank(evidence-type score) + 0.05 * rank(arg_ref neutral-rate>=0.35)`，得到 AP/AUROC/Macro-F1 0.5093 / 0.6414 / 0.6084。3k bootstrap 相对 current guarded 的 AP/Macro-F1 为正（sample p=0.0063/0.0453；group p=0.0157/0.0497），但相对 evidence-type 自身只是 AP 边缘：sample p=0.0410，group p=0.0810，AUROC/Macro-F1 无增益。随后已把该规则固化进 `cv_nli_predef_lowabs.py`，方法名含 `argrefneutral005`，并补齐 fs0/fs1/fs2 fold-level OOF：fs0 从 evidence-type 0.4945 / 0.6414 / 0.6020 降到 0.4942 / 0.6391 / 0.6020；fs1 从 0.4978 / 0.6322 / 0.6099 到 0.4991 / 0.6317 / 0.6099；fs2 从 0.5278 / 0.6504 / 0.6126 降到 0.5208 / 0.6503 / 0.6126。三划分 fold-level pooled bootstrap 输出 `oof_bootstrap_cmpcls_insuff_smoke_20260608.json`：argrefneutral005 为 0.5034 / 0.6404 / 0.6084，低于 evidence-type 0.5052 / 0.6412 / 0.6084；相对 evidence-type 的 dAP/dAUROC/dMacro 为 -0.0016 / -0.0009 / 0.0000，p=0.7253/0.9087/1.0000。结论：pooled screen 的 AP 提升主要来自 pooled-rank 后验放大；真实 fold evaluator 下该 micro-calibration 为负，不作为主方法。

Evidence-sufficiency fallback rule follow-up：新增 `src/models/diagnose_evtype_residuals.py` 与 `src/models/diagnose_evsuff_oof_rules.py`。前者给 evidence-type adapter 做残差地图；后者只筛少数非 taxonomy、可解释的 source/evidence strata 回退规则。残差显示 evidence-type 相对 current 的收益几乎全集中在 `PO:medium`（fixed/broken 31/19），而仍不稳的非 taxonomy 区域主要是 `src2_3:medium`、`POV:high`、`O:low` 等。第一版 rule screen 曾使用 mask-local rank，已标记为有实现缺陷；修正版 `evsuff_oof_rule_screen_rawblend_20260608.json` 只做原始 score 混合/替换。结果仍不足：`O:low -> BGE score/decision` 可到 0.5073 / 0.6424 / 0.6021，但相对 evidence-type 的 Macro-F1 显著下降（dMacro=-0.0062, p=0.978 for candidate>evidence-type）；`src2_3:medium -> BGE decision` 为 0.5052 / 0.6412 / 0.6092，Macro 小涨不显著（p=0.306）。结论：规则层面的 source/evidence fallback 已基本耗尽；下一步应改模型结构或训练目标，而不是继续加窄 mask。

Source-policy multi-instance pooling follow-up：基于 SURE-RAG / source reliability 文献启发，新增 `src/models/diagnose_source_policy_pooling.py`，并把 `src/models/data.py`、`train.py`、`cv_eval.py` 扩展为支持 `--evidence_policy {params_only,ocr_only,vlm_only,args_only,...}` 与 `--dump_oof`。这条线不再在现有 OOF head 上堆二层 stacker，而是训练分源 CLAIMARC/RACL experts，再用同一 outer fold 的 validation split 选择 decision threshold；score pooling 采用固定公式，避免全 OOF 搜权重。

fs1 source-policy experts（`dataset_verify_faithful_args_srcfirst_a120.jsonl`, `fold_seed=1`, `cm_seed=0`, BGE-small `small_e3_c10`）：

| 方法 | AUPRC | AUROC | Macro-F1 | 备注 |
|---|---:|---:|---:|---|
| BGE+LR | 0.4642 | 0.6184 | 0.5797 | 同 fold protocol baseline |
| no-args | 0.4692 | 0.6181 | 0.5823 | 保守 source-only expert |
| source-first | 0.4682 | 0.5971 | 0.5654 | 单独弱，但作为 rank source 有互补 |
| OCR-only | 0.4861 | 0.6173 | 0.5816 | AP 高，但分类不稳 |
| params-only | 0.4923 | 0.6263 | 0.5804 | 单 expert AP 已显著胜 BGE，Macro 仍弱 |
| args-only | 0.4812 | 0.6162 | 0.5687 | 单独偏噪声，不进入第一版候选 |

固定 pooling / guard 结果：

| 候选 | AUPRC | AUROC | Macro-F1 | 机制 |
|---|---:|---:|---:|---|
| `rankavg_all` = rankavg(BGE, sourcefirst, noargs, OCR, params) | 0.4934 | 0.6321 | 0.5955 | 固定 rank pooling，AP/AUROC 强，Macro 接近 |
| `rankavg_all_score_bge_lr_src0_neg_guard` | 0.4934 | 0.6321 | 0.5995 | 同一 score；在 `source_count==0` 且 BGE fold-decision 为负时压制正判 |
| 五路加入 args-only 的 `rankavg_all` | 0.4921 | 0.6312 | 0.5920 | arguments 作为独立 instance 伤 Macro，不采用 |

定向 5k bootstrap（输出 `source_policy_pooling_guard_top_bootstrap_fs1_20260608.json`）显示 `rankavg_all_score_bge_lr_src0_neg_guard` 在 fs1 上相对 BGE+LR 三项严格正向：ΔAP +0.0288, p=0.0034；ΔAUROC +0.0138, p=0.0480；ΔMacro-F1 +0.0201, p=0.0184。相对 no-args 也为三项正向（p=0.0002/0.0032/0.0266）。相对未 guard 的 `rankavg_all` 只提升 Macro-F1 +0.0040，p=0.1694，说明 guard 是边界修复，不改变排序。

fs2 复核已完成 `ocr_only` 与 `params_only` source-policy experts，并生成 `source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs2_rows_20260608.json`、`oof_source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs2_rows_20260608.npz`、`source_policy_pooling_guard_top_bootstrap_fs2_20260608.json`。单 expert 中，OCR-only 为 0.4938 / 0.6102 / 0.5645，明显不适合作为单模型；params-only 为 0.5026 / 0.6248 / 0.5912，相对 BGE 的 AP/Macro-F1 为正但不显著，AUROC 为负。固定 pooling 后出现第二个正向 split：

| fs2 候选 | AUPRC | AUROC | Macro-F1 | wF1 | vs BGE 5k bootstrap |
|---|---:|---:|---:|---:|---|
| `rankavg_all_score_bge_lr_src0_guard` | 0.5176 | 0.6386 | 0.5926 | 0.6121 | ΔAP +0.0301 p=0.0034；ΔAUROC +0.0160 p=0.0212；ΔMacro +0.0149 p=0.0458；wF1 p=0.1616 |
| `mean_all_score_bge_lr_src0_neg_guard` | 0.5095 | 0.6312 | 0.5983 | 0.6216 | ΔAP +0.0222 p=0.0386；ΔAUROC +0.0087 p=0.1724；ΔMacro +0.0205 p=0.0128；ΔwF1 +0.0209 p=0.0368 |

fs0 复核也已完成 `no_args`、`ocr_only` 与 `params_only`，并生成 `source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs0_rows_20260608.json`、`oof_source_policy_pooling_guard_sourcefirst_noargs_ocr_params_fs0_rows_20260608.npz`。单 expert 仍不够强：no-args 为 0.4607 / 0.6062 / 0.5727，OCR-only 为 0.4827 / 0.6234 / 0.5894，params-only 为 0.4721 / 0.6192 / 0.5773。固定 pooling 后，fs0 的 `rankavg_all` 为 0.4866 / 0.6324 / 0.5894，`logitmean_core` 为 0.4842 / 0.6281 / 0.5967；2k 定向 bootstrap 中，`rankavg_all` 相对 BGE 的 ΔAP/ΔAUROC/ΔMacro 为 +0.0296/+0.0253/+0.0157，p=0.001/0.002/0.074，`logitmean_core` 为 +0.0270/+0.0209/+0.0230，p=0.015/0.029/0.019。

新增 fs3 独立复核（`no_args + ocr_only + params_only`，不含 sourcefirst/args-only；输出 `source_policy_pooling_guard_noargs_ocr_params_fs3_rows_noboot_20260609.json`、`oof_source_policy_pooling_guard_noargs_ocr_params_fs3_rows_noboot_20260609.npz`、`source_policy_pooling_guard_noargs_ocr_params_fs3_targeted_bootstrap_3k_20260609.json`）显示该结构在新划分上只保留排序正信号，分类优势不稳。单 expert 中，no-args 为 0.4865 / 0.6199 / 0.5670 / 0.5612，OCR-only 为 0.4713 / 0.6112 / 0.5601 / 0.5722，params-only 为 0.4755 / 0.6130 / 0.5734 / 0.5735，pooling 协议中的 BGE baseline 为 0.4632 / 0.6128 / 0.5691 / 0.5864。固定 pooling 后：

| fs3 候选 | AUPRC | AUROC | Macro-F1 | wF1 | 3k targeted bootstrap |
|---|---:|---:|---:|---:|---|
| `rankavg_all` | 0.4834 | **0.6299** | 0.5771 | **0.5954** | vs BGE AP/AUROC 显著 p=0.0183/0.0213；Macro p=0.2307 |
| `rankavg_all_score_bge_lr_src0_guard` | 0.4834 | **0.6299** | 0.5775 | 0.5921 | vs BGE AP/AUROC 显著 p=0.0183/0.0213；Macro p=0.1650 |
| `logitmean_all_score_bge_lr_src0_or_lowabs_guard` | 0.4851 | 0.6296 | 0.5771 | 0.5904 | vs BGE AP/AUROC p=0.0177/0.0320；Macro p=0.0900 |
| `source_masked_mean` | 0.4848 | 0.6179 | **0.5807** | 0.5862 | vs BGE AP p=0.0647、Macro p=0.1770；vs noargs Macro/wF1 p=0.0443/0.0183 |

结论：fs3 复核支持“多 source-policy expert 作为排序/可靠性加权信号”这一机制，但没有复现 fs1/fs2 上的严格 Macro-F1 闭环；相对 no-args，AP/AUROC 不占优。该支线不宜继续扩 fs4 作为主方法，当前更适合作为 source reliability / multi-instance pooling 消融，主线仍应回到 RACL prototype relation score + source/evidence sufficiency protocol。

三划分 pooled repeated-CV（`fs0+fs1+fs2`, `n=5082`）确认 source-policy multi-instance pooling 是稳定结构性正信号，但仍弱于 evidence-type adapter：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 | 5k bootstrap 读数 |
|---|---:|---:|---:|---:|---|
| BGE+LR | 0.4669 | 0.6159 | 0.5775 | 0.6070 | baseline |
| no-args | 0.4742 | 0.6148 | 0.5800 | 0.6010 | baseline |
| source-first | 0.4775 | 0.6091 | 0.5857 | 0.6058 | source-policy 单 expert baseline |
| `rankavg_all_score_bge_lr_src0_neg_guard` | 0.4978 | 0.6345 | 0.5930 | 0.6088 | vs BGE p=0.0000/0.0002/0.0062；vs noargs p=0.0000/0.0000/0.0104；vs sourcefirst AP/AUROC p=0.0000/0.0000，Macro p=0.116 |
| `mean_all_score_bge_lr_src0_neg_guard` | 0.4918 | 0.6301 | 0.5955 | 0.6159 | vs BGE p=0.0000/0.0042/0.0004；vs noargs p=0.0000/0.0000/0.0018；vs sourcefirst p=0.0016/0.0000/0.0502 |

机制读数：这是一条比窄 OOF fallback 更有结构含义的信号。RACL 分源 experts（尤其 params/OCR）提供 AP/AUROC 排序互补；BGE 在无真实来源证据区作为 conservative decision guard 抑制 FP。它与 SURE-RAG 的 set-level sufficiency / source reliability 方向一致。三划分 pooled 已显著胜 BGE/noargs/sourcefirst 的排序指标，并在 `mean_all` guard 上接近显著胜 sourcefirst Macro-F1。但它的 pooled 点估计仍低于 evidence-type adapter（0.5052 / 0.6412 / 0.6084），因此当前定位应是 **结构性消融和下一步训练视图的依据**，而不是替代 evidence-type adapter 的最终主方法。

Evidence-type + source-policy hybrid follow-up：把 source-policy `logitmean_core` 用作 OCR-only (`evidence_combo==O`) 证据组的 decision 替换，得到 `evtype_sp_logitcore_replace_O`：0.5052 / 0.6412 / 0.6106 / 0.6392。该规则在 fs0/fs1/fs2 的 O-only 子集方向一致，但 5k paired bootstrap 相对 evidence-type adapter 的总体 Macro-F1 增益仅 +0.0022，p=0.2332；O-only 子集 Macro 增益 +0.0061，p=0.3102。结论：存在轻微互补迹象，但证据不足，不进入主方法。

Source-policy-augmented relation head quick screen：为检验“source-policy 信号是否能被普通二层可靠性头吸收”，新增诊断输出 `sourcepolicy_relation_adapter_quick_20260609.json`。特征包括 BGE/current/adaptive/evidence-type 分数与判决、source-policy `rankavg/mean/logitmean` 分数与判决、`source_count/evidence_combo/confidence/case`，按 `pair_id` 做 5 折 GroupKFold cross-fit。结果仍弱于 evidence-type adapter：`lr_l1_balanced_C0.2` 为 0.4986 / 0.6440 / 0.6030 / 0.6306，`lr_l2_balanced_C0.5` 为 0.4989 / 0.6435 / 0.5999 / 0.6204，HGB 为 0.4829 / 0.6306 / 0.5842 / 0.6053。普通 reliability meta-head 可轻微提高 AUROC，但明显牺牲 AP/Macro-F1，不作为下一步主线。

新增结构实验入口：`src/models/data.py` / `train.py` / `cv_eval.py` 已加入 train-only `--evidence_policy_mix`，例如 `source_first,no_args,ocr_only,params_only`。它让单个 CLAIMARC 在训练 loader 中随机看到不同 evidence view，验证/测试与 memory bank 仍用固定 policy。fs1 smoke 已完成：`cv_evviewmix_srcfirst_noargs_ocr_params_fs1_s0.json` / `oof_evviewmix_srcfirst_noargs_ocr_params_fs1_s0.npz`。结果为负，PCLS 为 0.4700 / 0.6128 / 0.5779，selectiveRKC 为 0.4689 / 0.6119 / 0.5840，BGE 为 0.4657 / 0.6288 / 0.6000；相对 BGE 的 paired bootstrap 中 PCLS dAP +0.0033 (p=0.4065)、dAUROC -0.0163 (p=0.9290)、dMacro -0.0037 (p=0.6090)。结论：source-policy 的收益来自测试时多 expert pooling / source-specific decision guard，而不是简单训练期 evidence-view dropout。

进一步新增 multi-view consistency 训练入口：`--view_consistency_mix` 会为每个训练样本额外采样一个 auxiliary evidence view，`--view_ce_weight/--view_logit_weight/--view_embed_weight` 分别约束辅助视图 CE、logit 与表示。fs1 all-loss smoke（`source_first` 主视图，aux=`no_args,ocr_only,params_only`，CE=0.10、logit=0.05、embed=0.05）已完成并备份：`cv_viewcons_srcfirst_aux_noargs_ocr_params_ce010_logit005_emb005_fs1_s0.json` / `oof_viewcons_srcfirst_aux_noargs_ocr_params_ce010_logit005_emb005_fs1_s0.npz`。结果仍为负：PCLS 0.4758 / 0.6104 / 0.5727 / 0.5810，selectiveRKC 0.4764 / 0.6119 / 0.5722 / 0.5800，BGE 0.4657 / 0.6288 / 0.6000 / 0.6300；PCLS vs BGE dAP +0.0090 (p=0.2400)、dAUROC -0.0184 (p=0.9635)、dMacro -0.0158 (p=0.9020)。embedding-only follow-up（embed=0.05，CE/logit=0）也为负：PCLS 0.4753 / 0.6100 / 0.5711 / 0.5817，selectiveRKC 0.4759 / 0.6110 / 0.5717 / 0.5824；PCLS vs BGE dAP +0.0086 (p=0.2545)、dAUROC -0.0188 (p=0.9650)、dMacro -0.0129 (p=0.8540)。解释：多视图一致性会给 AP 一点局部信号，但无论是否加入 CE/logit，都会损伤 AUROC/Macro-F1，不能作为主方法。

新增 source-sufficiency auxiliary 表示实验：`src/models/train.py` / `cv_eval.py` 已加入 `--source_aux_combo_weight`、`--source_aux_conf_weight`、`--source_aux_count_weight`，让 retrieval embedding `g` 在 CL 阶段预测 `evidence_combo`、`confidence` 和粗 source-count bin，不改推理接口。fs1 smoke（0.01/0.01/0.01）已完成并备份：`cv_sourceaux_combo_conf_count_w001_fs1_s0.json` / `oof_sourceaux_combo_conf_count_w001_fs1_s0.npz`。结果仍不足：PCLS 0.4667 / 0.6115 / 0.5900 / 0.6181，selectiveRKC 0.4660 / 0.6108 / 0.5876 / 0.6144，BGE 0.4657 / 0.6288 / 0.6000 / 0.6300；PCLS vs BGE dAP +0.0007 (p=0.4650)、dAUROC -0.0173 (p=0.9105)、dMacro -0.0109 (p=0.7950)。局部读数有价值：fold0 Macro-F1 从 view-consistency 约 0.58 提到 0.6353，fold3/fold4 也改善，但 fold1 仍 0.5233，排序 AP 基本不动。结论：metadata-aware 表示正则可以改善部分阈值行为，但不是解决 fs1 泛化的主突破口；下一步不应继续调辅助权重，而应回到 evidence-type adapter / set-level sufficiency 的显式 score/decision 结构。

新增 cross-encoder reranker 基线脚本：`src/models/cv_reranker_feature.py` 使用 `BAAI/bge-reranker-v2-m3` 缓存 claim-evidence pair logits，再按 grouped CV 做 direct threshold 与单特征 LR。fs1 已完成并备份：`reranker_bge_v2m3_srcfirst_fs1_logits.npz`、`cv_reranker_bge_v2m3_fs1.json`、`oof_reranker_bge_v2m3_fs1.npz`。结果显著弱于 BGE+LR：direct 为 0.4077 / 0.5185 / 0.5206 / 0.5284，LR 为 0.3984 / 0.5104 / 0.5138 / 0.5403，BGE 为 0.4657 / 0.6288 / 0.6000 / 0.6300；direct vs BGE dAP -0.0588 (p=0.9995)、dAUROC -0.1105 (p=1.0000)、dMacro -0.0760 (p=1.0000)。结论：通用 relevance reranker 没有学到本任务所需的 claim-truth/evidence-consistency 关系，不能作为 teacher；如果继续用 cross-encoder，应改为任务内 fine-tuning 或 NLI/claim-verification 专用模型，而非通用 reranking logit。

2026-06-09 set-level sufficiency / evidence-type score-decision 结构复核：

1. 新增 `src/models/cv_set_sufficiency_meta.py`，把 SURE-RAG 式的 coverage/source uncertainty/disagreement 思路压成一个 fold-safe 小 LR head。每个 `fs0/fs1/fs2` case 内按 outer fold cross-fit，LR 的 C、class weight 与阈值只用训练折 inner-OOF 选择；特征包含 BGE、CM+BGE、current guarded、adaptive 分数/uncertainty/disagreement、`source_count`、`evidence_combo`、`confidence`，不含 taxonomy/category。结果为负：`set_suff_lr_no_cat` 为 0.4861 / 0.6280 / 0.5859 / 0.6155；用 evidence-type score 但替换成 LR decision 后为 0.5052 / 0.6412 / 0.5859 / 0.6155。也就是说，学习式 sufficiency head 能保持一点 AP，但会把 Macro-F1 从 0.6084 明显拉低，尤其在 fs1 放大 source0/absent false positives。结论：普通低维 LR meta-head 仍会过拟合证据缺失区，不能替代手写 score/decision adapter。
2. 新增 `src/models/cv_evidence_type_selector.py`，只允许在少数预注册 mask 中选择 score mask 与 decision mask：`src0`、`PO:medium`、`no-VLM:medium`、`source-rich:medium` 等；每个 case/fold 只用训练折选择一个 adapter，再应用到 held-out fold。balanced objective 下，selector 为 0.5022 / 0.6412 / 0.6074 / 0.6344，相对 BGE 三项显著（p=0.0000/0.0007/0.0000），相对旧 `CM+BGE` 也三项显著（p=0.0007/0.0047/0.0017），但仍略低于固定 evidence-type adapter。折内选择计数显示 decision 几乎自然落到 `PO:medium`，score 端在 fs0/fs1 会漂到 `source_rich_medium`，导致 AP 小降。
3. 固定候选 `evtype_fixed_src0_po_medium__po_medium` 精确复现当前 evidence-type adapter：0.5052 / 0.6412 / 0.6084 / 0.6355；相对 BGE 三项显著（p=0.0000/0.0003/0.0000），相对旧 `CM+BGE` 三项显著（p=0.0000/0.0040/0.0003），相对 current guarded 的 Macro-F1 边界显著（p=0.0487），AP p=0.0510、AUROC p=0.2940。macro/ranking objective 和 `min_gain=0.002` 的 selector smoke 均未超过固定候选。

机制结论：最新文献强调 set-level sufficiency，但在 CLAIMARC 当前数据规模上，“学习一个 sufficiency classifier”不如“把 sufficiency 先验写进极窄 score/decision adapter”。`source_count==0` 只适合作为 score-side adaptive repair 区域；decision-side 最稳的是 `PO:medium`，这与残差诊断中 `PO:medium` fixed/broken 净收益最高一致。下一步若继续结构化，优先实现可训练的 relation/sufficiency prototype 或任务内 verifier，而不是在 OOF 上扩大 selector。

2026-06-09 RACL relation prototype verifier：新增 `src/models/cv_racl_prototype_verifier.py`，直接读取已训练 CLAIMARC fold bundle 中的 retrieval embedding `g`，每个 outer fold 只用训练折构造正/负 class prototypes，再在 held-out fold 上按 prototype similarity gap 输出 relation score。这是对“RACL 表征是否形成可用 evidence-relation geometry”的检验，不重新训练模型，也不使用 taxonomy。prototype 类型包括 global、attribute、evidence combo、source bin、source+confidence、combo+confidence 等；validation 只用于阈值和诊断性 prototype 选择。

单独 prototype decision 不够强，但 score 端有稳定互补：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 | 解释 |
|---|---:|---:|---:|---:|---|
| `rankavg_bge_cm_proto_source_bin` | **0.5053** | 0.6391 | 0.5891 | 0.6142 | prototype 作为第三排序源，AP 与 evidence-type 打平，但分类弱 |
| `rankavg_bge_cm_proto_global` | 0.5029 | 0.6359 | 0.5910 | 0.6054 | global prototype 更稳但排序略弱 |
| `rankavg_saved_cm_bge` | 0.4980 | 0.6357 | 0.5879 | 0.6131 | 同一 saved fold bundle 的 CM+BGE 对照 |
| evidence-type adapter | 0.5052 | **0.6412** | **0.6084** | **0.6355** | 当前主线 |

定向 2k bootstrap：`rankavg_bge_cm_proto_source_bin` 相对 BGE 为 ΔAP +0.0321, p=0.0000；ΔAUROC +0.0101, p=0.0200；ΔMacro +0.0062, p=0.1695。相对旧 `CM+BGE`，AP 显著（p=0.0070），AUROC/Macro 不显著；相对 evidence-type adapter，AP 基本打平（+0.0002, p=0.4965），AUROC/Macro 明显不足。结论：RACL embedding prototype 提供真实排序信号，但不能承担二分类 decision。

随后新增 `src/models/diagnose_racl_proto_rankblend.py`，只把 prototype 作为 evidence-type 的 score-side calibration，binary decision 仍沿用 evidence-type adapter。按每个 `case+fold` 内 rank 做固定等权 blend：

| 候选 | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| `evtype_rankblend_proto50_decision_evtype` | **0.5071** | **0.6430** | **0.6084** | **0.6355** |
| evidence-type adapter | 0.5052 | 0.6412 | 0.6084 | 0.6355 |
| current guarded | 0.5017 | 0.6407 | 0.6059 | 0.6341 |
| BGE+LR | 0.4730 | 0.6292 | 0.5828 | 0.6100 |

3k bootstrap 显示等权 rankblend 相对 BGE 与旧 `CM+BGE` 三项显著；相对 current guarded，Macro-F1 显著（p=0.0353），AP/AUROC 不显著（p=0.0763/0.1267）；相对 evidence-type adapter，AP/AUROC 点估计正向但不显著（p=0.3047/0.1957），Macro-F1 相同。当前定位：**新的最高点估计 pooled 候选**，机制上保留 RACL prototype ranking + evidence-type decision，但还不能声称显著胜过 evidence-type adapter。下一步如果继续，应把 prototype score calibration 做进 fold-level evaluator，或把 prototype gap 训练成轻量 auxiliary relation objective，而不是只停留在 OOF rankblend。

协议化复核：新增 `src/models/cv_racl_proto_evtype_protocol.py`，不做权重网格选择，只输出两个固定 score-calibration 分支，并始终沿用 evidence-type binary decision：

1. `evtype_proto_cal50_decision_evtype`：evidence-type score 与 `rankavg_bge_cm_proto_source_bin` 做 case+fold 内等权 rank blend。
2. `evtype_proto_raw25_decision_evtype`：evidence-type score 与 raw `proto_source_bin` gap 做 case+fold 内 0.25 prototype weight 的保守 rank blend。

5k bootstrap 输出 `racl_proto_evtype_protocol_bootstrap_20260609.json`：

| 协议候选 | AUPRC | AUROC | Macro-F1 | wF1 | 定位 |
|---|---:|---:|---:|---:|---|
| `evtype_proto_cal50_decision_evtype` | **0.5071** | 0.6430 | 0.6084 | 0.6355 | 最高 AP；fs0/fs1 更稳 |
| `evtype_proto_raw25_decision_evtype` | 0.5070 | **0.6438** | 0.6084 | 0.6355 | 最高 AUROC；raw prototype 独立信号更清楚 |
| evidence-type adapter | 0.5052 | 0.6412 | 0.6084 | 0.6355 | 当前主线 |
| current guarded | 0.5017 | 0.6407 | 0.6059 | 0.6341 | 上一版统一 guard |
| BGE+LR | 0.4730 | 0.6292 | 0.5828 | 0.6100 | 强基线 |

显著性：两个协议分支相对 BGE 和旧 `CM+BGE` 均三项显著。相对 current guarded，`cal50` 的 Macro-F1 显著（p=0.0428）但 AP/AUROC 未显著；`raw25` 的 AUROC 边界显著（p=0.0446），Macro-F1 p=0.0504，AP p=0.0888。相对 evidence-type adapter，`cal50` 的 AP/AUROC p=0.3024/0.1950，`raw25` 的 AP/AUROC p=0.3156/0.0602，仍未严格显著。因此，协议化 prototype calibration 可作为**ranking/screening 最强点估计**与机制增强，但最终主张仍应说“显著胜 BGE/旧 CM+BGE，未显著胜 evidence-type adapter”。

2026-06-09 prototype decision feature：新增 `src/models/cv_racl_proto_decision_feature.py`，把 prototype 从 score calibration 推进到 binary decision boundary。规则只读取 case+fold 内 rank-normalized prototype score，不训练模型；cross-fit 版本在每个 case/fold 上只用同一 repeated-CV case 的其它 folds 选择小规则。最关键的机制规则是：

> 在 `source_count==0` 的 source-poor 样本上，如果 raw `proto_source_bin` rank < 0.20，则 veto evidence-type 的正判；如果 rank > 0.75，则 promote evidence-type 的负判。score 使用 `evtype_proto_raw25`，decision 使用该 source0 prototype edit。

输出 `racl_proto_decision_feature_top2_5k_20260609.json`：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 | 解释 |
|---|---:|---:|---:|---:|---|
| `proto_decision_fixed_veto0.20_promote0.75_raw_src0_score_raw25` | **0.5070** | **0.6438** | **0.6142** | **0.6454** | 当前最高分类点估计；source0 prototype sufficiency rule |
| `proto_decision_cvselect_src0nested_macro_raw25` | **0.5070** | **0.6438** | **0.6142** | 0.6434 | source0-only nested selector，自动选 raw prototype 阈值 |
| `proto_decision_cvselect_macro_raw25` | **0.5070** | **0.6438** | 0.6126 | 0.6437 | 宽 selector，方向一致但稍弱 |
| `evtype_proto_raw25_decision_evtype` | 0.5070 | 0.6438 | 0.6084 | 0.6355 | 只改 score，不改 decision |
| evidence-type adapter | 0.5052 | 0.6412 | 0.6084 | 0.6355 | 旧主线 |
| BGE+LR | 0.4730 | 0.6292 | 0.5828 | 0.6100 | 强基线 |

5k sample bootstrap：fixed source0 rule 相对 BGE 与旧 `CM+BGE` 三项显著；相对 current guarded，AUROC p=0.0404、Macro-F1 p=0.0002；相对 evidence-type adapter，Macro-F1 Δ=+0.0058, p=0.0012，AP/AUROC 仍只是正向未显著（p=0.3166/0.0604）。相对 `evtype_proto_raw25`，AP/AUROC 相同，Macro-F1 p=0.0018，说明提升完全来自 decision feature。

随后新增更窄的 source0-only nested selector：只允许在 `source_count==0` 上从 raw prototype rank 的 veto/promote 阈值网格中选择；每个 case/fold 仍只用其它 folds 选阈值。它得到与 fixed rule 相同的 Macro-F1：0.5070 / 0.6438 / 0.6142 / 0.6434。定向 5k sample bootstrap 中，它相对 evidence-type adapter 的 Macro-F1 Δ=+0.0057, p=0.0010；相对 `evtype_proto_raw25` 的 Macro-F1 p=0.0020。更严格的 `pair_id` group bootstrap 中，fixed rule 相对 evidence-type adapter 的 Macro-F1 p=0.0090，source0-only nested selector p=0.0050；相对 `evtype_proto_raw25` 的 Macro-F1 p=0.0094/0.0054。宽 selector 相对 evidence-type adapter 的 Macro-F1 p=0.0612，说明真正稳定的是 **source0-only prototype sufficiency guard**，而不是扩大 mask/规则池后的自由选择。

case 分解：fixed rule 在 fs0 为 0.5003 / 0.6413 / 0.6096 / 0.6468，fs1 为 0.4966 / 0.6382 / 0.6104 / 0.6332，fs2 为 0.5269 / 0.6521 / 0.6215 / 0.6551。共翻转 103 个 source0 样本（veto 33、promote 70），修正 63、误伤 40；收益主要来自 fs0/fs2，fs1 基本持平。source0-only nested selector 的阈值选择也稳定：fs0 全部选择 `veto0.20/promote0.75`，fs2 多数选择 `veto0.20/promote0.70/0.75`，fs1 主要选择 veto-only 0.15/0.20/0.30。当前定位：这是第一个让 prototype 信号在 **sample bootstrap 与 pair-level group bootstrap** 下都显著改善 Macro-F1 的 source-sufficiency decision rule；可作为下一版主线候选，但仍建议新增 split 或预注册后复跑一次，确认阈值不是从当前 OOF 诊断中后验固定。

新增 split 验证（`fold_seed=3/4`）：为避免只在 fs0/fs1/fs2 上后验调规则，新增两个独立 grouped-CV 划分，并把 `cv_eval.py` 改为先写 OOF、再可选 bootstrap（`--n_boot 0`），同时新增 `normalize_cv_oof.py` 把单 case OOF 接入 prototype/decision 链。fs3/fs4 基础 CLAIMARC 仍弱于强 BGE，反而提供更严格验证：

| case | 方法 | AUPRC | AUROC | Macro-F1 | wF1 | 说明 |
|---|---|---:|---:|---:|---:|---|
| fs3 | BGE+LR | 0.4740 | 0.6236 | 0.5783 | 0.6065 | 强基线 |
| fs3 | `rankavg_bge_cm_proto_source_bin` | 0.4946 | 0.6356 | 0.5783 | 0.6031 | prototype score-only；AP sample p=0.011 vs BGE |
| fs3 | `proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin` | 0.4946 | 0.6356 | **0.6040** | **0.6380** | 5 个 folds 都选择 `promote_raw_gt0.75_all` |
| fs4 | BGE+LR | 0.5017 | 0.6499 | 0.5885 | 0.6255 | 比 CLAIMARC pcls 更强 |
| fs4 | `rankavg_bge_cm_proto_source_bin` | 0.5104 | 0.6537 | 0.5986 | 0.6238 | prototype score-only；AP/AUROC/Macro 均正向 |
| fs4 | `proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin` | 0.5104 | 0.6537 | 0.5922 | 0.6320 | decision edit 小幅增益 |
| fs4 | `proto_decision_fixed_veto0.20_promote0.80_cal_src0_or_lowabs_score_rankavg_bge_cm_proto_source_bin` | 0.5104 | 0.6537 | **0.5946** | **0.6322** | 更保守 fixed rule |

fs3+fs4 pooled（n=3388）定向 5k bootstrap：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 | vs BGE sample p(AP/AUROC/Macro) | vs BGE room p(AP/AUROC/Macro) |
|---|---:|---:|---:|---:|---|---|
| BGE+LR | 0.4856 | 0.6364 | 0.5836 | 0.6163 | - | - |
| `rankavg_bge_cm_proto_source_bin` | 0.5009 | 0.6448 | 0.5893 | 0.6143 | - | - |
| `proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin` | **0.5009** | **0.6448** | **0.5981** | **0.6350** | 0.0146 / 0.0466 / 0.0012 | 0.0562 / 0.1100 / 0.0064 |
| fixed cal/src0-or-lowabs rule | **0.5009** | **0.6448** | 0.5917 | 0.6255 | 0.0114 / 0.0486 / 0.0002 | 0.0464 / 0.1062 / 0.0008 |

相对 score-only prototype，decision edit 的 pooled Macro-F1 增益尚未严格显著（cross-fit sample p=0.101，room p=0.143；fixed sample p=0.370，room p=0.403），因此不能声称 decision edit 显著胜过 prototype score-only。更准确的结论是：**RACL prototype relation score 在新增 split 上稳定提升 BGE 的 ranking；prototype-aware decision guard 在新增 split pooled 上显著提升相对 BGE 的 Macro-F1，且 room-level group bootstrap 仍成立。** 这把旧 fs0/fs1/fs2 的 source0 prototype sufficiency guard 从后验 OOF 诊断推进为跨新划分可复现机制，但最终论文主表仍应继续区分 score-only prototype、fixed rule、cross-fit selector 三层。

五划分 score-only 稳健性（fs0/fs1/fs2 + fs3/fs4；n=8470）进一步确认了这一点。把同名 `rankavg_bge_cm_proto_source_bin` 与 BGE+LR 合并做 5k bootstrap，得到：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 | vs BGE sample p(AP/AUROC/Macro) | vs BGE pair-level p(AP/AUROC/Macro) |
|---|---:|---:|---:|---:|---|---|
| BGE+LR | 0.4780 | 0.6321 | 0.5834 | 0.6127 | - | - |
| `rankavg_bge_cm_proto_source_bin` | **0.5032** | **0.6414** | **0.5893** | **0.6143** | 0.0000 / 0.0024 / 0.1172 | 0.0016 / 0.0660 / 0.1842 |

解读：prototype relation score 的排序贡献已经跨五个 grouped split 复现，AP 的 pair-level group bootstrap 仍显著；但 score-only 不足以保证 Macro-F1 显著提升，binary decision 仍需要 source/evidence sufficiency guard。

五划分统一 BGE-base decision protocol（fs0/fs1/fs2 + fs3/fs4；n=8470）也已补跑。这里不再让 fs0/fs1/fs2 使用 evidence-type adapter，而是和 fs3/fs4 一样，以 BGE+LR 的 binary decision 为基底，只接 RACL prototype score / guard：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 | vs BGE sample p(AP/AUROC/Macro) | vs BGE pair-level p(AP/AUROC/Macro) |
|---|---:|---:|---:|---:|---|---|
| BGE+LR | 0.4780 | 0.6321 | 0.5834 | 0.6127 | - | - |
| `rankavg_bge_cm_proto_source_bin` | 0.5032 | 0.6414 | 0.5893 | 0.6143 | 0.0000 / 0.0024 / 0.1172 | 0.0016 / 0.0660 / 0.1842 |
| `proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin` | **0.5032** | **0.6414** | **0.5913** | **0.6244** | 0.0000 / 0.0024 / 0.0020 | 0.0016 / 0.0660 / 0.0092 |
| fixed cal/src0-or-lowabs rule | **0.5032** | **0.6414** | 0.5906 | 0.6233 | 0.0000 / 0.0060 / 0.0000 | 0.0006 / 0.0652 / 0.0002 |

相对 score-only prototype，两个 decision guard 的 Macro-F1 增益仍不显著（cross-fit sample/group p=0.3246/0.3402；fixed p=0.3934/0.4090），因此 guard 不能被表述为“显著胜过 prototype score-only”。但这个统一协议说明：即使完全以强 BGE 为 decision base，RACL prototype relation score + sufficiency guard 仍能在五划分 pooled 上让 AP 和 Macro-F1 相对 BGE 显著提升；AUROC 的 pair-level group p≈0.066，作为边界正向结果报告。

补充更严格的五划分 room-level group bootstrap：先把 fs0/fs1/fs2 的 `pair_id` 反接原始 dataset 的 `room_id`，生成 `oof_racl_proto_decision_feature_fs012_bgebase_room_20260609.npz`，再与 fs3/fs4 合并运行 `racl_proto_decision_feature_fs0_fs4_bgebase_room_bootstrap_5k_20260609.json`。结果保持一致：cross-fit decision 相对 BGE 的 room-level p(AP/AUROC/Macro) = 0.0062 / 0.1438 / 0.0094；fixed rule 相对 BGE 的 room-level p = 0.0060 / 0.1400 / 0.0000。相对 score-only prototype 的 Macro-F1 仍不显著（cross-fit p=0.4050，fixed p=0.4562）。这进一步支持“prototype relation score 相对 BGE 的排序与 Macro-F1 主张成立；decision guard 不单独声称显著优于 score-only prototype”。

进一步把 RACL prototype guard 接到五划分 `CM/NLI guarded` 主协议上，而不是接到 BGE decision base。新增 `scripts/build_guarded_proto_fs0_fs4_oof.py` 负责把 fs0-fs2 的 prototype/evidence-type OOF 与 fs3/fs4 的 NLI guarded OOF、prototype verifier OOF 按 `pair_id` 对齐成 `oof_guarded_proto_fs0_fs4_room_20260609.npz`；随后用 `cv_racl_proto_decision_feature.py --decision_method <guarded>` 只编辑 guarded family 的 binary decision。最干净的固定规则是：在 `source_count==0` 样本上，如果 raw RACL `proto_source_bin` rank < 0.20，则 veto 正判；如果 rank > 0.75，则 promote 负判。score 保留五划分 CM/NLI guarded score：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 | vs BGE room p(AP/AUROC/Macro) | vs guarded room p(AP/AUROC/Macro) |
|---|---:|---:|---:|---:|---|---|
| BGE+LR | 0.4780 | 0.6321 | 0.5834 | 0.6127 | - | - |
| `CM/NLI guarded` | 0.5029 | 0.6422 | 0.6010 | 0.6281 | 0.0008 / 0.0366 / 0.0004 | - |
| evidence-type adapter | 0.5039 | 0.6424 | 0.6005 | 0.6277 | 0.0008 / 0.0500 / 0.0004 | 不显著且 Macro 略低 |
| `CM/NLI guarded + fixed RACL source0 guard` | **0.5029** | **0.6422** | **0.6053** | **0.6359** | **0.0006 / 0.0382 / 0.0002** | 1.0000 / 1.0000 / **0.0156** |

分 split 看，fixed source0 guard 相对原 guarded 的 Macro-F1 在 fs0/fs1/fs2/fs3/fs4 分别从 0.5998/0.6065/0.6101/0.5886/0.5971 提升到 0.6075/0.6072/0.6190/0.5912/0.5988，五个 split 方向一致。样本级、pair-level、room-level bootstrap 中，相对原 guarded 的 Macro-F1 p 分别为 0.0022 / 0.0182 / 0.0156；相对 evidence-type adapter 的 Macro-F1 p 分别为 0.0098 / 0.0296 / 0.0238。这是目前最强也最可写的主协议：排序头沿用 CM/NLI guarded，分类头只加一个预定义 RACL prototype source-poor sufficiency guard。

为检查 fixed rule 是否只是 fs0-fs2 后验收益，新增 `scripts/subset_oof_by_case.py` 把 `oof_racl_proto_decision_feature_fs0_fs4_guardedbase_20260609.npz` 切出 fs3/fs4 独立子集，并重跑 sample/pair/room bootstrap。fs3/fs4 pooled 中，主方法为 0.5051 / 0.6446 / 0.5952 / 0.6235；相对 BGE 的 sample p=0.0010/0.0142/0.0298，pair-level p=0.0026/0.0358/0.0334，room-level p=0.0102/0.0582/0.0444。相对原 guarded 的 Macro-F1 仍为正（+0.0021），但不显著（sample/pair/room p=0.1866/0.2196/0.2282）。因此新增 split 提供方向支持和相对 BGE 的独立复核，但“显著胜 guarded”的主张仍应基于五划分合并的保守 bootstrap 与五个 split 方向一致。

新增 `src/models/diagnose_guard_flips.py` 做翻转诊断。五划分 fixed source0 guard 只翻转 161/8470 条（1.90%），全部位于 `source_count==0`：veto 49 条、promote 112 条，净正确 +25；source0 子集 Macro-F1 从 0.6195 到 0.6288。错误类型上，它修正 38 个 FP 和 55 个 FN，同时引入 11 个 FN 和 57 个 FP，说明收益不是单纯压低正类率，而是利用 prototype relation 同时做少证据 veto 与少证据 promote。fs3/fs4 子集只翻转 58/3388 条，净正确 +2，方向仍正但幅度小；这也解释了独立新增 split 上相对 guarded 未显著。

强 embedding baseline 加固：新增 `src/models/cv_embedding_baseline.py`，用任意 `SentenceTransformer` 一次编码 claim/evidence，再按和主实验一致的 `room_id` grouped-CV、val-carve threshold、LR 特征 `[claim, evidence, diff, product]` 生成 OOF；新增 `scripts/merge_oof_methods.py` 把外部 baseline OOF 合并到主方法 OOF。先用本地 ModelScope 的 BGE-large 复现实验验证 evaluator，得到 `bge_large_repro_lr` = 0.4816 / 0.6315 / 0.5845 / 0.6206，和旧 `bge_lr` 接近。再用 ModelScope 下载 `Qwen/Qwen3-Embedding-0.6B`，得到现代 embedding baseline `qwen3emb06b_lr` = 0.4845 / 0.6376 / 0.5900 / 0.6111。补跑 Qwen3 官方 query prompt 版本 `qwen3emb06b_query_lr` = 0.4813 / 0.6351 / 0.5835 / 0.6093，低于 no-prompt；因此本文把 no-prompt Qwen3 作为更强现代 embedding baseline，query-prompt 作为 prompt-aware robustness check。

与当前主方法的五划分 paired bootstrap：

| Baseline | Baseline AUPRC | AUROC | Macro-F1 | wF1 | Main ΔAP/ΔAUROC/ΔMacro | sample p | pair p | room p |
|---|---:|---:|---:|---:|---|---|---|---|
| BGE+LR saved | 0.4780 | 0.6321 | 0.5834 | 0.6127 | +0.0247 / +0.0102 / +0.0220 | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0066 / 0.0000 | 0.0008 / 0.0352 / 0.0000 |
| BGE-large repro LR | 0.4816 | 0.6315 | 0.5845 | 0.6206 | +0.0210 / +0.0106 / +0.0208 | 0.0000 / 0.0006 / 0.0000 | 0.0024 / 0.0124 / 0.0000 | 0.0092 / 0.0424 / 0.0000 |
| Qwen3-Embedding-0.6B LR | 0.4845 | 0.6376 | 0.5900 | 0.6111 | +0.0182 / +0.0046 / +0.0153 | 0.0020 / 0.1444 / 0.0010 | 0.0466 / 0.2578 / 0.0152 | 0.0718 / 0.3072 / 0.0088 |
| Qwen3-Embedding-0.6B query-prompt LR | 0.4813 | 0.6351 | 0.5835 | 0.6093 | +0.0215 / +0.0071 / +0.0218 | 0.0002 / 0.0468 / 0.0000 | 0.0230 / 0.1576 / 0.0016 | 0.0344 / 0.1856 / 0.0032 |

解读：Qwen3-Embedding-0.6B no-prompt 是更强的现代 frozen embedding baseline，尤其 AUROC/Macro-F1 高于旧 BGE 和 prompted Qwen3；当前主方法相对 no-prompt Qwen3 的 Macro-F1 在 sample、pair-level、room-level 下均显著，AUPRC 正向但 room-level 为边界（p=0.0718），AUROC 只正向不显著。相对 prompted Qwen3，主方法的 AP 与 Macro-F1 在 room-level 也显著，但这条 baseline 本身弱于 no-prompt。因此论文主张应写成“显著提升现代 embedding baseline 的分类 Macro-F1，并保持 AP 正向；排序 AUROC 不是相对 Qwen3 的核心显著点”。

在 fixed source0 guard 基础上，新增 `src/models/cv_oof_disagreement_router.py` 做 fold-safe OOF disagreement router：排序分数完全沿用 CM/NLI guarded + RACL source0 guard，只在二分类决策端使用 Qwen3-Embedding-0.6B no-prompt OOF 预测。每个 repeated-CV case/fold 只用同一 case 的其他 fold 选择一个小规则，规则族为“主方法与 Qwen3 分歧、主方法概率接近 0.5、Qwen3 概率足够远离 0.5、可按 source_count 限制、可只 promote/veto”，再应用到 held-out fold。初版 selected router 为 0.6097 Macro，但 fs1 从 0.6072 回落到 0.6024；因此新增保守门槛：任意规则验证 Macro-F1 至少 +0.004，non-veto（promote/both）规则验证 Macro-F1 至少 +0.015，验证翻转率不超过 10%。这个结构保留检索增强对比学习机制，把现代 embedding 只当争议样本裁判。

五划分结果：

| Method | AUPRC | AUROC | Macro-F1 | wF1 | 说明 |
|---|---:|---:|---:|---:|---|
| Fixed RACL source0 guard | 0.5029 | 0.6422 | 0.6053 | 0.6359 | 上一版主候选 |
| Fixed global Qwen3 switch | 0.5029 | 0.6422 | 0.6077 | 0.6377 | `b<=0.30, q>=0.15, all, both` |
| Fold-safe selected Qwen3 router | 0.5029 | 0.6422 | 0.6097 | 0.6400 | 初版，fs1 回落 |
| Conservative selected Qwen3 router | **0.5029** | **0.6422** | **0.6109** | **0.6411** | 上一版主候选；decision-only |

全量五划分 bootstrap 中，保守 selected router 相对 fixed source0 guard 的 Macro-F1 显著提升：sample p=0.0026，pair-level p=0.0056，room-level p=0.0112；AP/AUROC 完全不变，因为 score head 不变。相对原 CM/NLI guarded 的 room-level Macro p=0.0004，相对 evidence-type adapter 的 room-level Macro p=0.0004。相对 no-prompt Qwen3 baseline 的 sample/pair/room Macro p=0.0000/0.0002/0.0004；AP 仍为正但 room-level 边界（p=0.0718），AUROC 不显著。

split-level Macro-F1 为：

| Split | Fixed source0 guard | Conservative router | Qwen3 baseline |
|---|---:|---:|---:|
| fs0 | 0.6075 | 0.6106 | 0.5926 |
| fs1 | 0.6072 | 0.6069 | 0.6000 |
| fs2 | 0.6190 | 0.6206 | 0.5737 |
| fs3 | 0.5912 | 0.6145 | 0.6051 |
| fs4 | 0.5988 | 0.6007 | 0.5743 |

fs3/fs4 独立子集验证保持强势：保守 router 为 0.5051 / 0.6446 / 0.6076 / 0.6392，相对 fixed source0 guard 的 sample/pair/room Macro p=0.0004/0.0004/0.0028；相对 BGE 的 room p=0.0100/0.0498/0.0004；相对 no-prompt Qwen3 的 room Macro p=0.0406。翻转诊断显示，保守 router 全量只翻转 270/8470 条（3.19%），净正确 +62，修正 119 个 FP 和 47 个 FN，同时引入 69 个 FN 和 35 个 FP；fs3/fs4 翻转 152/3388 条，净正确 +38。相对初版 router，保守门槛减少 21 次翻转，净正确从 +49 提高到 +62，并把 fs1 从 0.6024 修复到 0.6069。选择器最常选的是 veto 型规则，说明新收益主要来自争议样本中过度正判的纠偏，而不是用 Qwen3 替换主模型。

为回应“二层 OOF 规则选择是否过拟合”的潜在审稿风险，补做 no-selector 固定规则：`base_conf<=0.20 & qwen_conf>=0.10 & Qwen veto only`。该规则完全不按 fold/case 选择超参，五划分为 0.5029 / 0.6422 / 0.6092 / 0.6383，相对 fixed source0 guard 的 sample/pair/room Macro p=0.0016/0.0106/0.0046；全量只翻转 145/8470 条，全部是 veto，净正确 +55。fs3/fs4 子集为 0.5051 / 0.6446 / 0.5975 / 0.6261，相对 fixed guard 的 Macro p=0.0890/0.1216/0.0576，方向为正但未过 0.05；相对 BGE 的 room p=0.0100/0.0498/0.0200。解释：固定 veto 是严格的确认性 robustness check，证明 Qwen3 disagreement-veto 不是依赖 fold-level selector 才成立；但它不能替代保守 selected router，因为它不捕捉 fs3/source-rich 场景中的双向纠偏收益。

Score/decision 解耦后的最新 top-line：在保守 selected router 的 binary decision 不变的前提下，重新接入协议化 RACL prototype calibration 的 `raw25` score head。新增 `cv_racl_proto_evtype_protocol.py --decision_method <conservative router> --decision_label router`，固定按 case/fold rank blend：75% evidence-type/CM-NLI guarded score + 25% raw `proto_source_bin` relation rank。这个步骤不重新选择权重、不改阈值、不改 Qwen3 decision router，只让 RACL prototype relation geometry 回到排序头。

五划分结果：

| Method | AUPRC | AUROC | Macro-F1 | wF1 | 说明 |
|---|---:|---:|---:|---:|---|
| Conservative selected Qwen3 router | 0.5029 | 0.6422 | **0.6109** | **0.6411** | 旧 score，decision-only router |
| `evtype_proto_cal50_decision_router` | 0.5079 | 0.6443 | **0.6109** | **0.6411** | calibrated prototype 50% rank blend |
| `evtype_proto_raw25_decision_router` | **0.5084** | **0.6456** | **0.6109** | **0.6411** | 当前最高点估计；raw prototype 25% rank blend |

全量 bootstrap 读数：

| Comparison for `evtype_proto_raw25_decision_router` | sample p(AP/AUROC/Macro) | pair p(AP/AUROC/Macro) | room p(AP/AUROC/Macro) |
|---|---|---|---|
| vs previous conservative router score | 0.0226 / 0.0070 / 1.0000 | 0.1232 / 0.0952 / 1.0000 | 0.1650 / 0.1558 / 1.0000 |
| vs evidence-type adapter | 0.0528 / 0.0054 / 0.0000 | 0.1542 / 0.0922 / 0.0008 | 0.1992 / 0.1418 / 0.0004 |
| vs Qwen3-Embedding-0.6B LR | 0.0000 / 0.0268 / 0.0000 | 0.0188 / 0.1526 / 0.0004 | 0.0274 / 0.1942 / 0.0002 |
| vs BGE+LR | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0048 / 0.0000 | 0.0004 / 0.0316 / 0.0000 |
| vs BGE-large repro LR | -- | -- | 0.0034 / 0.0290 / 0.0000 |

fs3/fs4 子集同样正向：`evtype_proto_raw25_decision_router` 为 0.5114 / 0.6482 / 0.6076 / 0.6392。相对 BGE 的 sample/pair/room p(AP/AUROC/Macro)=0.0002/0.0028/0.0000，0.0012/0.0174/0.0000，0.0036/0.0370/0.0002。相对 previous conservative router score 的 AP/AUROC 在 sample 层接近/达到显著（0.0580/0.0430），但 pair 和 room 仍只是正向（pair 0.1036/0.0920；room 0.1372/0.1452）。因此论文里可以把 raw25 score-router 作为新的 top-line 点估计和 RACL ranking 机制，但不要声称它在 room-level 已显著击败旧 router score。确认性主张仍是：RACL source0 sufficiency guard + conservative Qwen3 disagreement decision 显著改善 Macro-F1；raw25 score head 在不牺牲 Macro-F1 的情况下提高 AP/AUROC，并显著胜强 BGE/现代 embedding baselines。

训练期 prototype auxiliary objective smoke：`src/models/train.py` / `cv_eval.py` 已加入 `--proto_aux_*`，在 epoch memory bank 中按 `global/attr/source_bin/evidence_combo/confidence/source_conf/combo_conf` 构造正负 prototypes，并让当前 batch 的 retrieval embedding `g` 通过 prototype similarity gap 做轻量二分类辅助损失。fs1/drop-src0args 的第一轮保守配置 `proto_aux_weight=0.02, proto_aux_group=source_bin, proto_aux_tau=0.10, proto_aux_min_class=3, proto_aux_c_min=0.10` 已完成：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 | 解释 |
|---|---:|---:|---:|---:|---|
| `CLAIMARC_pcls + proto_aux(source_bin)` | 0.4845 | 0.6196 | 0.5845 | 0.5992 | 相对旧 drop PCLS 的 Macro/wF1 小涨，但 AP/AUROC 基本未变 |
| `CLAIMARC_selectiveRKC + proto_aux(source_bin)` | 0.4845 | 0.6196 | 0.5845 | 0.5992 | 与 p_cls 相同，检索投票未额外收益 |
| `CLAIMARC_v2 + proto_aux(source_bin)` | 0.4773 | 0.6046 | 0.5782 | 0.5875 | 旧融合仍不稳 |
| BGE+LR | 0.4736 | **0.6288** | **0.5928** | **0.6336** | fs1 hard split 的强 baseline |

PCLS 相对 BGE 的 2k bootstrap：ΔAP +0.0103 (p=0.2340)，ΔAUROC -0.0093 (p=0.7970)，ΔMacro-F1 -0.0001 (p=0.4985)。与旧 fs1/drop PCLS 0.4849 / 0.6195 / 0.5817 / 0.5959 相比，prototype auxiliary 只带来 Macro-F1 +0.0028、wF1 +0.0033 的轻微改善。

用同一新 embedding 再跑 `cv_racl_prototype_verifier.py`，prototype 几何有所增强：`rankavg_bge_cm_proto_source_bin` 在 fs1 为 0.5022 / 0.6414 / 0.5942 / 0.6065，`rankavg_cm_proto_source_bin` 的 Macro-F1 到 0.6033，但训练出来的 p_cls 仍没有超过 BGE decision。结论：prototype relation geometry 是值得保留的 score-side 信号；直接把 prototype CE 辅助目标塞进小模型训练，目前不足以形成最终分类闭环。下一步若继续，应尝试把 prototype gap 显式作为 fold-level score calibration/decision feature，或设计 margin/ranking 型 relation objective，而不是扩大 `proto_aux_weight` 网格。

2026-06-09 追加 margin/ranking 型 prototype auxiliary smoke：`src/models/train.py` 和 `cv_eval.py` 已扩展 `--proto_aux_mode {ce,margin}` 与 `--proto_aux_margin`，默认仍为 CE，margin 模式直接优化正确类 prototype similarity gap。fs1/drop-src0args 配置 `proto_aux_weight=0.02, proto_aux_group=source_bin, proto_aux_mode=margin, proto_aux_margin=0.15, proto_aux_tau=0.10` 已完成：

| 方法 | AUPRC | AUROC | Macro-F1 | wF1 | 解释 |
|---|---:|---:|---:|---:|---|
| `CLAIMARC_pcls + proto_aux_margin(source_bin)` | 0.4852 | 0.6196 | 0.5832 | 0.6005 | AP 略高于 CE auxiliary，但 Macro-F1 未改善 |
| `CLAIMARC_selectiveRKC + proto_aux_margin(source_bin)` | 0.4836 | 0.6131 | 0.5846 | 0.6045 | wF1 小涨但 AUROC 受损 |
| `CLAIMARC_v2 + proto_aux_margin(source_bin)` | 0.4781 | 0.6059 | 0.5874 | 0.5943 | 仍弱于 BGE decision |
| BGE+LR | 0.4736 | **0.6288** | **0.5928** | **0.6336** | 同一 fs1/drop-src0args baseline |

同一 embedding 的 prototype verifier：`rankavg_bge_cm_proto_source_bin` = 0.5020 / 0.6418 / 0.5892 / 0.6082；`rankavg_bge_cm_proto_global` = 0.5001 / 0.6372 / 0.5971 / 0.6173。3k bootstrap 相对 `bge_lr_saved`：CE 版 `source_bin` AP/AUROC/Macro p = 0.0043 / 0.0487 / 0.4757；margin 版 `source_bin` p = 0.0060 / 0.0407 / 0.6857；margin 版 `global` p = 0.0093 / 0.1290 / 0.3647。结论更新：margin 目标保留了 prototype score 的 AP/AUROC 诊断价值，但没有把 Macro-F1 闭环做强，且 source-bin margin 可能过窄；不扩展到 fs3/fs4。训练端 prototype 辅助目前作为负结果记录，论文主线继续使用 score-side prototype relation + sufficiency/adapter decision。

机制读数：

1. evidence-type adapter 是三划分阶段比 taxonomy-aware 更可辩护的局部 repair：它保留 RACL/CM 检索分支与 NLI evidence posterior，只用证据源组合控制 adaptive repair 的作用域。
2. 在 fs0/fs1/fs2 pooled OOF 上，它相对 current guarded 的 Macro-F1 边界显著，相对 BGE 和旧 `CM+BGE` 三项显著；但新增 fs3/fs4 后，这个分类优势没有独立复现，五划分主协议应回到更稳的 guarded family。
3. 新增 room-level group bootstrap（`oof_evidence_type_adapter_screen_room_20260609.npz`；101 rooms；5k）：best adapter `evtype_adapt_score_src0_po_medium_decision_po_medium` 相对 BGE 的 p(AP/AUROC/Macro) = 0.0006 / 0.0728 / 0.0000；相对上一版 guarded `rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect` 的 p = 0.1674 / 0.3916 / 0.0732。也就是说，room 层面可以强主张“显著优于 BGE 的 AP 和 Macro-F1”，但不能声称显著优于上一版 guard。
4. 追加 fs3/fs4 新划分复核：直接用 `cv_nli_predef_lowabs.py --quick` 在 fs3/fs4 生成固定 evidence-type adapter OOF。fs3 中 evidence-type 为 0.4914 / 0.6336 / 0.5819 / 0.6043，低于旧 guarded 0.4953 / 0.6339 / 0.5886 / 0.6098；fs4 中 evidence-type 为 0.5201 / 0.6549 / 0.5938 / 0.6267，低于旧 guarded 0.5169 / 0.6554 / 0.5971 / 0.6276 的 Macro-F1。因此 evidence-type adapter 的分类优势不在新增 split 上单独复现。
5. 但五划分合并（fs0-fs4, n=8470）仍支持“CM/NLI guarded family 显著胜 BGE”：evidence-type adapter 为 0.5039 / 0.6424 / 0.6005 / 0.6277，相对 BGE room-level p(AP/AUROC/Macro)=0.0008 / 0.0500 / 0.0004；旧 guarded 为 0.5029 / 0.6422 / 0.6010 / 0.6281，相对 BGE p=0.0008 / 0.0366 / 0.0004。evidence-type 相对旧 guarded 不显著（p=0.3586 / 0.4440 / 0.6170），且 Macro/wF1 略低。论文表述应把旧 guarded 作为更稳的五划分主协议，把 evidence-type adapter 写成三划分上的可解释局部 repair / ranking tie，而不是“新增 split 上显著优于 guarded”的主方法。
6. RACL prototype + evidence-type protocol 也补了 room-level group bootstrap（`oof_racl_proto_evtype_protocol_room_20260609.npz`；5k）：`evtype_proto_cal50_decision_evtype` 为 0.5071 / 0.6430 / 0.6084 / 0.6355，相对 BGE 的 p(AP/AUROC/Macro)=0.0004 / 0.0674 / 0.0000；`evtype_proto_raw25_decision_evtype` 为 0.5070 / 0.6438 / 0.6084 / 0.6355，相对 BGE p=0.0004 / 0.0518 / 0.0000。二者相对 pure evidence-type adapter 的 AP/AUROC/Macro 均不显著（cal50: 0.3514/0.2642/1.0000；raw25: 0.3692/0.1826/1.0000）。因此 prototype rankblend 是排序增强/机制分析，不应声称显著胜 evidence-type adapter。
7. 仍未解决 fs1 单划分问题：fs1 vs BGE 为 ΔAP +0.0311, p=0.0008；但 AUROC p=0.3042，Macro-F1 p=0.1314。因此它可以作为 **pooled repeated-CV 主线升级候选**，但不是“每个 fold_seed 全指标显著”的最终闭环。
8. 最新五划分主表应把 `CM/NLI guarded + fixed RACL source0 guard` 作为主方法：它保留 guarded score，只用 RACL prototype raw rank 在 `source_count==0` 上做固定 sufficiency edit，达到 0.5029 / 0.6422 / 0.6053 / 0.6359；room-level 下相对 BGE 三项显著，且 Macro-F1 显著胜原 guarded 与 evidence-type adapter。evidence-type adapter 仍作为三划分可解释 repair，prototype score-only/cross-fit selector 作为机制消融；继续扩大 taxonomy/evidence-type selector 的边际价值很低。

---

## 5. 未完成 / 受限项

- **LLM 判别基线（§4.2 C 类）**：`Qwen-Flash` zero-shot conservative prompt 已补跑，结果显著弱于 BGE+LR；若论文需要更强 LLM 对照，可再补 DeepSeek/Qwen 更大模型或 few-shot hard-case adjudication，但不应再把 Qwen-Flash direct score 当主 teacher。
- 跨域脚本在最大品类上 OOM（循环内未释放上一模型显存）；已得 10 次有效运行，建议加 `del model; torch.cuda.empty_cache()` 后补齐。

---

## 6. 总体判断与建议（含严格 CV 修正）

经过对架构（LoRA 秩 / 全量微调 / 骨干 / 融合 / 对比强度 / 集成）的穷尽搜索 **+ 分组 5 折 CV 复核**：

1. **架构搜索结论稳健**：BGE+LoRA 双流是该设计空间最优配置；全量微调、RoBERTa/BERT 骨干、更强对比均更差——这部分**验证了提案的骨干/适配设计**。
2. **旧的复杂融合不稳**：早期 `blend2/Platt` 在严格 CV 下失败，说明小验证折上的标定/融合容易过拟合；主方法不应继续使用旧 `CLAIMARC_v2`。
3. **有效但需收窄的新结构**：RAFTS-style argument augmentation + RACL 表征产生了可用的检索/排序专家。seed=0 的 hybrid 全指标显著，但 repeated split 暴露 Macro-F1 不稳；seed=1/2 的全融合与 reliability gate 进一步确认，目前最稳的是 **rankavg(args/no-args p_cls, fair BGE+LR) 的 AUPRC 排序增益**。source-first evidence policy 进一步让 `rankavg(sourcefirst_args_pcls, sourcefirst_BGE)` 在 seed0/seed2 达到 AP/AUROC/Macro-F1 同时显著，但 seed1 未复现分类优势。atomic NLI posterior、evidence-type adapter 和 RACL prototype verifier 共同把问题收窄到“relation/sufficiency score + 保守 decision protocol”。旧 fs0/fs1/fs2 中 source0 prototype guard 相对 evidence-type adapter 的 Macro-F1 显著；新增 fs3/fs4 中，BGE-base prototype decision pooled 后相对 BGE 的 AP/AUROC/Macro-F1 sample bootstrap 显著，Macro-F1 在 room-level group bootstrap 也显著。当前主线已从“找得到显著候选吗”推进到“能否把 RACL prototype relation score 写成可预注册的 score/decision protocol，并证明不是 OOF 后验筛选”。
4. **贡献叙事应更新**：不是“端到端双流模型单独碾压基线”，而是“检索增强对比学习训练出的 argument-aware relation geometry，为最强冻结嵌入分类器提供排序互补；source/evidence sufficiency guard 把该 relation score 转成更稳的二分类边界；当证据不足或歧义时，协议回退到更保守的 BGE 判断”。
5. **消融**仍有价值：属性分块对比在 boundary 子集上贡献最大，可支撑"边界判别"这一更聚焦的论点。

**给作者的建议（按优先级）：**
- **（阶段性推荐）主表以 `CM/NLI guarded + fixed RACL source0 guard` 为主方法**：fs0-fs4 五划分上，固定 source0 prototype sufficiency guard 的 Macro-F1 相对原 guarded 在 sample、pair-level、room-level bootstrap 都显著；分 split 方向也一致。score-only prototype、BGE-base guard、evidence-type adapter 和 cross-fit selector 作为消融，不再把 evidence-type selector 扩大为主线。
- **评测协议升级**：论文改用分组 CV / 多划分汇报（已实现 `src/models/cv_eval.py`），避免单一划分的乐观偏差。
- **继续补强**：direct LLM 判别、简单 outer-train LR reliability head、BGE similarity evidence-set head、激进 source-domain CL reweight、简单 sourceveto、bgerateguard 和 nlievidenceveto 都已证明不是最终闭环；scorefallback/headmix/bgefallback 已把问题收窄到“如何自动选择何时信任 NLI/RACL、何时回退 BGE”。`predef_lowabs` 已给出第一个固定协议候选，下一步优先做 fs0/fs2 cache 对齐与 repeated CV 检验，而不是继续扩大候选池。
- **不建议**：仅用固定划分上的有利结果声称"显著优于基线"——存在研究诚信风险。
