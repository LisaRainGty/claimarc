# CLAIMARC 项目逐文件理解与进展地图

> 更新时间：2026-06-09。  
> 目的：把当前工作区里的代码、文档、脚本和关键结果逐文件定位，形成后续继续实验的导航图。

## 0. 当前研究状态

CLAIMARC 的研究对象是直播电商中的 `(product_id, attribute_id)` 级虚假宣传风险识别。数据流水线已经从评论、主播 SRT、商品参数/OCR/VLM 三源证据构造出 pair-level 数据集；模型侧已经完成从早期 CLAIMARC-v2 到 clean-RACL、argument augmentation 和 hybrid OOF 融合的一系列复核。

最重要的结论是：

1. 早期固定划分上 CLAIMARC-v2 看起来能超过基线，但严格 grouped CV 显示旧的 blend2/Platt 融合不稳。
2. clean-RACL 的低容量、置信过滤、warmup+CL 方向有效，单模型在一些划分上全指标正向，但对最强 fair BGE+LR 还未稳定显著。
3. RAFTS-style LLM argument augmentation 明显增强排序/检索专家，`rankavg(args/no-args p_cls, BGE+LR)` 的 AUPRC 提升更稳；Macro-F1 在 repeated grouped split 下仍不稳定。
4. 目前不能把 seed=0 的全指标显著结果当成最终论文主张。seed=1/2 全融合与 reliability gate 确认：排序增益可复现，`rankavg(args/no-args p_cls, fair BGE+LR)` 已在 `fold_seed=0/1/2` 稳定显著提升 AUPRC；分类 Macro-F1/wF1 仍未稳定显著超过 fair BGE+LR。最新进展是把 source-first CM p_cls 检索分支接回 predef-only evaluator，并与 NLI evidence posterior / RACL prototype relation score 做固定 rank-weighted fusion，再接 source-sufficiency guarded decision。旧三划分中 evidence-type adapter 到 0.5052/0.6412/0.6084，room-level group bootstrap 相对 BGE 的 p(AP/AUROC/Macro) = 0.0006/0.0728/0.0000；相对上一版 guard 不显著。新增 fs3/fs4 复核显示 evidence-type adapter 分类优势不单独复现，但五划分合并仍为 0.5039/0.6424/0.6005/0.6277，相对 BGE room-level p=0.0008/0.0500/0.0004；旧 guarded 为 0.5029/0.6422/0.6010/0.6281，相对 BGE p=0.0008/0.0366/0.0004，且 Macro/wF1 略高。当前 top-line 候选是 `RACL raw25 score + fixed RACL source0 guard + conservative Qwen3 disagreement router decision`，五划分为 0.5084/0.6456/0.6109/0.6411；相对 BGE room-level p=0.0004/0.0316/0.0000，相对 no-prompt Qwen3 的 room-level p=0.0274/0.1942/0.0002。raw25 score 相对旧 conservative router score 的 AP/AUROC 是正向点估计，但 pair/room group bootstrap 未显著；因此它是新的主表点估计和 RACL ranking 机制，不是“显著击败旧 score”的确认性主张。当前主线已从“端到端 CLAIMARC 单模型”转为“RACL relation geometry + source/evidence sufficiency decision protocol”。

## 1. 根目录与环境文件

| 文件 | 作用 | 当前进展 |
|---|---|---|
| `README.md` | 项目总说明，定义 CLAIMARC 名称、数据来源、目录结构和三阶段流水线。 | 说明仍偏早期，`processed/` 和 `final/` 已不再为空；后续可更新数据规模和实验结论。 |
| `src/README.md` | 代码流水线说明，列出 Stage A/B/C/labels/final 的输入输出。 | 与当前代码结构一致，是数据构造部分的快速入口。 |
| `requirements.txt` | 运行依赖的宽松版本。 | 可用于远程新机器安装。 |
| `requirements_lock.txt` | 更固定的依赖快照。 | 用于复现实验环境。 |
| `env.sh` | 本地/远程环境变量入口。 | 注意不要把密钥写入文档或输出；运行前只在 shell 中加载。 |

## 2. 文档目录

| 文件 | 作用 | 当前进展 |
|---|---|---|
| `docs/Methodology_Data.md` | 详细方法与数据构造文档：Stage A 评论属性抽取、Stage B 主播 claim 对齐、Stage C 商品事实证据、弱监督标签和数据 schema。 | 是数据构造和标签逻辑的主文档，内容较完整。 |
| `docs/Experiment_Results_v2.md` | 实验结果主日志，记录 CLAIMARC-v2、严格 CV、clean-RACL、argument augmentation、hybrid、distillation、teacher-guided RACL 和 seed 复核。 | 当前最关键的研究进展文档。最新闭环是五划分 `CM/NLI guarded + fixed RACL source0 guard`：room-level 下相对 BGE 三项显著，且 Macro-F1 显著胜原 guarded。 |
| `docs/FINAL_PROTOCOL_STATUS_20260609.md` | 当前主协议状态页：把 confirmatory 主张、exploratory 机制、不能声称的点和关键产物拆开。 | 新增于 2026-06-09；打开项目时优先看这一页判断论文主表。 |
| `docs/RESTORE_ON_NEW_SERVER.md` | 远程 GPU 主机可能释放时的恢复说明。 | 与当前“远程随时释放”的工作方式匹配，应继续维护。 |
| `docs/从客观核验到消费者感知：直播电商虚假宣传识别的检索增强对比学习框架.md` | 论文/提案草稿。 | 方法叙事需要按最新实验修正：从“单模型碾压”改为“RACL relation geometry + CM/NLI guarded score + source-poor sufficiency decision”。 |
| `docs/PROJECT_FILE_MAP.md` | 本文件。 | 新增，用作项目地图。 |

## 3. 通用基础模块

| 文件 | 作用 | 当前进展 |
|---|---|---|
| `src/config.py` | 全局路径、模型名、LLM/VLM 网关、Stage A/B/C 超参、标签权重参数。 | 路径基于 `CLAIMARC_ROOT`，适合本机/远程迁移；包含评价泄漏属性过滤词。 |
| `src/run_pipeline.py` | 数据流水线编排器，按 A0→A1→A2→A3→B0→B1→B2B3→B4B5→C1→C2→C3→C4→C5→labels→final 顺序运行。 | 可端到端重跑，也可按阶段/品类/pilot 运行。 |
| `src/common/io_utils.py` | JSON/JSONL、评论 xls 读取、文本归一化、Jaccard。 | 数据读取和轻量文本匹配的基础工具。 |
| `src/common/llm.py` | Matpool OpenAI-compatible LLM/VLM 客户端，含缓存、重试、JSON 解析、图像 data URL 编码和并发。 | 所有 LLM/VLM 阶段和 argument augmentation 的关键依赖。 |
| `src/common/product_index.py` | 读取 `product_index.json`，把 clips/products 汇总为 `ProductBundle`。 | 负责 product_id 到 SRT、评论、图片、参数的统一索引。 |
| `src/common/srt.py` | SRT 解析、多个 clip 拼接、字符区间到时间戳反查。 | 支撑 B1 的 source-span grounding。 |
| `src/common/embedding.py` | BGE embedding 和层次聚类；失败时可回退 LLM 聚类。 | 用于 A0/A2 属性标准化。 |
| `src/common/embed_worker.py` | 独立 Python 嵌入 worker。 | 让主流水线 Python 和带 torch 的 Python 解耦。 |

## 4. Stage A：评论属性 schema 与抽取

| 文件 | 作用 | 当前进展 |
|---|---|---|
| `src/stage_a/a0_build_cas.py` | 从商品参数 key 构建品类 CAS，BGE 聚类后 LLM 裁决 canonical/aliases/value_type。 | 已形成按品类的标准属性表，是后续评论抽取和 pair 枚举的基础。 |
| `src/stage_a/a1_extract_aspects.py` | 在 CAS 约束下抽取评论 aspect；无法映射的写成 `FREE::<客观属性>`；过滤 personal。 | 已包含防止“性价比/满意/回购”等整体评价泄漏为属性的 prompt 约束。 |
| `src/stage_a/a2_aggregate_free.py` | 对 FREE aspect 做去重、聚类、LLM 合并/新增，生成 CAS+ 与 resolution 表。 | 解决评论中新增属性和参数 schema 不完全一致的问题。 |
| `src/stage_a/a3_resolve_labels.py` | 把 raw_aspects 中的 FREE id 重写到 CAS+ 标准 id，输出 resolved_aspects。 | 产物进入 Stage B 的 A_cmt 聚合。 |

## 5. Stage B：主播 claim 与评论对齐

| 文件 | 作用 | 当前进展 |
|---|---|---|
| `src/stage_b/b0_acmt.py` | 按 product 聚合评论侧命中的属性集合 `A_cmt(p)`；过滤笼统评价泄漏属性。 | 决定 claim 抽取和 pair 枚举的候选空间。 |
| `src/stage_b/b1_claim_extract.py` | 用 LangExtract/OpenAI-compatible 后端在 SRT 中抽取 source-grounded attribute claims。 | 保留 char interval 和时间戳反查，尽量防止主播 claim 幻觉。 |
| `src/stage_b/b2_b3_passage.py` | 枚举 `(product, attribute)` pair，并把同属性 claim 做时序去重和 passage 拼接。 | 输出 pair_skeleton，含 `has_claim_srt`、passage 和 segments。 |
| `src/stage_b/b4_b5_align.py` | LLM 判断评论是否直接回应主播该属性 claim，再聚合 pair_records。 | 弱监督标签的重要上游，决定评论证据是否 supportability-aligned。 |

## 6. Stage C：商品事实证据

| 文件 | 作用 | 当前进展 |
|---|---|---|
| `src/stage_c/c1_image_triage.py` | VLM 给详情图分流：参数表/证书/尺码图走 OCR，其余走 VLM。 | 控制视觉证据成本和后续 OCR/VLM 分工。 |
| `src/stage_c/c2_params.py` | 从商品参数中匹配每个属性的文本证据；alias 反查优先，LLM 兜底。 | 默认关闭 BGE 语义匹配，因为短中文属性相似度不可靠。 |
| `src/stage_c/c3_ocr.py` | 对 OCR 图跑 PaddleOCR，再由 LLM 按属性抽取原文片段。 | 对图文参数、证书、规格表提供证据。 |
| `src/stage_c/c4_vlm.py` | 对主图和代表图做多图 VLM 视觉证据抽取。 | 只输出视觉可直接观察到的客观证据。 |
| `src/stage_c/c5_fact_records.py` | 把 params/OCR/VLM 三源证据并列写入 fact_records，不做裁决。 | 保留证据覆盖度和 confidence 档位。 |

## 7. 标签与最终数据集

| 文件 | 作用 | 当前进展 |
|---|---|---|
| `src/labels/build_labels.py` | 根据 aligned 评论的正负、强度、显式 fact hit、疑似刷评等构造弱监督 `y` 和可信度 `c`。 | 标签是当前任务的主要噪声来源；`c` 已被训练和评估广泛使用。 |
| `src/final/join_split.py` | 合并 Stage B pair、Stage C fact、labels，按 room_id 分组切 train/val/test。 | 生成基础 `dataset.jsonl` 和 Table 1 统计。严格实验后来改用 grouped CV 复核。 |

## 8. 模型核心文件

| 文件 | 作用 | 当前进展 |
|---|---|---|
| `src/models/data.py` | CLAIMARC 双流 tokenization：claim flow 与 evidence flow；已加入 `[ARG_SUP]`、`[ARG_REF]`、`[ARG_GAP]`。 | argument augmentation 已进入 evidence 流；默认 args-first，带 `_evidence_policy="source_first"` 的数据会把参数/OCR/VLM 放到 arguments 前。现已支持 `params_only/ocr_only/vlm_only/args_only/no_args/...` source-policy tokenization，并把 `evidence_combo` 与 `confidence` 透传到 batch；`ClaimDataset` 新增 train-only `evidence_policy_mix` 和 auxiliary `evidence_consistency_mix`，可在 batch 中带 `e_view_ids/e_view_mask` 做多视图一致性训练。 |
| `src/models/model.py` | CLAIMARC 双流模型：共享 encoder、LoRA/topk/full 训练模式、TwoStreamFusion、LRC 分类头、retrieval embedding 头。 | 当前核心结构仍保留 RACL 机制；小数据下低容量 BGE-small LoRA(r=8) 更稳。 |
| `src/models/train.py` | 训练与固定 split 评估：ASL/BCE/Focal、置信加权 CE、属性分块 RACL、memory bank、p_cls/RKC 指标、embedding 导出。 | 已加入 `cl_c_min`、`cl_neg_c_min`、`encoder_name`、BGE teacher distillation、teacher-guided RACL、source-domain CE/CL weight scales、`cl_neg_filter` evidence-type hard negative、`cl_neg_bonus` soft ranking bonus、`--evidence_policy` 分源专家训练、`--evidence_policy_mix` train-only 多视图训练、`--view_consistency_mix` 的 auxiliary CE/logit/embed consistency、`--source_aux_*` evidence metadata 表示正则，以及 `--proto_aux_*` 训练期正/负 prototype 辅助 relation loss；`--proto_aux_mode` 支持默认 CE 与 margin gap。普通 BCE 蒸馏为负；保守分歧蒸馏提升 AUPRC；teacher-guided RACL 略提 Macro-F1 但仍不胜 fair BGE；激进 source-domain CL reweight、`medium_evtype_conf` hard-negative 过滤、soft bonus、简单 view dropout、view consistency、source auxiliary representation、prototype CE auxiliary 和 margin auxiliary 在 fs1 上均不足；source-policy/prototype 收益暂时来自测试时多 expert pooling、source guard 和 score-side calibration。 |
| `src/models/fusion_eval.py` | 离线融合和评估：p_cls、kNN、selectiveRKC、blend2、Platt、ARF、paired bootstrap。 | 已发现复杂融合容易过拟合；当前更适合作为诊断和 hybrid 组件。 |
| `src/models/retrieval_calibrator.py` | 检索校准/局部融合相关工具。 | 早期 v2 融合探索文件，当前不是主结论核心。 |
| `src/models/argument_aug.py` | 用 LLM 生成支持/反驳/证据缺口 argument，并写入 dataset args 版本。 | 已生成 `dataset_verify_faithful_args.jsonl`，且 prompt 不使用标签，避免直接泄漏。 |
| `src/models/make_evidence_policy_dataset.py` | 生成 evidence-ordering 数据变体。 | 已生成 `dataset_verify_faithful_args_srcfirst_a120.jsonl`：source-first，argument 每段最多 120 字符；新增 `--drop_args_without_source`，已生成 drop-src0args 诊断数据。 |

## 9. 模型评估与实验脚本

| 文件 | 作用 | 当前进展 |
|---|---|---|
| `src/models/baselines.py` | 冻结 BGE + LR 的基础基线，claim/evidence/concat 三种特征。 | 已修正 `evidence_text` 读取 arguments，并与 `_evidence_policy` 保持同顺序，保证 source-first 公平对照。 |
| `src/models/baselines_ft.py` | BERT/RoBERTa 等监督 fine-tuning 基线。 | 用于强监督 Transformer 对照。 |
| `src/models/baselines_extra.py` | 冻结 BGE+LR、kNN 等额外基线/embedding 特征。 | grouped CV 中 fair BGE+LR 的来源。 |
| `src/models/cv_eval.py` | grouped K-fold OOF 主评估，训练 CLAIMARC 和基线，汇总 paired bootstrap。 | 已加入 `--fold_seed`、`--evidence_policy`、`--evidence_policy_mix`、`--view_consistency_mix`、`--source_aux_*`、`--proto_aux_*`（含 CE/margin mode）、`--dump_oof` 与 `--n_boot`；OOF dump 已前置到 bootstrap 之前，GPU 主机释放时可优先保住结果。是当前最严格协议，也是 source-policy experts、evidence-view dropout、consistency smoke、source-aux smoke、prototype-aux smoke 和新增 fs3/fs4 validation 的统一入口。 |
| `src/models/normalize_cv_oof.py` | 把 `cv_eval` 输出的 `p__method/yhat__method/fold_id` 规范化为后处理脚本使用的 `method__p/method__yhat/fold/case`，可拼接多个 repeated-CV case。 | 新增于 2026-06-09；用于把 fs3/fs4 单独 CV OOF 接入 prototype verifier / decision feature 链路，并支持方法重命名如 `CLAIMARC_pcls=sourcefirst_cm_pcls_saved`。 |
| `src/models/cv_arg_expert_fusion.py` | argument 检索专家融合评估。 | 用于 seed=0 的 argument-ARF 方向探索。 |
| `src/models/cv_fusion_search.py` | 基于保存 OOF bundle 的 leakage-safe 融合搜索。 | 已在 seed=1 验证 no-args/args bundle 的 `pair_id` 对齐无误；结论是 rank-average 可显著提升 AUPRC，但 Macro-F1/wF1 未复现 seed=0 的全指标显著。 |
| `src/models/cv_main_hybrid_eval.py` | 便宜复跑主 hybrid 候选：rankavg/prob blend/validation selection。 | seed=1 表明 `rankavg(args_pcls,BGE)` AUPRC 更稳，但 Macro-F1 不稳。 |
| `src/models/cv_threshold_diagnose.py` | 诊断 val 阈值和 pooled oracle 阈值的差异。 | 证明 seed=1 的 Macro-F1 问题不只是阈值迁移。 |
| `src/models/cv_reranker_feature.py` | Cross-encoder reranker feature baseline：缓存 reranker logits，再按 grouped CV 做 direct/LR OOF。 | 新增于 2026-06-09；`BAAI/bge-reranker-v2-m3` fs1 显著弱于 BGE+LR，说明通用 relevance reranker 不能直接作为 CLAIMARC teacher。 |
| `src/models/cv_set_sufficiency_meta.py` | Fold-safe set-level sufficiency meta-head：在每个 repeated-CV case 内按 outer fold cross-fit 小 LR/HGB，特征为 BGE/current/adaptive score disagreement、source_count、evidence_combo、confidence 等。 | 新增于 2026-06-09；默认 LR-only。主结果 `set_suff_lr_no_cat` 为 0.4861/0.6280/0.5859，显著弱于 evidence-type adapter；说明普通 sufficiency classifier 会过拟合 source0/absent 假阳性，不继续作为主线。 |
| `src/models/cv_evidence_type_selector.py` | Fold-safe 受约束 evidence-type adapter selector：每个 case/fold 只在预定义 source/evidence masks 中选择 score mask 与 decision mask，再应用 held-out fold。 | 新增于 2026-06-09；balanced selector 为 0.5022/0.6412/0.6074，接近但低于固定 `src0_or_PO-medium score + PO-medium decision` 的 0.5052/0.6412/0.6084。机制验证显示 decision 端自然选择 `PO:medium`，score 端自由选择反而略漂。 |
| `src/models/cv_racl_prototype_verifier.py` | RACL embedding prototype verifier：读取已保存 fold bundle 中的 retrieval embedding `g`，每个 outer fold 用训练折正/负 class prototypes 生成 relation score；支持 global/attribute/source/evidence-type prototypes。 | 新增于 2026-06-09；不重训模型。旧 fs0/fs1/fs2 最佳 `rankavg_bge_cm_proto_source_bin` 为 0.5053/0.6391/0.5891，说明 prototype 有排序互补但不能承担 binary decision。fs3/fs4 独立验证中，prototype source-bin rank 在强 BGE 上仍提升 AP/AUROC。默认 `n_boot=0`，显著性用 `bootstrap_oof_methods.py` 定向比较；dump 已透传 `room_id/attribute_id` 供 group bootstrap。 |
| `src/models/diagnose_racl_proto_rankblend.py` | OOF-level prototype score calibration 诊断网格：按 case+fold rank blend evidence-type score 与 RACL prototype score，decision 保持 evidence-type。 | 新增于 2026-06-09；等权 `evtype_rankblend_proto50_decision_evtype` 为 0.5071/0.6430/0.6084；相对 BGE/旧 CM+BGE 三项显著，但相对 evidence-type adapter 的 AP/AUROC 未显著。CLI 已修复：显式 `--weight` 不再自动附带默认网格。 |
| `src/models/cv_racl_proto_evtype_protocol.py` | 固定协议版 RACL prototype calibration：不做 selector 或权重搜索，只输出 `cal50` calibrated-prototype 与 `raw25` raw-prototype 两个 score 分支；`--decision_method/--decision_label` 可把 evidence-type decision 或 conservative router decision 接到同一 score head 上。 | 新增于 2026-06-09；`evtype_proto_raw25_decision_evtype` 为 0.5070/0.6438/0.6084，是 ranking/screening 协议化候选。接入 conservative router decision 后，`evtype_proto_raw25_decision_router` 为 0.5084/0.6456/0.6109/0.6411，是当前 top-line 点估计；相对 BGE room-level p=0.0004/0.0316/0.0000，相对旧 router score 的 AP/AUROC room-level 正向但未显著。 |
| `src/models/cv_racl_proto_decision_feature.py` | Prototype decision feature：把 raw prototype rank 作为 source/evidence sufficiency guard，支持 fixed source0 rule、source0-only nested threshold selector、较宽 per-case/fold selector，以及可配置 `--decision_method/--score_method`。 | 新增于 2026-06-09；旧 fs0/fs1/fs2 fixed `source_count==0` rule 为 0.5070/0.6438/0.6142，相对 evidence-type adapter Macro-F1 sample p=0.0012、pair-level group p=0.0090。新增 fs3/fs4 BGE-base 验证中，`proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin` pooled 为 0.5009/0.6448/0.5981/0.6350，相对 BGE 的 Macro-F1 sample p=0.0012、room-level p=0.0064。五划分统一 BGE-base 复核为 0.5032/0.6414/0.5913/0.6244；相对 BGE 的 AP/Macro-F1 在 sample、pair-level 与 room-level bootstrap 均显著，AUROC group p 约 0.14/0.066 边界不足；decision edit 相对 score-only prototype 仍不显著。guarded-base 复核把 `decision_method` 切为 CM/NLI guarded，fixed source0 rule 达到 0.5029/0.6422/0.6053/0.6359；相对 BGE room-level p=0.0006/0.0382/0.0002，相对原 guarded Macro-F1 room-level p=0.0156，是当前 router 的 RACL base。独立 fs3/fs4 子集相对 BGE 的 sample/pair/room Macro p=0.0298/0.0334/0.0444；相对原 guarded 方向为正但未显著。 |
| `src/models/diagnose_guard_flips.py` | 对两个 OOF 方法的 binary decision flip 做诊断：按 case/source/evidence/category 汇总 veto/promote、修复 FP/FN、引入 FP/FN 和 Macro-F1 变化。 | 新增于 2026-06-09；fixed source0 guard 五划分翻转 161/8470 条、净正确 +25。保守 router 相对 fixed source0 guard 翻转 270/8470 条、净正确 +62，fs3/fs4 翻转 152/3388 条、净正确 +38。 |
| `src/models/cv_embedding_baseline.py` | 任意 SentenceTransformer frozen embedding + LR 的 grouped-CV baseline：一次编码 claim/evidence，按 fold_seed/case 输出 OOF；支持 `--claim_prompt_name/--evidence_prompt_name` 和显式 prompt。 | 新增于 2026-06-09；BGE-large repro 为 0.4816/0.6315/0.5845，Qwen3-Embedding-0.6B no-prompt 为 0.4845/0.6376/0.5900，query-prompt 为 0.4813/0.6351/0.5835。当前主方法相对 no-prompt Qwen3 的 Macro-F1 在 sample/pair/room bootstrap 均显著，AP 正向但 room-level 边界，AUROC 不显著；相对 query-prompt Qwen3 的 room-level AP/Macro 也显著。 |
| `src/models/cv_oof_disagreement_router.py` | Fold-safe OOF disagreement router：在每个 case/fold 内用其他 fold 选择一个小 Qwen3 disagreement switch，再应用到 held-out fold；只改 binary decision，不改输入 score。支持保守门槛 `--min_val_delta/--non_veto_min_val_delta/--max_val_flip_rate`。 | 新增于 2026-06-09；保守 selected Qwen3 router decision 为 0.5029/0.6422/0.6109/0.6411，相对 fixed RACL source0 guard 的 Macro-F1 sample/pair/room p=0.0026/0.0056/0.0112；fs3/fs4 子集为 0.6076 Macro，room p=0.0028。接上 raw25 RACL score 后为 0.5084/0.6456/0.6109/0.6411。no-selector fixed veto 为 0.5029/0.6422/0.6092/0.6383，用作确认性 robustness check。 |
| `scripts/build_guarded_proto_fs0_fs4_oof.py` | 构建五划分 guarded-score + prototype-feature OOF：fs0-fs2 读取 prototype/evidence-type protocol OOF，fs3/fs4 把 NLI guarded OOF 与 prototype decision OOF 按 `pair_id` 合并。 | 新增于 2026-06-09；产出 `oof_guarded_proto_fs0_fs4_room_20260609.npz`，供 `cv_racl_proto_decision_feature.py --decision_method <CM/NLI guarded>` 生成当前主方法候选。 |
| `scripts/subset_oof_by_case.py` | 按 OOF 的 `case` 数组切子集，保留所有按行对齐的数组。 | 新增于 2026-06-09；用于把五划分主 OOF 切出 fs3/fs4 独立验证子集。 |
| `scripts/merge_oof_methods.py` | 按 `(case, pair_id)` 把外部 OOF 方法的 `__p/__yhat` 合并进主 OOF。 | 新增于 2026-06-09；用于把 BGE-large repro、Qwen3 no-prompt、Qwen3 query-prompt embedding baseline 接入主方法 paired bootstrap。 |
| `src/models/llm_risk_baseline.py` | direct LLM 风险判别基线：缓存 `risk_score/evidence_state/rationale`，并按 grouped CV val-threshold 协议评估；支持 `broad/conservative` prompt 和 shard 分片。 | `Qwen-Flash conservative` 已全量跑完，单独显著弱于 BGE+LR；与 BGE rankavg 只有不显著 AP 小增且伤分类；粗粒度 `evidence_state` 不富集正类。适合作为审稿基线，不适合作为主 teacher。 |
| `src/models/cv_reliability_gate.py` | leakage-safe 可靠性门控诊断：每个 outer fold 只在 val carve 上选择 BGE-uncertainty/confidence-advantage switch 或小 LR gate，再应用到 test。 | 当前最有希望的新结构线。seed0/seed1/seed2 均确认 rank-average AUPRC 排序增益稳定显著；Macro-F1 点估计可超过 BGE，但只有 seed2 的一个门控候选显著，尚未复现。 |
| `src/models/cv_reliability_head.py` | outer-train reliability head：每个 outer fold 在 train split 拟合小 LR head，在 val carve 选超参与阈值，再测 held-out test；BGE 特征一次缓存。 | fs2 负结果：`reliability_head_macro` AP/AUROC 低于 BGE，Macro-F1 小增不显著；不如简单 rankavg。说明现有概率/元数据不足以支撑可训练二层头。 |
| `src/models/cv_source_condition_gate.py` | source-conditioned rankavg 诊断：按 source_count/source_len/confidence/BGE uncertainty 在 val carve 中选择 mask，再测 held-out test。 | 说明 source-rich mask 能提高 Macro-F1 点估计，但严格 OOF yhat bootstrap 后分类显著性不足；对 dual-head router 的 decision head 有指导价值。 |
| `src/models/cv_dual_head_router.py` | leakage-safe dual-head 诊断：ranking score head 与 binary decision head 解耦，AP/AUROC 用 score，Macro-F1/wF1 用 fold 内 val 学到的 OOF yhat；含 source/domain group threshold。 | fs0/fs2 可产生三指标显著候选；fs1 原始和 drop-src0args 仍未闭环。当前最清楚地定位了“排序头已成形、fs1 判定边界不足”的问题。 |
| `src/models/cv_nli_dual_guard.py` | Atomic NLI posterior + BGE rankmix + dual-head guard/headmix 评估；支持 `bgeedit_*`、`scoregroup_*`、`sourceveto_*`、`bgefallback_*`、`scorefallback_*`、`predef_lowabs_*`、`bgeadvfallback_*`、`bgerateguard_*`、`nlievidenceveto`、`compact_router_*`、`compact_router_nested_*`、confidence/srcbin/source-aware decision 候选，并输出 OOF yhat/score 供 bootstrap 和残差诊断。 | fs1 drop-src0args 已三指标显著闭环；fs1 的 `srcbin_conf_bgefallback_src0_src2_3_lowabs` 是更简单 strict 候选且机制贴近 fs0 lowabs fallback；`predef_lowabs_r25_scorefallback_srcconf_bgefallback` 已把 fs1 的 score-side/decision-side lowabs fallback 固定成少参数 strict 协议；fs2 由 `rankmix_nli25` score + `bgefallback_src0_src2_3` decision 三指标显著闭环；fs0 由 `scorefallback_bge025_src0_src2_3_lowabs` 三指标显著闭环。第一版 `compact_router_valselect`、`compact_router_nested_*`、`bgeadvfallback_*`、`bgerateguard_*` 和 `nlievidenceveto` 均为负/不足，说明不能靠小验证折、组级预测率规则或硬 posterior veto 自动复现 split-specific 成功；下一步应检验固定 source/confidence fallback 协议的 repeated-CV 稳定性。 |
| `src/models/cv_nli_predef_lowabs.py` | predef-only 评估器：只跑 NLI+BGE rankmix、scoreguard、score-side BGE fallback、decision-side BGE fallback、少数 `predef_lowabs` 协议，以及 fold-level `scorefallback self-threshold vs protected decision fallback` 二值开关；不加载 noargs/args RACL 临时目录。 | 用于规避 fs0 noargs 资产不匹配。最新加入 `--cm_tmp/--cm_seed`，读取 source-first CM p_cls，并输出 `rankavg_sourcefirst_cm_pcls_*`、`rankw_sourcefirst_cm*_nli*`、`rankw_sourcefirst_*_decision_switch_*`、source-rich confidence guard、`score_src0ormedium_cmreinforce025...src4pmedium_cmbgenli` adaptive 修复、`score_sportsgeneral_cm025_decision_sports_cm025` taxonomy-aware 诊断候选，以及 `score_src0orpomedium_cmreinforce025...pomedium_cmbgenli` evidence-type adapter。当前机制最干净的统一候选仍为 guarded CM/NLI rank fusion；pooled 下一版机制候选为 evidence-type adapter。 |
| `src/models/bootstrap_oof_methods.py` | 通用 OOF paired bootstrap 工具：读取一个或多个 `.npz` 的 `method__p` / `method__yhat` 或 source-policy 风格 `p__method` / `yhat__method`，输出 case-level 与 pooled AP/AUROC/Macro-F1/wF1 和配对 bootstrap。 | 新增于 CM p_cls + NLI weighted ranking 之后；现也用于 source-policy repeated-CV 汇总、hybrid OOF 显著性检查和 fs3/fs4 prototype decision validation。已加入 `--group_key`、`--skip_case`、`--only_group`，可按 `pair_id` 或 `room_id` 做 group bootstrap。不重训、不重新选择模型，适合作为 leakage-safe 显著性汇总工具。 |
| `src/models/diagnose_fallback_mechanisms.py` | 从已保存 OOF `.npz` 做 fallback 残差机制诊断：按 fold/source/confidence/category/BGE 不确定性统计 BGE 错误被修复与 BGE 正确被破坏的逐样本翻转。 | 产出 `fallback_mechanism_diagnosis_20260608.json`。诊断显示 fs0 收益集中在 BGE 不确定/lowabs 假阳性，fs1 的固定协议实质是保护 lowabs/source0 并让 RACL/NLI 接管 source-rich 区域，fs2 是正类召回边界修复。 |
| `src/models/diagnose_protected_hybrid_oof.py` | OOF-level protected hybrid 筛查：复用已保存的 fold-safe OOF score/yhat，合成少量 `protected BGE regions + RACL/NLI regions` 规则并做 paired bootstrap。 | 探索性脚本，不重训、不作为论文主张。当前最接近统一的 `rank25_bge025_lowabs + protect_lowabs_scoreguard_srcbin_conf` 在 fs1/fs2 严格，但 fs0 新缓存 AUROC/Macro-F1 不严格。 |
| `src/models/diagnose_taxonomy_adapter_oof.py` | OOF-level validation-safe taxonomy adapter 诊断：读取 adaptive OOF `.npz` 与 CV JSON `fold_meta`，在每个 outer fold 内按 validation 指标分别选择 score head 和 decision head，再做 pooled paired bootstrap。 | 产出 `taxonomy_adapter_oof_screen_20260608.json`。结论为负：固定 taxonomy-aware combo 是 pooled 显著诊断上界，但 validation-safe selector 未显著胜 current guarded，不能作为最终主方法。 |
| `src/models/diagnose_evidence_type_adapter_oof.py` | OOF-level evidence-type adapter 筛查：把原始 dataset 的 `evidence_params/evidence_ocr/evidence_vlm` 按 `pair_id` 接到 adaptive OOF，合成 `source_count==0` 与 `PO:medium` 上的 adaptive score/decision 窄修复。 | 产出 `evidence_type_adapter_oof_screen_20260608.json` 和 `oof_evidence_type_adapter_screen_20260608.npz`；已派生 `oof_evidence_type_adapter_screen_room_20260609.npz` 做 room bootstrap。三划分 pooled 为 0.5052/0.6412/0.6084，room-level 相对 BGE p=0.0006/0.0728/0.0000，相对 current guarded p=0.1674/0.3916/0.0732。新增 fs3/fs4 evaluator OOF 后，五划分 evidence-type 为 0.5039/0.6424/0.6005/0.6277，相对 BGE room-level p=0.0008/0.0500/0.0004，但相对旧 guarded 不显著且 Macro/wF1 略低。 |
| `src/models/diagnose_common_oof_methods.py` | OOF-level common-method audit：枚举多个 `.npz` 中共同存在的所有 `method__p` / `method__yhat`，按 pooled 与单 split 指标排序，并对主要候选做 paired bootstrap。 | 产出 `common_oof_method_sweep_20260608.json`。fs0/fs1/fs2 adaptive quick 共有 227 个方法；未发现漏网的非 taxonomy 现成候选。最高 pooled AP 为 adaptive fixed 0.5049/0.6413/0.6071，最高 pooled AUROC/Macro-F1 为 taxonomy-aware fixed 0.5031/0.6421/0.6086；fs1 局部高分方法会明显牺牲 pooled AP/AUROC。 |
| `src/models/diagnose_relation_oof_adapter.py` | OOF-level relation adapter 诊断：在 evidence-type OOF 上用 `pair_id` 分组 cross-fit 小型 LR/HGB 二层模型，测试现有概率、source/confidence/evidence_combo 元数据能否学习“何时信任 RACL/NLI”。 | 产出 `relation_oof_adapter_screen_20260608.json` 和 `oof_relation_adapter_screen_20260608.npz`。结论为负：无 category 的 LR/HGB 只有 0.4782/0.6294/0.5890 和 0.4786/0.6242/0.5789；带 category/taxonomy 的诊断上界也只有 0.4950/0.6400/0.6001，弱于 evidence-type adapter。不要继续普通二层 stacker。 |
| `src/models/diagnose_nli_source_pooling_oof.py` | OOF-level NLI source-pooling 微校准诊断：把 `cache_nli_srcargs_a120.npz` 的 source-block posterior 特征按 `pair_id` 对齐到 evidence-type OOF，只做 score-side 小权重校准，decision 沿用基底。 | 产出 `nli_source_pooling_oof_screen_20260608.json`、`nli_source_pooling_oof_top1_3k_20260608.json` 和 `oof_nli_source_pooling_screen_20260608.npz`。top pooled screen 为 `evtype + 5% arg_ref neutral-rate35`，0.5093/0.6414/0.6084，但 fold-level fs0/fs1/fs2 OOF pooled 后只有 0.5034/0.6404/0.6084，低于 evidence-type 0.5052/0.6412/0.6084；确认为负结果。 |
| `src/models/diagnose_evtype_residuals.py` | OOF-level evidence-type 残差地图：按 case/source/confidence/evidence_combo/category 统计 FP/FN 与相对 BGE、CM+BGE、current 的 fixed/broken。 | 产出 `evtype_residual_diagnosis_20260608.json`。显示收益集中在 `PO:medium`，残余不稳区域主要为 `src2_3:medium`、`POV:high`、`O:low` 等，但没有直接给出可安全固化的统一规则。 |
| `src/models/diagnose_evsuff_oof_rules.py` | OOF-level evidence-sufficiency fallback 小规则筛查：在 evidence-type 上测试少数 source/evidence mask 回退到 BGE、CM+BGE 或 current。 | 第一版 `evsuff_oof_rule_screen_20260608.json` 使用 mask-local rank，有实现缺陷，不引用；修正版 `evsuff_oof_rule_screen_rawblend_20260608.json` 显示 `O:low -> BGE` 伤 Macro-F1，`src2_3:medium -> BGE` 的 Macro 小涨不显著。规则层补丁基本耗尽。 |
| `src/models/diagnose_source_policy_pooling.py` | Source-policy / multi-instance evidence pooling 诊断：读取不同 `evidence_policy` 训练出的 CLAIMARC fold bundle，在每个 outer fold 的 validation split 上选阈值，固定 score pooling，支持 BGE/noargs decision guard 与 OOF dump。 | fs0/fs1/fs2 已完成。三划分 pooled repeated-CV 中，`rankavg_all_score_bge_lr_src0_neg_guard` 为 0.4978/0.6345/0.5930，vs BGE p=0.0000/0.0002/0.0062；`mean_all_score_bge_lr_src0_neg_guard` 为 0.4918/0.6301/0.5955，vs BGE p=0.0000/0.0042/0.0004，vs sourcefirst p=0.0016/0.0000/0.0502。新增 fs3 独立复核中，`rankavg_all` 为 0.4834/0.6299/0.5771/0.5954，vs BGE AP/AUROC 显著但 Macro 不显著；`source_masked_mean` Macro 0.5807 但 vs BGE 不显著。结论：source-policy 稳定贡献主要是排序/可靠性加权，点估计仍弱于 evidence-type/prototype 主线，当前定位为结构性消融。 |
| `src/models/diagnose_sourcefirst_failures.py` | source-first rankavg 失败分组诊断。 | fs1 发现 `source_count=0` / `confidence=absent` 样本上 args 分支偏高，是 drop-src0args 与 source-conditioned router 的直接动机。 |
| `src/models/cv_oof_blend.py` | OOF 层融合探索。 | 早期/辅助融合脚本。 |
| `src/models/ensemble.py` | 多模型结果集成。 | 早期固定划分 ensemble 结果来源之一。 |
| `src/models/analysis.py` | 数据/结果分析辅助脚本。 | 早期结果审计使用。 |
| `src/models/compile_results.py` | 汇总实验结果到表格/JSON。 | 早期结果整理工具。 |
| `src/models/crossdomain.py` | 跨域少样本/留一品类早期评估。 | 结果显示 RKC 零样本略有价值，但增量 support 不稳。 |
| `src/models/crossdomain_v2.py` | 跨域评估改进版。 | 当前记录中有 OOM 未补齐项。 |
| `src/models/run_all.py` | 批量运行早期主模型/基线实验。 | 历史实验入口。 |
| `src/models/run_arch.py` | 架构搜索入口。 | 支撑 LoRA 秩、骨干、对比强度等搜索结论。 |
| `src/models/run_cap.py` | capped/容量控制实验入口。 | 早期容量控制探索。 |
| `src/models/run_experiments.py` | 早期实验批跑脚本。 | 历史入口。 |
| `src/models/run_final.py` | 固定划分 final comparison 批跑。 | 产生早期 `all_results_final.jsonl`。 |
| `src/models/run_tune.py` | 参数调优批跑。 | 早期 tune 结果来源。 |
| `src/models/run_v2.py` | CLAIMARC-v2 主运行脚本。 | 早期 v2 结果来源。 |
| `src/models/run_verifytest.py` | verify/test 数据上的实验入口。 | 生成 `dataset_verify*` 后的过渡实验。 |
| `src/models/verify_subset.py` | 抽样/核验子集构造辅助。 | 用于数据清理和人工/LLM 核验前处理。 |
| `src/models/verify_improve.py` | 数据核验改进脚本。 | 生成更干净的 verify 数据版本。 |
| `src/models/relabel_verify.py` | 对 verify 数据重新标注/修正。 | 生成 `dataset_verify_faithful.jsonl` 的相关清理链路。 |

## 10. 运行脚本

| 文件 | 作用 | 当前进展 |
|---|---|---|
| `scripts/setup_server.sh` | 远程主机环境初始化。 | 新 GPU 主机恢复时使用。 |
| `scripts/sync_to_server.sh` | 本机到远程同步。 | 需要继续排除 `._*`、大模型和原始大数据。 |
| `scripts/sync_back.sh` | 远程结果拉回本机，可循环执行。 | 对随时释放的 GPU 主机很关键。 |
| `scripts/run_food_pipeline.sh` | food pilot 数据流水线。 | 早期小样端到端验证。 |
| `scripts/run_food_bge.sh` | food 子集 BGE 实验。 | 早期基线/烟测。 |
| `scripts/run_food_030_cov.sh` | A2 阈值/coverage 相关 food 实验。 | 用于确定 A2 聚类阈值。 |
| `scripts/sweep_a2.sh` | A2 聚类阈值扫描。 | 支撑 `CLAIMARC_A2_DISTANCE=0.30` 选择。 |
| `scripts/run_cleancl_sweep.sh` | clean-RACL 第一轮 sweep。 | 找到 `small_e3_c10` 方向。 |
| `scripts/run_cleancl2_sweep.sh` | clean-RACL 第二轮 sweep。 | 排序型变体有提升但分类不如主配置。 |
| `scripts/run_cleancl_post.sh` | clean-RACL 后处理/评估。 | 生成 cleancl 固定划分和 CV 结果。 |
| `scripts/run_args_small_post.sh` | argument augmentation 后处理/评估。 | 生成 args 小模型与 hybrid 结果。 |
| `scripts/run_fs1_srcfirst_drop_srcdom_cv.sh` | fs1 drop-src0args 上的训练期 source-domain CL reweight 实验。 | 负结果：`source0_cl_scale=0.20, source_rich_cl_scale=1.50` 后 PCLS AP 0.4806 / AUROC 0.6206 / Macro-F1 0.5792，弱于不加权 drop 变体。 |
| `scripts/run_source_policy_experts_fs1.sh` | 训练分源 evidence-policy experts 的恢复脚本；可用环境变量 `FS`、`POLICIES`、`N_BOOT` 切换 split、专家集合与是否内置 bootstrap。 | fs0/fs1/fs2 source-policy experts 与 pooling 已完成；fs3 也已完成 `no_args ocr_only params_only` 与 `n_boot=0` pooling/targeted bootstrap。默认 `N_BOOT=0`，优先落 OOF，适合随时释放的 GPU 主机。 |

## 11. 关键数据与结果文件

| 文件/目录 | 作用 | 当前进展 |
|---|---|---|
| `data/final/dataset.jsonl` | 初始最终数据集。 | 由完整流水线生成，后续又产生 verify/faithful 清理版本。 |
| `data/final/dataset_verify.jsonl` | verify 版数据。 | 过渡版本。 |
| `data/final/dataset_verify_faithful.jsonl` | 去噪 faithful 版主数据。 | 当前 no-args 主实验数据。 |
| `data/final/dataset_verify_faithful_args.jsonl` | 加入 LLM arguments 的 faithful 数据。 | 当前 argument-aware 实验主数据。 |
| `data/final/dataset_verify_faithful_args_srcfirst_a120.jsonl` | source-first argument-aware 数据变体。 | 商品参数/OCR/VLM 前置，arguments 后置且每段最多 120 字符；用于验证证据源顺序是否缓解 argument 噪声。 |
| `data/final/dataset_verify_faithful_args_srcfirst_a120_drop_src0args.jsonl` | source-first 的 fs1 诊断变体。 | 对 677 条无真实来源证据样本清空 argument/risk_cues；提升 fs1 排序头 AP/AUROC，但 Macro-F1 仍不显著。 |
| `data/final/v2/` | 早期 CLAIMARC-v2 架构搜索、消融、跨域和固定划分结果。 | 证明早期设计有一定方向性，但严格 CV 推翻了部分强结论。 |
| `data/final/cleancl/` | clean-RACL、argument CV、hybrid、threshold diagnosis、distillation 和 teacher-guided RACL 的核心结果。 | 当前最重要的结果目录；`cv_fusion_search_full_fs1_s0_2k.json` 记录了 seed=1 全融合搜索，排序显著、分类未闭环。 |
| `data/final/cleancl/llm_qwen_flash_args_conservative_full*.jsonl/json` | direct LLM 判别预测与评估结果。 | Qwen-Flash conservative 全量 1,694 条；作为负结果和后续 evidence-state 特征候选保留。 |
| `data/final/cleancl/cv_reliability_gate_fs*_s0.json` | reliability gate 的 seed0/seed1/seed2 评估结果。 | `rankavg(args/no-args p_cls, BGE+LR)` 的 AUPRC 提升已跨三个 split seed 稳定显著；分类 Macro-F1 仍未稳定闭环。 |
| `data/final/cleancl/cv_reliability_head_fs2_s0.json` | outer-train reliability head 的 fs2 诊断结果。 | 负结果：训练式 LR head 没有超过 rankavg，后续不要只在现有概率上堆二层学习器。 |
| `data/final/cleancl/cv_args_srcfirst_a120_small_e3_c10_fs*_s0.json` / `cv_reliability_gate_srcfirst_a120_fs*_s0.json` | source-first 单模型与融合结果。 | `rankavg(sourcefirst_args_pcls, sourcefirst_BGE)` 在 fs0/fs2 三指标显著，在 fs1 只复现 AUPRC；是当前最有机制解释力的新线索。 |
| `data/final/cleancl/cv_dual_head_router_srcfirst_a120_fs*_s0.json` | source-first dual-head router 筛查结果。 | `n_boot=1000`：fs0/fs2 有三指标显著候选，fs1 无；用于筛选下一轮结构，不作为最终主表。 |
| `data/final/cleancl/cv_args_srcfirst_a120_drop_src0args_small_e3_c10_fs1_s0.json` / `cv_reliability_gate_srcfirst_drop_src0args_fs1_s0.json` / `cv_dual_head_router_srcfirst_drop_src0args_fs1_s0.json` | drop-src0args 的 fs1 结果链。 | AP/AUROC 可显著超过 BGE（best score head AP 0.4950, AUROC 0.6392），但严格 Macro-F1 p=0.2165；继续说明 fs1 分类边界仍未解决。 |
| `data/final/cleancl/cv_args_srcfirst_a120_drop_src0args_srcdom_s0cl02_rcl15_fs1_s0.json` | source-domain CL reweight 的 fs1 CV 结果。 | 负结果：PCLS AP 0.4806 / AUROC 0.6206 / Macro-F1 0.5792；不要优先沿该超参继续扩展。 |
| `data/final/cleancl/cv_args_srcfirst_a120_drop_src0args_evhn_medium_fs1_s0.json` / `cv_tmp_args_srcfirst_a120_drop_src0args_evhn_medium_fs1_s0/` | 训练期 evidence-type hard-negative 过滤试验：`--cl_neg_filter medium_evtype_conf`。 | 负/不足结果：PCLS AP 0.4856 / AUROC 0.6173 / Macro-F1 0.5899；较原始 drop PCLS 只小幅改善 AP/Macro-F1、伤 AUROC，且仍弱于 BGE Macro-F1 0.5928。 |
| `data/final/cleancl/cv_args_srcfirst_a120_drop_src0args_evhn_soft005_fs1_s0.json` / `cv_tmp_args_srcfirst_a120_drop_src0args_evhn_soft005_fs1_s0/` | 训练期 evidence-type hard-negative soft bonus：`--cl_neg_bonus 0.05 --cl_neg_bonus_filter medium_evtype_conf`。 | 负/不足结果：PCLS AP 0.4849 / AUROC 0.6193 / Macro-F1 0.5861，对 BGE 不显著且不如 evidence-type OOF adapter。结论：evidence-type 不宜继续作为训练期 negative sampling 主线，下一步转向 score/decision adapter 或 auxiliary relation score。 |
| `data/final/cleancl/cv_guarded_group_router_drop_src0args_fs1_s0_v2.json` / `cv_nli_dual_guard_srcargs_drop_fs1_s0_weights_2k.json` / `cv_nli_dual_guard_srcargs_drop_fs1_s0_groupscoreedit_2k.json` / `cv_nli_dual_guard_srcargs_drop_fs1_s0_multigroup_top10_5k.json` / `cv_nli_dual_guard_srcargs_drop_fs1_s0_headmix_top6_5k.json` / `cv_nli_dual_guard_srcargs_drop_fs1_s0_scorefallback_top8_5k.json` / `cv_nli_dual_guard_srcargs_drop_fs1_s0_nlievidenceveto_top6_5k.json` / `oof_nli_dual_guard_srcargs_drop_fs1_s0_nlievidenceveto.npz` / `cv_nli_dual_guard_srcargs_drop_fs1_s0_predef_lowabs_top6_5k.json` / `oof_nli_dual_guard_srcargs_drop_fs1_s0_predef_lowabs.npz` | fs1 hard-split 的 guarded decision、atomic NLI posterior ranking、grouped score edit、score/decision cross-head、compact headmix、scorefallback/bgefallback/nlievidenceveto/predef_lowabs 结果。 | 当前 fs1 最高点估计 strict 候选仍是 `rankmix_nli25_hgb_bge` score + confidence headmix decision，AP 0.5008 / AUROC 0.6425 / Macro-F1 0.6117，5k p=0.0016/0.0082/0.0198；`rankmix_nli50_scorefallback_bge025_src0_src2_3_lowabs` 点估计 AP 0.5045 / AUROC 0.6433 / Macro-F1 0.6123 但 AUROC/Macro-F1 不严格；`predef_lowabs_r25_scorefallback_srcconf_bgefallback` 三项显著且规则最接近可预注册协议，AP 0.4940 / AUROC 0.6424 / Macro-F1 0.6101，p=0.0004/0.0030/0.0372。 |
| `data/final/cleancl/cv_nli_dual_guard_srcargs_drop_fs0_s0_headmix_top6_5k.json` / `cv_nli_dual_guard_srcargs_drop_fs0_s0_scorefallback_quick_top8_5k.json` / `oof_nli_dual_guard_srcargs_drop_fs0_s0_scorefallback_quick5k.npz` / `cv_nli_dual_guard_srcargs_drop_fs0_s0_compact_router_nohead_quick.json` / `cv_nli_dual_guard_srcargs_drop_fs2_s0_headmix_top6_5k.json` / `cv_nli_dual_guard_srcargs_drop_fs2_s0_scoregroup_top6_5k.json` / `cv_nli_dual_guard_srcargs_drop_fs2_s0_bgefallback_top8_5k.json` / `oof_nli_dual_guard_srcargs_drop_fs2_s0_bgefallback.npz` / `cv_nli_dual_guard_srcargs_drop_fs0_s0_scoregroup_quick.json` / `cv_nli_dual_guard_srcargs_drop_fs2_s0_sourceveto_quick.json` / `cv_nli_dual_guard_srcargs_drop_fs*_s0_bgeadv_quick.json` / `cv_nli_dual_guard_srcargs_drop_fs1_s0_nested_quick.json` / `cv_nli_dual_guard_srcargs_drop_fs1_s0_rateguard_quick.json` / `cv_nli_dual_guard_srcargs_drop_fs1_s0_nlievidenceveto_quick.json` / `cv_nli_dual_guard_srcargs_drop_fs1_s0_predef_lowabs_quick.json` / `cv_nli_dual_guard_srcargs_drop_fs2_s0_predef_lowabs_newcache_quick.json` / `cv_nli_predef_lowabs_srcargs_drop_fs0_s0_newcache_top20_5k.json` / `oof_nli_predef_lowabs_srcargs_drop_fs0_s0_newcache_5k.npz` / `cv_nli_predef_lowabs_srcargs_drop_fs0_s0_valselect_newcache_quick.json` / `cv_nli_predef_lowabs_srcargs_drop_fs2_s0_newcache_quick.json` | drop-src0args 的 fs0/fs1/fs2 NLI dual-guard 与 predef-only 复核/诊断。 | fs0 旧缓存 `rankmix_nli25_scorefallback_bge025_src0_src2_3_lowabs` 为 AP 0.4915 / AUROC 0.6309 / Macro-F1 0.5959，三项显著 p=0.0150/0.0458/0.0202；fs0 新缓存 predef-only `rankmix_nli25_scorefallback_bge100_src0` 为 AP 0.4859 / AUROC 0.6338 / Macro-F1 0.5973，三项显著 p=0.0094/0.0490/0.0176；fs2 `rankmix_nli25 + scoreguard_clip_drop20_min30_confidence_bgefallback_src0_src2_3` 为 AP 0.5128 / AUROC 0.6405 / Macro-F1 0.6040，三项显著 p=0.0126/0.0270/0.0028；`compact_router_valselect`/`nested`、`bgeadvfallback_*`、`bgerateguard_*`、`nlievidenceveto`、`predef_lowabs_valselect_*` 均为负/不足；fs2 新缓存 predef 只涨分类、不保排序，不扩 5k。 |
| `data/final/cleancl/fallback_mechanism_diagnosis_20260608.json` | fs0/fs1/fs2 strict fallback 候选的 OOF 机制诊断。 | 核心结论：统一方法应写成 `protected BGE regions + RACL/NLI regions`，而不是继续在小 validation carve 上自动选头；fs0 需要 BGE 不确定/lowabs 假阳性校正，fs1 需要保护 lowabs/source0 后让 source-rich 区域用 NLI+BGE，fs2 需要正类召回边界修复并约束新增 FP。 |
| `data/final/cleancl/protected_hybrid_oof_screen_20260608.json` / `protected_hybrid_forced_bootstrap_20260608.json` | protected hybrid 的 OOF-level 候选筛查与 5k 强制 bootstrap。 | 点估计支持统一方向，但未严格闭环：`rank25_bge025_lowabs + protect_lowabs_scoreguard_srcbin_conf` 在 fs1/fs2 strict，fs0 旧缓存 Macro-F1 p=0.0566，fs0 新缓存 AUROC/Macro-F1 p=0.0800/0.1514。暂不进入正式主方法。 |
| `data/final/cleancl/protected_hybrid_fs0_newcache_failure_20260608.json` / `scorefallback_selfthr_forced_bootstrap_20260608.json` | protected hybrid 失败机制补充诊断。 | fs0 新缓存统一候选的失败来自 medium/source-rich 组新增 FP；fs0 strict 候选依赖 scorefallback 自带阈值，fs1 则需要 decision fallback。下一步若统一，只能做极小二值开关，而不是把一个 decision head 横套所有 split。 |
| `data/final/cleancl/cv_nli_predef_lowabs_srcargs_drop_fs0_s0_switch_relaxed_quick.json` / `cv_nli_predef_lowabs_srcargs_drop_fs1_s0_switch_relaxed_quick.json` | fold-level 二值 switch quick 复核。 | `fp02_gain008` 变体在 fs0 等同 `scorefallback_bge100_src0`，Macro-F1 0.5948；在 fs1 提到 Macro-F1 0.6063，但仍低于固定 `predef_lowabs_r25_scorefallback_srcconf_bgefallback` 0.6089。机制可用但不足，暂不扩 5k。 |
| `data/final/cleancl/cv_nli_predef_lowabs_srcargs_drop_fs1_s0_switch_reverse_quick.json` / `cv_nli_predef_lowabs_srcargs_drop_fs0_s0_switch_reverse_top20_5k.json` / `cv_nli_predef_lowabs_srcargs_drop_fs0_s0_decoupled_top12_5k.json` / `oof_nli_predef_lowabs_srcargs_drop_fs*_switch_reverse*.npz` / `oof_nli_predef_lowabs_srcargs_drop_fs0_s0_decoupled_5k.npz` | reverse switch 与 decoupled score calibration 结果。 | fs1 reverse quick 复现 fixed protected；fs0 reverse 5k 为 AP 0.4900 / AUROC 0.6265 / Macro-F1 0.5959，p=0.0030/0.1880/0.0180；decoupled score 最高 AUROC 0.6286，最接近严格三项的 AUROC p=0.0566，仍不作为最终统一方法。 |
| `data/final/cleancl/cv_nli_predef_lowabs_srcargs_drop_fs*_s0_nondropbge_cmpcls_quick.json/.npz` / `...cmpcls_decision_quick.json/.npz` / `...cmpcls_weighted_quick.json/.npz` / `...cmpcls_adaptive_quick.json/.npz` / `oof_bootstrap_cmpcls_decoupled_20260608.json` / `oof_bootstrap_cmpcls_weighted_20260608.json` / `oof_bootstrap_cmpcls_weighted_switch_20260608.json` / `oof_bootstrap_cmpcls_weighted_guard_20260608.json` / `oof_bootstrap_cmpcls_adaptive_quick_20260608.json` / `taxonomy_adapter_oof_screen_20260608.json` / `evidence_type_adapter_oof_screen_20260608.json` / `oof_evidence_type_adapter_screen_20260608.npz` / `oof_evidence_type_adapter_screen_room_20260609.npz` / `evidence_type_adapter_room_bootstrap_group_5k_20260609.json` / `common_oof_method_sweep_20260608.json` / `relation_oof_adapter_screen_20260608.json` / `oof_relation_adapter_screen_20260608.npz` / `nli_source_pooling_oof_screen_20260608.json` / `nli_source_pooling_oof_top1_3k_20260608.json` / `oof_nli_source_pooling_screen_20260608.npz` / `cv_nli_predef_lowabs_srcargs_drop_fs*_s0_nondropbge_cmpcls_insuff_smoke.json/.npz` / `oof_bootstrap_cmpcls_insuff_smoke_20260608.json` / `evtype_residual_diagnosis_20260608.json` / `evsuff_oof_rule_screen_20260608.json` / `evsuff_oof_rule_screen_rawblend_20260608.json` / `cv_nli_predef_lowabs_srcargs_drop_fs1_s0_nondropbge_cmpcls_evtype_smoke.json/.npz` | source-first CM p_cls + BGE/NLI evidence rank 的三划分 OOF 结果、pooled bootstrap、taxonomy/evidence-type adapter 诊断、room-level group bootstrap、common-method audit、relation adapter、NLI source-pooling 与 evidence-sufficiency fallback 诊断，以及 evaluator smoke。 | 机制最干净的统一候选：`rankw_sourcefirst_cm040_nli060_decision_switch_sf025lowabs_lowabs_srcconf_fp05_gain008_srcge2_lowmedium_cmbgeprotect` pooled AP 0.5017 / AUROC 0.6407 / Macro-F1 0.6059；相对 BGE room-level p(AP/AUROC/Macro)=0.0012/0.0690/0.0000。adaptive 候选 pooled 0.5049 / 0.6413 / 0.6071。taxonomy-aware fixed combo pooled 0.5031 / 0.6421 / 0.6086，但 validation-safe taxonomy selectors 未显著胜 current guarded。evidence-type adapter pooled 0.5052 / 0.6412 / 0.6084，相对 BGE room-level p=0.0006/0.0728/0.0000，相对 current guarded room-level p=0.1674/0.3916/0.0732；是更可辩护的下一版 pooled 主线候选，但不能声称显著胜 current guarded。common-method audit 确认 227 个已保存共同方法中没有漏网的非 taxonomy 现成候选；relation adapter、NLI source-pooling 和 evidence-sufficiency fallback 规则均为负或不足；fs1 AUROC/Macro-F1 仍未单划分显著。 |
| `data/final/cleancl/cv_sourcepolicy_*_small_e3_c10_fs*_s0.json` / `oof_sourcepolicy_*_small_e3_c10_fs*_s0.npz` / `cv_tmp_sourcepolicy_*_small_e3_c10_fs*_s0/` / `source_policy_pooling_*fs*.json` / `oof_source_policy_pooling_*fs*.npz` / `source_policy_pooling_guard*_bootstrap*.json` | 分源 evidence-policy experts 与固定 multi-instance pooling / BGE decision guard 结果。 | fs1：OCR-only 0.4861/0.6173/0.5816，params-only 0.4923/0.6263/0.5804；rankavg + source0 BGE-negative guard 为 0.4934/0.6321/0.5995，相对 BGE 三项显著。fs2：pooling 后 `rankavg_all_score_bge_lr_src0_guard` 为 0.5176/0.6386/0.5926，AP/AUROC/Macro 显著。fs0：`rankavg_all` 为 0.4866/0.6324/0.5894，排序显著、Macro 边界。fs3：`rankavg_all` 为 0.4834/0.6299/0.5771/0.5954，AP/AUROC vs BGE 显著但 Macro 不显著；`source_masked_mean` Macro 0.5807 但 vs BGE 不显著。结论：source-policy 是可靠的排序消融，不作为最终主方法。 |
| `data/final/cleancl/cv_evidence_set_head_*_quick.json` / `cv_nli_evidence_head_*_quick.json` | evidence-set similarity 与 tiny-NLI posterior 的筛选诊断。 | BGE 相似度集合统计为负；tiny NLI posterior 单独弱，但与 BGE rankmix 后提供显著排序互补。 |
| `data/final/cleancl2/` | clean-RACL 第二轮 sweep 结果。 | 排序提升但分类不足，暂不作为主方法。 |
| `data/final/*.log` | 各批实验 stdout/stderr 日志。 | 用于追溯每次运行配置和错误。 |

## 12. 下一步建议

1. **主线结构**：普通 BGE teacher BCE 蒸馏已验证为负结果；conservative disagreement distillation 对 AUPRC 有边界收益但 Macro-F1 仍不稳；teacher-guided RACL 对 Macro-F1 有小幅帮助但仍不胜 fair BGE；阈值/先验校准和 `agree_pos` 均不能单独解决。当前 top-line 是 `RACL raw25 score + fixed RACL source0 guard + conservative Qwen3 disagreement router decision`：score head 固定混入 raw RACL prototype relation rank，decision head 先用 RACL source0 sufficiency guard，再用 fold-safe Qwen3 disagreement router 处理少量争议样本，五划分为 0.5084 / 0.6456 / 0.6109 / 0.6411。source-policy multi-instance pooling（params/OCR/noargs/sourcefirst experts）+ BGE guard 在 fs0-fs2 上显著胜 BGE/noargs/sourcefirst 的排序指标，但 fs3 只复现 AP/AUROC、未复现 Macro-F1，因此定位为结构性消融；简单 view dropout、view consistency、source auxiliary representation、fold-safe set-sufficiency LR head 和第一版 prototype CE auxiliary 均为负或不足。协议化 RACL prototype calibration、source0-only prototype decision feature、five-split fixed source0 guard、conservative Qwen3 router 和 modern embedding baselines 共同构成当前主线的消融链条。
2. **评估协议**：所有核心候选至少在 `fold_seed=0/1/2` grouped CV 上报告；只把 repeated split 都稳定的结果写进主结论。
3. **基线补齐**：Qwen-Flash zero-shot direct LLM 与 BGE reranker v2-m3 cross-encoder 已补且均显著弱于 BGE+LR；如论文需要更强大模型时代对照，再补 DeepSeek/Qwen 大模型或 few-shot hard-case adjudication。
4. **论文叙事**：把 CLAIMARC 定位为“消费者感知标签下的检索增强对比表征 + evidence sufficiency guard + disagreement adjudication”。可以主张显著提升强文本/现代 embedding 基线的 Macro-F1，并把 AP/AUROC 的新点估计归因于固定 raw25 RACL prototype score head；不要声称 Qwen3 router 改善 ranking，也不要声称 raw25 在 room-level 已显著击败旧 router score。
