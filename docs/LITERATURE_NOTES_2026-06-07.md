# 2026-06-07 文献扫描与 CLAIMARC 下一轮方法启发

## 0. 2026-06-08 追加扫描：歧义、证据质量与假阳性约束

### 2026-06-09 追加复核：relation prototype 与 source-sufficiency protocol

这轮结合新增 `fold_seed=3/4` 结果复核了最新 fact verification / RAG sufficiency 文献。结论更明确：CLAIMARC 不应再追求“端到端分类头单独碾压 BGE”，而应把 RACL embedding 写成可审计的 relation/sufficiency score，再用极窄、预注册的 source/evidence decision protocol 把排序信号转成保守二分类。

- SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Generation  
  来源：arXiv 2026  
  https://arxiv.org/abs/2605.03534  
  关键点：retrieval 不等于 verification；证据充分性是 set-level property，需要聚合 relation strength、coverage、conflict、disagreement 与 retrieval uncertainty。  
  对 CLAIMARC 的启发：新增 RACL prototype verifier 正好提供 relation strength；`source_count/source_bin/confidence/evidence_combo` 是 coverage/uncertainty 的轻量代理。fs3/fs4 中 prototype score 稳定提升 BGE ranking，而 decision guard 只在 sufficiency 相关区域动手，这是比自由 stacker 更可发表的结构。

- Retrieval-Augmented Generation with Estimation of Source Reliability  
  来源：EMNLP 2025  
  https://aclanthology.org/2025.emnlp-main.1738.pdf  
  关键点：异质来源可靠性会显著影响 RAG 结果，source reliability 应进入 retrieval 与 aggregation。  
  对 CLAIMARC 的启发：params/OCR/VLM/arguments/source0 不能同权；`proto_source_bin` 的复现增益说明 source-stratified prototypes 比普通 global prototype 更贴合任务机制。

- RAFTS: Retrieval Augmented Fact Verification by Synthesizing Contrastive Arguments  
  来源：ACL 2024  
  https://aclanthology.org/2024.acl-long.556/  
  关键点：supporting/refuting contrastive arguments 能帮助小模型做 evidence-aware verification。  
  对 CLAIMARC 的启发：arguments 在当前项目中更适合作为 retrieval/ranking expert，而不是直接覆盖原始 evidence 流；这与多次实验中“排序增益稳、分类边界不稳”的观察一致。

- AmbiFC: Fact-Checking Ambiguous Claims with Evidence  
  来源：TACL 2024  
  https://aclanthology.org/2024.tacl-1.1/  
  关键点：真实 fact-checking 中证据常常支持多种解释，单一 hard verdict 会掩盖 ambiguity。  
  对 CLAIMARC 的启发：room-level bootstrap 下 AUROC/AP 的保守性说明当前数据仍受歧义和弱标签噪声限制；主张应强调 Macro-F1 的 conservative decision gain 与 ranking/screening gain，并把 source-poor/ambiguous 样本作为重点错误分析。

对应到最新实验，fs3+fs4 pooled 的 BGE-base prototype decision 已经给出一个更干净的验证闭环：`rankavg_bge_cm_proto_source_bin` 证明 RACL relation geometry 在新增划分上有排序互补；`proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin` 相对 BGE 的 Macro-F1 在 sample bootstrap 与 room-level group bootstrap 下均显著。下一步优先把 fixed guard / narrow cross-fit selector 写成预注册协议，再追加独立 split 或更强 room-level grouped repeated CV，而不是继续扩大后验规则池。

### 2026-06-08 追加扫描 B：RAG-reasoning、迭代检索与轻量事实核验流水线

这轮扫描补看了 2025 FEVER / Findings 以及最新 RAG-reasoning survey。共同趋势不是“让一个大模型直接判真伪”，而是把 claim 拆解、扩展检索 query、迭代找证据，再用轻量 verifier 或 relation posterior 做最后裁决。

- Fathom: A Fast and Modular RAG Pipeline for Fact-Checking  
  来源：FEVER 2025  
  https://aclanthology.org/2025.fever-1.20/  
  关键点：HyDE-style question generation + BM25/semantic dual-stage retrieval + lightweight LLM verdict/rationale。  
  对 CLAIMARC 的启发：我们已经有 LLM arguments，但它们不应直接覆盖商品来源证据；更合理的是作为 query/evidence expansion 专家，与 BGE/NLI posterior 分开进入 rankmix。

- FIRE: Fact-checking with Iterative Retrieval and Verification  
  来源：Findings NAACL 2025  
  https://aclanthology.org/2025.findings-naacl.158/  
  关键点：把 retrieval 和 verification 做成迭代代理框架，避免一次检索后直接给 verdict。  
  对 CLAIMARC 的启发：当前错误主要在 source-poor / low-confidence 组。后续若继续用 LLM，不要再做 single-pass risk score，而应让 LLM/NLI 先提出缺失证据或冲突点，再决定是否需要回退到 BGE/保守标签。

- A Survey of RAG-Reasoning Systems in Large Language Models  
  来源：Findings EMNLP 2025  
  https://aclanthology.org/2025.findings-emnlp.648/  
  关键点：RAG 和 reasoning 需要双向耦合：reasoning 帮助检索分解与筛选，retrieval 给 reasoning 提供 grounded premises。  
  对 CLAIMARC 的启发：当前最有效结构正是“两段式”：NLI/RACL rankmix 负责 evidence-aware sorting，source/confidence fallback 负责 grounded decision。下一步应该把 router 预定义成 evidence sufficiency rule，而不是继续让小 validation fold 学复杂选择器。

- Zero-Shot Fact Verification via Natural Logic and LLMs  
  来源：Findings EMNLP 2024  
  https://aclanthology.org/2024.findings-emnlp.991/  
  关键点：natural logic 提供比黑盒 verdict 更可解释的 claim-evidence 对齐算子。  
  对 CLAIMARC 的启发：`cv_nli_dual_guard.py` 的 atomic NLI posterior 是同一思想的轻量版本。与其继续换更大的 direct LLM，不如把 posterior 聚合成 `support_mass / conflict_mass / insufficiency_mass`，作为预定义 source/confidence rule 的输入。

- Factuality of Large Language Models: A Survey  
  来源：EMNLP 2024  
  https://aclanthology.org/2024.emnlp-main.1088/  
  关键点：LLM factuality 的关键难点在自动证据检索、事实分解、评估可靠性和 hallucination 控制。  
  对 CLAIMARC 的启发：Qwen direct baseline 弱于 BGE+LR 并不意外；本任务的可发表贡献应强调 grounded evidence representation、检索增强对比学习和可审计的 evidence sufficiency，而不是大模型直接判别。

### 2026-06-08 追加扫描 D：source reliability、sufficiency verification 与分源 pooling

这轮补看 2025 EMNLP/ACL 与 2026 arXiv 后，最新方向进一步支持“把检索增强对比学习当 source/evidence ranking expert，再用证据充分性决定是否采纳”的结构，而不是继续做 OOF 小规则。

- SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Generation  
  来源：arXiv 2026  
  https://arxiv.org/abs/2605.03534  
  关键点：retrieval 不是 verification；证据充分性是 set-level property，独立 passage score 无法发现 missing hops / conflicts。论文聚合 pair-level verifier 的 coverage、relation strength、disagreement、conflict、retrieval uncertainty 做 selective decision。  
  对 CLAIMARC 的启发：当前 `sourcefirst/noargs/OCR/params` 分源专家和 atomic NLI posterior 正应聚合成 set-level sufficiency signals。先做 fixed rank pooling，再接 BGE-protected decision guard，是比窄 mask fallback 更合理的下一步。

- NAACL: Noise-AwAre Verbal Confidence Calibration for Robust LLMs in RAG Systems  
  来源：arXiv 2026  
  https://arxiv.org/abs/2601.11004  
  关键点：contradictory / irrelevant retrieved evidence 会放大过度自信；需要显式噪声感知的校准规则。  
  对 CLAIMARC 的启发：OCR/params/sourcefirst 专家 AP 上升但 Macro 不稳，正是 noisy evidence 排序与分类边界脱钩。下一步应把 evidence noise/confidence 进入 decision calibration，而不是让高 AP 专家直接定阈值。

- Retrieval-Augmented Generation with Estimation of Source Reliability  
  来源：EMNLP 2025  
  https://aclanthology.org/2025.emnlp-main.1738.pdf  
  关键点：RAG 对异质来源可靠性脆弱，RA-RAG 显式估计 source reliability 并把它注入 retrieval 与 generation。  
  对 CLAIMARC 的启发：商品参数、OCR、VLM、arguments 的可靠性差异不应只在事后规则里出现；更适合训练/评估分源 experts，并在 pooling 里按 source presence 与 confidence 做可靠性聚合。

- Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models  
  来源：ACL 2025  
  https://research.google/pubs/astute-rag-overcoming-imperfect-retrieval-augmentation-and-knowledge-conflicts-for-large-language-models/  
  关键点：imperfect retrieval 与 source conflict 是 RAG 瓶颈；方法通过 source-aware consolidation 和 reliability-based finalization 增强鲁棒性。  
  对 CLAIMARC 的启发：我们的 BGE/sourcefirst/noargs/OCR rank pooling 应配合 reliability-based finalization；如果只平均分数，AP 会涨但 Macro 仍被假阳性牵制。

- MultiMind / TriAligner: Multi-Source Alignment for Fact-Checked Claim Retrieval  
  来源：SemEval 2025  
  https://aclanthology.org/2025.semeval-1.303/  
  关键点：dual-encoder contrastive learning 可学习不同来源之间的相对重要性，并用 hard negatives 增强 fact-checked claim retrieval。  
  对 CLAIMARC 的启发：保留 RACL 是合理的，但应让 RACL 服务于 multi-source retrieval/ranking；最终 decision 要额外看 source reliability 与 evidence sufficiency。

更新后的实验优先级：

1. 不再扩 `bgeadvfallback_*`、`bgerateguard_*` 和第一版 `compact_router_nested_*`；它们分别说明“validation group BGE advantage”、“组级过预测率回退”和“把 validation 再切半后自动选头”都不够稳。
2. 普通二层 relation adapter 也已排除：按 `pair_id` 分组 cross-fit 的 LR/HGB stacker 弱于 evidence-type adapter，说明现有 OOF 概率和元数据不足以自动学习稳定信任函数。
3. 第一版 NLI source-pooling micro-calibration 已排除：`arg_ref neutral-rate35` 的 5% score 校准在 pooled screen 可到 AP 0.5093，但补齐 fs0/fs1/fs2 fold-level OOF 后只有 0.5034 / 0.6404 / 0.6084，低于 evidence-type 0.5052 / 0.6412 / 0.6084。
4. evidence-sufficiency narrow fallback 也已基本耗尽：修正版 rawblend 中 `O:low -> BGE` 伤 Macro-F1，`src2_3:medium -> BGE` 的 Macro 小涨不显著；继续加更窄 mask 会变成事后规则挖掘。
5. 下一步不要再靠二值预测率设计规则、普通 stacker 或单个 posterior 微特征，而应把 atomic NLI posterior 聚合成 `support_mass / contradiction_mass / insufficiency_mass`，或把 params/OCR/VLM/arguments 作为 evidence instances 做 source-stratified pooling。
6. 若继续用 LLM，优先做 claim/evidence unit 的 query expansion 或 missing-evidence diagnosis，不再蒸馏单一 risk score。

### 2026-06-08 追加扫描 C：CM p_cls + NLI weighted rank 的文献对齐

这轮实验已经把 `sourcefirst_cm_pcls` 与 NLI evidence posterior 合成固定 rank-weighted score。补看最新 fact-checking/RAG 论文后，当前结果更适合被解释为 **evidence-calibrated retrieval ranking**，而不是又一个事后 ensemble。

- RAFTS: Retrieval Augmented Fact Verification by Synthesizing Contrastive Arguments  
  来源：ACL 2024 / arXiv  
  https://arxiv.org/abs/2406.09815  
  关键点：先检索、重排可验证证据，再生成 supporting/refuting contrastive arguments；小模型也能借助结构化 argument 获得更强 fact verification。  
  对 CLAIMARC 的启发：`sourcefirst_cm_pcls` 是真正保留 RACL/argument 表征价值的检索专家。现在 CM p_cls 与 NLI075 的 AP/AUROC pooled 严格增益，说明主方法可以把 argument-aware RACL 写成 ranking expert，而不是强行让它直接承担最终二分类边界。

- Resolving Conflicting Evidence in Automated Fact-Checking  
  来源：IJCAI 2025 / CONFACT  
  https://arxiv.org/abs/2505.17762  
  关键点：RAG fact-checking 面对来源可信度不同的冲突证据时会明显脆弱；source credibility 需要进入检索与生成阶段。  
  对 CLAIMARC 的启发：当前 fs1/fs0 的失败都与 `source0/lowabs/source-rich medium` 的证据质量分层有关。下一版 decision guard 应显式建 source/confidence reliability，而不是让 NLI/RACL 分数在所有来源上同权触发正类。

- SUCEA: Reasoning-Intensive Retrieval for Adversarial Fact-checking through Claim Decomposition and Editing  
  来源：arXiv 2025  
  https://arxiv.org/abs/2506.04583  
  关键点：把 adversarial claim 分解、去上下文化、迭代检索与 claim editing，再聚合 evidence 预测 entailment。  
  对 CLAIMARC 的启发：直播电商 claim 已被压到 `(product, attribute)`，但证据仍可再拆成 params/OCR/VLM/argument/NLI atomic units。下一步更像 evidence-unit aggregation，而不是继续扩大单层 selector。

- AlignRAG: Leveraging Critique Learning for Evidence-Sensitive Retrieval-Augmented Reasoning  
  来源：arXiv 2025  
  https://arxiv.org/abs/2504.14858  
  关键点：用 contrastive critique synthesis 训练 evidence-sensitive critic，检测推理是否与检索证据对齐。  
  对 CLAIMARC 的启发：`NLI075` 在当前 pooled AUROC 上补强 CM p_cls，正是轻量 critic/relation posterior 的作用。若继续用 LLM，应训练或提示它输出 evidence-alignment critique，而不是输出单一 risk score。

- Fraunhofer SIT at CheckThat! 2025: Multi-Instance Evidence Pooling for Numerical Claim Verification  
  来源：CLEF 2025 / CheckThat! 2025  
  https://publica.fraunhofer.de/entities/publication/aa900b47-0330-4533-bbfa-2a1f35400777  
  关键点：dense retrieval + contrastive cross-encoder reranking + multi-instance evidence pooling；attention/LogSumExp pooling 优于简单拼接。  
  对 CLAIMARC 的启发：当前把 CM p_cls、BGE、NLI posterior 做 rank-weighted fusion 已经接近 multi-instance evidence pooling 的简化版。下一步可把 decision guard 写成 source-stratified pooling：lowabs/source0 用 BGE protected decision，source-rich 使用 CM/NLI pooled rank。

### AmbiFC: Fact-Checking Ambiguous Claims with Evidence

- 来源：TACL 2024  
  https://aclanthology.org/2024.tacl-1.1/
- 关键点：真实 fact-checking 中，检索证据常常不能单义地支持/反驳 claim；论文用细粒度证据标注和 soft label 学习 veracity distribution，而不是把歧义强压成单一标签。
- 对 CLAIMARC 的启发：当前 fs2 的错误很像“证据不充分/解释歧义导致高分假阳性”。下一轮 decision head 不应只做 hard threshold，而应给 source-poor / low-confidence 样本加软 veto 或更高证据充分性要求。

### What Evidence Do Language Models Find Convincing?

- 来源：ACL 2024 long paper  
  https://aclanthology.org/2024.acl-long.403/
- 关键点：LLM 在冲突证据场景下过度依赖网页与 query 的相关性，而忽略来源质量、文风中立性、科学引用等人类会重视的可信度特征。
- 对 CLAIMARC 的启发：这解释了为什么直接 Qwen 判别和简单 evidence-state teacher 都弱：高相关话术/argument 可能被误当成强证据。需要在 ranking 后增加 source-quality / evidence-sufficiency 约束，尤其压制 `source0/src2_3 + low/absent confidence` 的高分样本。

### Beyond True or False: Retrieval-Augmented Hierarchical Analysis of Nuanced Claims

- 来源：ACL 2025 long paper  
  https://aclanthology.org/2025.acl-long.1434/
- 关键点：复杂 claim 需要拆成层级 aspect，并分别汇总 affirmative / neutral / opposing evidence；简单 true/false verdict 容易把局部条件下成立的陈述误判为整体成立。
- 对 CLAIMARC 的启发：直播电商 `(product, attribute)` 已经是一个 aspect-level task，但属性内部仍有来源维度和证据充分性维度。下一轮可以把 NLI posterior 聚合成 `support_mass / conflict_mass / insufficiency_mass`，先用于假阳性 veto，后续再扩展为层级 evidence audit 表。

### AVeriTeC / FEVER 2024 shared-task takeaways

- 来源：FEVER 2024 workshop / AVeriTeC shared task  
  https://aclanthology.org/events/fever-2024/
- 关键点：AVeriTeC 评分要求 verdict 正确且 evidence quality 达标；这比只看分类标签更接近真实 fact-checking。
- 对 CLAIMARC 的启发：论文评测可以补充 evidence-quality / source-sufficiency 分析；方法上也应把“证据是否足够判定”纳入 decision head，而不是让 rankmix 分数独自决定高风险标签。

## 1. 证据增强事实核验

### RAFTS: Retrieval Augmented Fact Verification by Synthesizing Contrastive Arguments

- 来源：ACL 2024 long paper / arXiv  
  https://arxiv.org/abs/2406.09815  
  https://aclweb.org/anthology/2024.acl-long.556.pdf
- 关键点：先检索和重排证据，再生成 supporting/refuting contrastive arguments，用 argument 作为 few-shot fact verification 的结构化中间层。
- 对 CLAIMARC 的启发：当前 `argument_aug.py` 基本复现了 supporting/refuting/evidence-gap 的思想；实验也证明 arguments 对排序有帮助。下一步不应只是拼接 argument，而应把 argument 分支作为独立专家或蒸馏信号，避免它扰乱 p_cls 分类边界。

### Conflicting evidence in RAG fact-checking

- 来源：IJCAI 2025, CONFACT  
  https://www.ijcai.org/proceedings/2025/1073.pdf
- 关键点：RAG fact-checking 在冲突证据下会过度相信低可信来源；应引入 source credibility / evidence quality，而不是盲目扩充上下文。
- 对 CLAIMARC 的启发：商品参数、OCR、VLM、主播话术、评论弱标签的可信度并不相同。现有 `c` 只反映评论标签可信度，下一步应考虑 evidence-source reliability，例如 params > OCR/VLM generated text > argument generated text；RACL negative/positive pair 也应按证据源质量加权。

### Context placement in retrieval-augmented fact-checking

- 来源：arXiv 2026  
  https://arxiv.org/abs/2602.14044
- 关键点：长上下文 fact-checking 中，相关证据位置影响很大，放在开头或结尾通常更好，中部证据容易被忽略。
- 对 CLAIMARC 的启发：`data.py` 目前把 argument 放在 evidence 流最前面，这是合理的；但 params/OCR/VLM 截断顺序也会影响模型。后续可做 evidence ordering 消融：高可信 evidence 优先，argument 放首位，低置信 VLM 放后。

### FactReasoner: Probabilistic factuality assessment

- 来源：EMNLP Findings 2025 / arXiv  
  https://arxiv.org/abs/2502.18573  
  https://research.ibm.com/publications/factreasoner-a-probabilistic-approach-to-long-form-factuality-assessment-for-large-language-models
- 关键点：把长文本拆成 atomic factual units，检索上下文后显式建模 entailment/contradiction/support posterior，而不是只让 LLM 给单一 verdict。
- 对 CLAIMARC 的启发：当前 direct LLM 判别脚本可以作为第一步；若它有效，应进一步把 LLM 输出拆成 `support_prob / contradiction_prob / insufficiency_prob` 三个 teacher 维度，用于 reliability head 或 OOF gating，而不是只蒸馏一个风险分数。

### FIRE: Fact-checking with Iterative Retrieval and Verification

- 来源：NAACL Findings 2025  
  https://aclanthology.org/2025.findings-naacl.158/
- 关键点：固定检索若干证据再一次性判别容易低效且不可靠；FIRE 把检索和判别写成迭代过程，由当前判断置信度决定是给最终答案还是继续生成检索 query。
- 对 CLAIMARC 的启发：当前 `source_count/source_bin/confidence` 与 BGE 不确定性正好可视为“是否继续检索/是否足够裁决”的替代信号。下一步 score-level calibration 不应只调 alpha，而应显式区分 sufficient-evidence 与 uncertain-evidence 区域；source0/lowabs 可以默认保守，source-rich/medium 才让 RACL/NLI 排序专家介入。

### FactLens: Benchmarking Fine-Grained Fact Verification

- 来源：ACL Findings 2025  
  https://aclanthology.org/2025.findings-acl.929/
- 关键点：复杂 claim 应拆成 sub-claims 做细粒度验证，并评估 sub-claim 质量；细粒度分解能减少证据检索歧义，提高可解释性。
- 对 CLAIMARC 的启发：CLAIMARC 的 `(product, attribute)` 已经比整句直播话术更细，但同一 attribute 仍混合商品参数、OCR/VLM 和 argument 证据。当前 atomic NLI posterior 与 `srcbin_conf` fallback 正是在做更细的证据层归因；论文叙事可以把它写成 product-attribute factuality 的 fine-grained verification。

### Fathom: A Fast and Modular RAG Pipeline for Fact-Checking

- 来源：FEVER 2025  
  https://aclanthology.org/2025.fever-1.20/
- 关键点：轻量开源 fact-checking pipeline 可以用 query expansion、BM25+semantic retrieval、轻量 LLM verdict 组合；模块化比单个大模型更适合低成本部署。
- 对 CLAIMARC 的启发：我们目前“BGE 排序 + RACL/argument 排序互补 + source/confidence-aware decision fallback”的 dual-head 结构符合模块化路线。Qwen-Flash direct verdict 为负，并不推翻 LLM/RAG 方向；它说明在该数据上直接 verdict 弱于专门的 embedding/retrieval 与证据源约束。

### E-Verify: Scalable Embedding-based Factuality Verification

- 来源：EMNLP Findings 2025  
  https://aclanthology.org/2025.findings-emnlp.308/
- 关键点：factuality verification 可以转向更可扩展的 embedding-based verification，而不是完全依赖生成式 LLM 判断。
- 对 CLAIMARC 的启发：这支持继续把 BGE/embedding baseline 当强主基线，也支持将贡献定位为 embedding/RACL 排序互补与 evidence-aware routing，而不是追逐 direct LLM verdict。

### Adaptive-RAG: Retrieval use should be conditional

- 来源：NAACL 2024  
  https://aclanthology.org/2024.naacl-long.389/
- 关键点：检索策略应随问题复杂度调整，在 no-retrieval、single-step retrieval 与 iterative retrieval 之间切换。
- 对 CLAIMARC 的启发：`predef_switchrev` 的实验结果与这一思想一致：不是所有样本都应交给 RACL/NLI，也不是所有样本都应回退 BGE。当前失败点在于 validation selector 太弱；更可靠的下一步是预定义 evidence-sufficiency / BGE-uncertainty 层，再在层内做单调校准。

### SciClaimEval: Cross-modal Claim Verification in Scientific Papers

- 来源：arXiv 2026  
  https://arxiv.org/abs/2602.07621
- 关键点：真实论文 claim verification 中，反驳样本通过修改 evidence（figures/tables）而不是改 claim 构造；模型普遍在跨模态证据定位和 aggregation 上低于 human baseline。
- 对 CLAIMARC 的启发：CLAIMARC 的商品 claim 同样是 cross-modal / multi-source：商品参数、OCR、VLM、评论弱标签同时存在。当前 `source_count/confidence` guard 与 `src4p&medium` NLI decision repair 可以写成 evidence-sufficiency adapter，而不是普通分类阈值技巧。

### MuSciClaims: Multimodal Scientific Claim Verification

- 来源：arXiv 2025  
  https://arxiv.org/abs/2506.04585
- 关键点：多模态科学 claim verification 中，即便强 VLM 也经常无法定位正确图内证据，并且存在偏向 supported 的判断倾向；错误来自 evidence localization、cross-modal aggregation 和细粒度图表理解。
- 对 CLAIMARC 的启发：直接 VLM/LLM verdict 弱并不意外；更稳的路线是把视觉/参数证据转成可追溯 evidence records，再让 RACL/BGE/NLI 分别承担 retrieval ranking、semantic baseline 和 entailment calibration。

### MultiMind / TriAligner: Multi-source contrastive alignment for claim retrieval

- 来源：arXiv 2025, SemEval-2025 Task 7  
  https://arxiv.org/abs/2512.20950
- 关键点：跨语言 fact-checked claim retrieval 用 dual-encoder contrastive learning，同时对多来源表示做 alignment，并通过 hard negatives 提升 retrieval accuracy。
- 对 CLAIMARC 的启发：这支持保留检索增强对比学习主线。我们现在的 `sourcefirst_cm_pcls + NLI rank calibration` 与它同属“multi-source alignment + contrastive retrieval expert”，区别是 CLAIMARC 还要处理弱标签与 evidence sufficiency。

### Multi-instance evidence pooling for numerical claim verification

- 来源：CheckThat! 2025 / Fraunhofer SIT  
  https://publica.fraunhofer.de/entities/publication/aa900b47-0330-4533-bbfa-2a1f35400777
- 关键点：数值 claim verification 采用三阶段：dense evidence retrieval、contrastive reranking、再用 MIL evidence pooling 做 claim classification。
- 对 CLAIMARC 的启发：当前 dual-head 结构可写成轻量 MIL 近似：score head 做 retrieval/reranking，decision head 在 source/confidence 组上做 evidence pooling 的保守裁决。后续若要更强方法，可把 `params/ocr/vlm/arguments` 的 per-source score 显式送入 MIL pooling，而不是只用合并后的文本。

### RAG confidence calibration under noisy evidence

- 来源：arXiv 2026, NAACL paper page  
  https://huggingface.co/papers/2601.11004
- 关键点：RAG 系统中，contradictory/irrelevant retrieved context 会放大模型过度自信；需要把检索噪声显式纳入 confidence calibration。
- 对 CLAIMARC 的启发：fs1/fs0 的失败正是 noisy evidence + weak labels 下的过度自信。taxonomy-aware 和 adaptive repair 的正向结果说明“何时信任 RACL/NLI、何时回退 BGE”应由 evidence noise / source sufficiency 控制，而不是靠单一全局阈值。

## 2. noisy-label 与 contrastive learning

### LNPL: Towards Robust Learning with Noisy and Pseudo Labels for Text Classification

- 来源：Information Sciences 2024  
  https://www.sciencedirect.com/science/article/pii/S0020025524000732
- 关键点：文本 noisy label 任务中，将 clean/noisy 样本区分后分别用 positive/negative training，并对噪声来源做正则，比一刀切训练更稳。
- 对 CLAIMARC 的启发：当前 `c` 已经提供弱 clean/noisy proxy。下一步可以把高 `c` 样本用于监督 CE + SupCon，低 `c` 样本更多用于 teacher distillation 或 negative/complementary regularization，避免低可信标签主导分类头。

### ECLB: Efficient contrastive learning on bi-level for noisy labels

- 来源：Knowledge-Based Systems 2024  
  https://www.sciencedirect.com/science/article/pii/S0950705124007627
- 关键点：同时做 feature-level 和 label-level contrastive learning，并用 adaptive mask 筛选可靠 positive pairs。
- 对 CLAIMARC 的启发：已实现 `cl_c_min` / `cl_neg_c_min` 是第一步。更进一步可把 pair 是否可靠从单一 `c` 扩展为 `label agreement + prediction agreement + feature similarity + same attribute` 的 adaptive mask。

### RCKD: Robust contrastive knowledge distillation for long-tailed noisy class labels

- 来源：Knowledge-Based Systems 2025  
  https://www.sciencedirect.com/science/article/pii/S0950705125014479
- 关键点：把多专家知识蒸馏与 dual-mode contrastive learning 结合，用于长尾 + noisy label 场景。
- 对 CLAIMARC 的启发：这正对应当前症状。BGE+LR 分类边界稳，CLAIMARC/argument 分支排序互补强；应让 CLAIMARC 学 BGE teacher 的软边界，同时保留 RACL 表征学习。

### WeStcoin: Weakly supervised text classification with noisy-labeled imbalanced samples

- 来源：Neurocomputing 2024  
  https://www.sciencedirect.com/science/article/pii/S0925231224013882
- 关键点：弱监督文本分类中，同时建模 clean-label pattern 和 noisy-label pattern，并考虑不平衡。
- 对 CLAIMARC 的启发：当前正类率约 38%，标签来自评论弱监督。可以显式建一个 label-noise/risk head 或 reliability head，而不是只把 `c` 当 sample weight。

### HEALON: Progressive hard sample attenuation for learning with noisy labels

- 来源：Intelligent Data Analysis 2026  
  https://journals.sagepub.com/doi/abs/10.1177/1088467X261433375
- 关键点：hard samples near decision boundaries 会破坏伪标签纠错的 memorization effect；应对 hard sample 做渐进式降权/多 snapshot 聚合，而不是把伪标签当真值强拉。
- 对 CLAIMARC 的启发：fold_seed=1 的失败集中在边界折，普通 BGE 蒸馏与 teacher-guided RACL 也印证 hard sample 不宜强蒸馏。下一步如果用 LLM/BGE teacher，应只在高一致/高置信样本上做强监督，对边界样本做 attenuation 或进入 gating 训练。

## 3. 直播电商/欺骗广告背景

### Triple compensation for counterfeits policy in live-streaming selling

- 来源：Electronic Commerce Research and Applications 2025  
  https://www.sciencedirect.com/science/article/pii/S1567422325000845
- 关键点：直播销售中的欺骗广告是平台治理问题，外部惩罚和平台政策会改变商家/主播策略。
- 对 CLAIMARC 的启发：论文动机可更明确地连接到平台治理和风险筛查，而不是只写 NLP fact-checking。

### Evolution analysis of live-streaming governance

- 来源：International Journal of Production Economics 2026  
  https://www.sciencedirect.com/science/article/pii/S0925527326000071
- 关键点：直播电商的非合规经营和欺骗广告是现实管理问题。
- 对 CLAIMARC 的启发：如果最终主结果更偏 AUPRC/排序，可以把应用场景定位为平台审核队列的高风险排序，而不是完全自动裁决。

## 4. 立即执行的实验路线

1. **BGE teacher distillation + RACL**：普通 BCE 蒸馏已实现并完成 smoke，结果为负：`distill_bge_weight=0.3` 会压窄概率分布并损害 Macro-F1。保守分歧蒸馏 `distill_bge_weight=0.1, distill_mode=disagree` 能把 seed=1 AUPRC 提到 0.4827（vs fair BGE+LR 0.4641, p=0.086），但 Macro-F1 仍落后。结论：teacher 不适合直接拉分类概率，更适合进入对比样本筛选/可靠性判断。
2. **adaptive contrastive mask**：teacher-guided RACL 的 `agree` 版本已跑 smoke。它比普通 BCE 蒸馏更合理，Macro-F1 到 0.5871，但仍不胜 fair BGE+LR，且 AUPRC 低于 conservative disagreement distillation。`agree_pos`（只过滤 anchor/positive）前两折更差，已提前停止；不要继续沿这个局部变体扩 seed。
3. **leakage-safe threshold / prior calibration**：单折诊断显示 fold 1 的 val/test 正类率漂移会把 CLAIMARC 阈值推高，造成正类召回崩溃；但 train-prior、smooth-prior、balanced-accuracy、Platt 等策略都不能单独提升 OOF Macro-F1。阈值层只能作为辅助，不是主突破口。
4. **full fusion seed=1 复核**：已确认 no-args 与 args OOF bundle 的 `pair_id` 在每折每个 split 上完全对齐。全融合结果显示 `rankavg(args_pcls, noargs_pcls, BGE+LR)` 对 fair BGE+LR 的 AUPRC 提升显著（0.4865 vs 0.4641, Δ=+0.0220, p=0.018），但 Macro-F1 仍低于 BGE+LR（0.5951 vs 0.5975）。因此当前最稳贡献是排序互补，不是分类全指标显著。
5. **evidence reliability ordering**：在 evidence 流中按 source reliability 排序，并限制低可信 VLM/argument 文本长度，减少噪声证据干扰。这个方向比继续调阈值更有希望，因为 seed=1 问题已经被诊断为分数边界/证据噪声，而非单纯阈值漂移。
6. **direct LLM 判别基线 / teacher**：`Qwen-Flash` broad prompt 分数过满，conservative prompt 全量后仍显著弱于 BGE+LR（AUPRC 0.4143 vs 0.4641；Macro-F1 0.5362 vs 0.5975）。它可作为审稿补充基线，但不能作为主 teacher；`rankavg(LLM, BGE)` 也只有不显著 AP 小增且伤分类。粗粒度 `evidence_state` 的正类率也几乎等于总体正类率，不应直接作为可靠特征。
7. **probabilistic teacher / reliability head**：受 FactReasoner 和 HEALON 启发，下一步不要只蒸馏单一 teacher 概率；如果继续用 LLM，应改成 atomic evidence unit 的 entailment/contradiction/insufficiency posterior，再与 BGE/CLAIMARC disagreement、样本 `c`、证据源质量一起训练 reliability/gating head。已试过的 `nlievidenceveto` 说明，直接把 tiny-NLI 聚合量做硬 veto 会过拟合 validation，fs1 最好 Macro-F1 只有 0.5984；posterior 更适合作为平滑 reliability/ordering 特征，而不是单规则翻转。
8. **BGE-uncertainty reliability gate**：新增 `cv_reliability_gate.py` 后，seed0/seed1/seed2 都显示 CLAIMARC/BGE rank-average 可以稳定提高 AUPRC（约 +0.02，三个 split seed 均显著）。Macro-F1 仍未稳定闭环：seed2 的 `switch_uncertain_macro(rankavg(args_pcls, noargs_pcls, BGE+LR))` 首次显著，但同名候选未在 seed0/seed1 复现。这与 HEALON 的 hard-sample attenuation 思路一致：不要强拉所有 hard sample，而是在 BGE 不确定区域让 RACL 排序专家有限介入。
9. **主张拆分**：排序任务用 AUPRC/AUROC，分类任务用 Macro-F1/wF1；只有 repeated grouped CV 稳定的结论进入主表。当前论文叙事应写成“argument-aware RACL 为强 BGE 分类器提供跨 split seed 显著的排序增益”。source-first evidence policy 是最新有用线索：`rankavg(sourcefirst_args_pcls, sourcefirst_BGE)` 在 fs0/fs2 同时显著提升 AUPRC/AUROC/Macro-F1，但 fs1 只复现 AUPRC。dual-head router 进一步支持把模型写成 ranking score head + source-aware decision head：fs0/fs2 已有三指标显著候选，fs1 现在也有 headmix 与 `srcbin_conf_bgefallback_src0_src2_3_lowabs` 两类三指标显著候选。最新 `predef_lowabs_r25_scorefallback_srcconf_bgefallback` 已把 fs1 的 score-side/decision-side fallback 压缩成固定 source/confidence 协议，并在 5k bootstrap 下三项显著；predef-only 路径又确认 fs0 新缓存中 `source0` score-side full BGE fallback 三项显著。分类显著优势不能靠现有概率上的简单 LR reliability head 闭环，fs2 已验证为负；第一轮激进 source-domain CL reweight、bgerateguard、nlievidenceveto 和 `predef_lowabs_valselect_*` 也为负。最新 OOF 机制诊断（`fallback_mechanism_diagnosis_20260608.json`）进一步把下一步收窄为 `protected BGE regions + RACL/NLI regions`：fs0 修 BGE 不确定/lowabs 假阳性，fs1 保护 lowabs/source0 并让 source-rich 样本用 NLI+BGE，fs2 修正类召回边界但要约束新增 FP。OOF-level protected hybrid 筛查显示该方向点估计全正，但最接近统一的 `rank25_bge025_lowabs + protect_lowabs_scoreguard_srcbin_conf` 只在 fs1/fs2 严格，fs0 新缓存 AUROC/Macro-F1 不严格。补充诊断显示 fs0 需要 scorefallback self-threshold，fs1/fs2 需要 protected decision fallback；因此后续不要直接把该 OOF hybrid 写成主方法，应在 fold-level evaluator 中只引入“scorefallback 自阈值 vs protected decision fallback”的极小 validation 开关，并对 medium/source-rich FP 加约束。

## 5. 2026-06-08 追加：检索、证据关系与 hard-negative 方向复核

### RAFTS: Retrieval Augmented Fact Verification by Synthesizing Contrastive Arguments

- 来源：ACL 2024  
  https://aclanthology.org/2024.acl-long.556/
- 关键点：RAFTS 先检索并重排可信来源证据，再基于证据生成支持/反驳两个方向的 contrastive arguments，最后做 claim verification。它的贡献不只是“用 LLM 写理由”，而是把证据检索、证据重排和支持/反驳论证显式拆开。
- 对 CLAIMARC 的启发：当前 arguments 已证明会在 `source_count=0` 时引入 speculative noise；因此 arguments 不能直接无条件进 evidence flow，而应当由真实来源证据约束。`source_first + drop_src0args`、NLI posterior、evidence-type adapter 都与 RAFTS 的“先证据、后论证”方向一致。

### RGCL: Retrieval-Guided Contrastive Learning for Hateful Meme Detection

- 来源：ACL 2024  
  https://aclanthology.org/2024.acl-long.291/
- 关键点：RGCL 用检索引导的对比训练构建 label-aware embedding space，使模型对细微但决定标签的图文差异更敏感。它强调 hard negatives 必须靠近目标判别边界，而不是只扩大负样本数量。
- 对 CLAIMARC 的启发：CLAIMARC 的 RACL 已经是属性分块 hard-negative 版本。2026-06-08 新增 `--cl_neg_filter medium_evtype_conf` 后，fs1/drop-src0args PCLS 为 0.4856 / 0.6173 / 0.5899，对 BGE 不显著且伤 AUROC；进一步用 `--cl_neg_bonus 0.05 --cl_neg_bonus_filter medium_evtype_conf` 只做 soft top-K 排序 bonus，PCLS 也只有 0.4849 / 0.6193 / 0.5861。结论：当前瓶颈不是训练期负样本池优先级，而是 score/decision 层如何判断何时信任证据关系。

### DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts

- 来源：arXiv 2024 / OpenReview 2025  
  https://arxiv.org/abs/2412.10510
- 关键点：DEFAME 是模块化、动态证据检索与多模态专家评估管线；它在 claim verification 中动态选择搜索深度、证据类型和评估工具，并引入更新 benchmark 避免模型记忆污染。
- 对 CLAIMARC 的启发：当前 `PO:medium` 正向、`OV:medium` 易 FP 的残差非常像“不同证据类型可靠性不同”的轻量版本。下一步不应再扩大 taxonomy selector，而应把 evidence type/source sufficiency 作为动态 evidence reliability signal：影响 score fusion、decision fallback 或 auxiliary relation score。

### LAHN: Label-aware Hard Negative Sampling with Momentum Contrastive Learning

- 来源：ACL Findings 2024 / arXiv  
  https://arxiv.org/abs/2406.07886
- 关键点：LAHN 强调 hard negatives 要 label-aware，并用 momentum memory 保持稳定；它不是简单过滤到一个很窄的小组，而是持续提供接近边界、但仍有足够多样性的负样本。
- 对 CLAIMARC 的启发：`medium_evtype_conf` 的硬过滤和 `0.05` soft top-K bonus 都不足，提示 fs1 瓶颈不是负样本优先级本身。若后续还借鉴 LAHN，应升级为 evidence-relation prototype 或 auxiliary relation head，而不是继续在当前 RACL negative sampling 上调权重。

### 立即执行建议

1. 不再继续 `cl_neg_filter` 或 `cl_neg_bonus` 扩三划分；保留为消融和负结果。
2. 不再继续普通 LR/HGB relation stacker；`diagnose_relation_oof_adapter.py` 的 group-safe 结果已说明该线会显著弱于 evidence-type adapter。
3. 不再继续单个 posterior 微特征校准；`arg_ref neutral-rate35` 已在真实 fold evaluator 下低于 evidence-type，说明 pooled screen 不能替代 fold-level protocol。
4. 新增 per-source evidence pooling：先构造 params/OCR/VLM/argument 的 support、contradiction、insufficiency、coverage 特征，再用预注册规则或极低维单调校准控制 score fusion。
5. 继续改 OOF/evaluator 层的 evidence-type score/decision adapter，而不是训练期 negative sampling。
6. 论文方法叙事继续保留“检索增强对比学习 + evidence posterior calibration”；把 taxonomy fixed 作为诊断上界，evidence-type adapter 作为当前最可辩护 pooled 主线候选。

## 6. 2026-06-08 追加：source reliability / sufficiency 与当前 source-policy 结果对齐

### SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification

- 来源：arXiv 2026  
  https://arxiv.org/abs/2605.03534
- 关键点：RAG 的关键不是“有没有检索到文本”，而是 evidence set 是否足够、是否冲突、是否存在不确定性；模型应在证据不足时选择保守输出或 abstain。
- 对 CLAIMARC 的启发：当前 `source_count==0` 上 BGE-negative guard 有明确机制意义：当没有真实来源证据时，RACL/source-policy pooling 的正判更容易是 speculative FP，应让更保守的 BGE 决策接管。fs1/fs2 的 source-policy guard 正是轻量 evidence-sufficiency gate。

### ERA: Evidence-based Reliability Alignment

- 来源：arXiv 2026  
  https://arxiv.org/abs/2604.20854
- 关键点：把可靠性从单一 confidence 标量转向显式 evidence distribution；系统需要知道答案由哪些 evidence state 支撑，而不是只输出一个概率。
- 对 CLAIMARC 的启发：这支持把 params/OCR/VLM/arguments 拆成多个 evidence views，而不是拼成一个长 evidence 流。新加的 `--evidence_policy_mix` 是单模型低成本版本；`diagnose_source_policy_pooling.py` 是多 expert 版本。

### Resolving Conflicting Evidence in Automated Fact-Checking

- 来源：arXiv 2025  
  https://arxiv.org/abs/2505.17762
- 关键点：事实核查中的冲突证据需要 source credibility；简单把全部检索证据塞给模型会降低稳定性。
- 对 CLAIMARC 的启发：`args_only` 作为独立 expert 在 fs1 伤 Macro-F1，OCR-only 在 fs2 单独为负，说明不同 evidence source 的可信边界不同。统一方法应做 source-specific pooling / credibility guard，而不是让 arguments 或 OCR 直接主导二分类。

### Retrieval-Augmented Generation with Estimation of Source Reliability

- 来源：EMNLP 2025  
  https://aclanthology.org/2025.emnlp-main.1738.pdf
- 关键点：显式估计 source reliability 并纳入 RAG 推理，比默认所有来源同等可信更稳。
- 对 CLAIMARC 的启发：fs1/fs2 的分源 experts 不是普通 ensemble，而是 source reliability 的可学习近似。下一步 fs0 完成后，应比较 `rankavg_all` 与 `mean_all` 的重复稳定性，决定最终写成 ranking-oriented source reliability pooling 还是 F1-oriented conservative pooling。

### Noise-AwAre Verbal Confidence Calibration for RAG

- 来源：arXiv 2026  
  https://arxiv.org/abs/2601.11004
- 关键点：噪声、无关、矛盾 retrieved context 会造成 confidence miscalibration。
- 对 CLAIMARC 的启发：旧的 Platt/blend2、小验证折 selector、普通 LR/HGB relation stacker 都不稳，说明本任务的核心不是再学一个全局校准器，而是先把 evidence noise/source sufficiency 分层。source-policy pooling 与 train-only evidence-view dropout 是更符合该文献方向的结构改动。

### 当前实验决策

1. source-policy multi-instance pooling 已在 fs1/fs2 显示结构性正信号；优先补完 fs0，再做三 split pooled bootstrap。
2. `rankavg_all_score_bge_lr_src0_guard/neg_guard` 和 `mean_all_score_bge_lr_src0_guard/neg_guard` 分别代表排序型与 F1 型候选；fs0 不应再临时新增大量候选，以免重复筛选偏差扩大。
3. `--evidence_policy_mix source_first,no_args,ocr_only,params_only` 是下一条单模型结构实验：它把 source reliability 变成训练期多视图一致性，而不是测试期多专家后处理。

## 7. 2026-06-09 追加：多视图一致性失败后的文献对齐

### Contrastive-RAG / critical reasoning with contrastive explanations

- 来源：NAACL 2025  
  https://aclanthology.org/2025.naacl-long.557/
- 关键点：该工作强调 RAG 系统面对 noisy context 时需要显式比较不同 passages 与最终答案的相关性，利用 contrastive explanations 诱导模型作 critical reasoning，而不是只把检索文本拼接后让模型隐式吸收。
- 对 CLAIMARC 的启发：CLAIMARC 的 argument-aware RACL 与 source-policy experts 已经具备“支持/反驳/缺口”对比结构，但简单 `evidence_policy_mix` 和 all-loss `view_consistency_mix` 都失败，说明不能要求不同 evidence view 产生同一分类边界。更合理的方向是让不同 view 产出可比较的 relation/sufficiency 表示，再用低维 source-sufficiency head 或预注册 guard 汇聚。

### Source reliability / conflicting evidence follow-up

- 来源：SURE-RAG (arXiv 2026)、conflicting-evidence fact-checking (IJCAI/arXiv 2025)、RA-RAG (EMNLP 2025)  
  https://arxiv.org/abs/2605.03534  
  https://arxiv.org/abs/2505.17762  
  https://aclanthology.org/2025.emnlp-main.1738.pdf
- 关键点：最新 RAG/fact-checking 线索共同指向 set-level evidence sufficiency、source reliability、conflict/disagreement 与 uncertainty，而不是全局概率校准。
- 对 CLAIMARC 的启发：三划分 source-policy pooling 已证明分源专家的排序信号稳定；但 evidence-view dropout 和 all-loss consistency 说明训练期“随机视图增强”会伤分类边界。下一步应保留 RACL 检索增强机制，但把 source-policy 信号改成显式结构：例如 evidence-view relation prototypes、source-specific sufficiency logits、或只在 representation 空间做轻量一致性，不再强制 auxiliary view 的 CE/logit 与主视图一致。

### 2026-06-09 实验落点

1. source-policy repeated-CV 已补完 fs0/fs1/fs2：`rankavg_all_score_bge_lr_src0_neg_guard` 与 `mean_all_score_bge_lr_src0_neg_guard` 均显著胜 BGE/noargs，排序指标也显著胜 sourcefirst，但点估计仍弱于 evidence-type adapter。
2. `evidence_policy_mix=source_first,no_args,ocr_only,params_only` fs1 smoke 为负；结论更新为：source-policy 的收益主要来自测试时多 expert/source guard，而不是训练期随机 view dropout。
3. all-loss `view_consistency_mix` fs1 smoke 为负：PCLS 0.4758 / 0.6104 / 0.5727，BGE 0.4657 / 0.6288 / 0.6000；dAP +0.0090 不显著，AUROC/Macro-F1 明显受损。embedding-only consistency 也为负：PCLS 0.4753 / 0.6100 / 0.5711。view consistency 关闭。
4. source auxiliary representation（`evidence_combo/confidence/source_count` 三个 metadata 预测头）有局部 Macro-F1 正信号，fold0 到 0.6353、fold3/fold4 也改善，但 pooled PCLS 只有 0.4667 / 0.6115 / 0.5900，仍低于 BGE Macro-F1 0.6000，AP 几乎不变。训练期 metadata 正则不是主突破口。
5. BGE reranker v2-m3 cross-encoder feature baseline 显著弱于 BGE+LR：direct 0.4077 / 0.5185 / 0.5206，LR 0.3984 / 0.5104 / 0.5138。通用 relevance reranker 分数不能替代 claim-verification / evidence-consistency posterior；若继续 cross-encoder，应做任务内 fine-tuning 或使用 NLI/verification 专用模型。
6. 真正值得设计的是 evidence-type aware set-level sufficiency / source reliability score-decision 结构，而不是继续扩大 OOF 后处理、普通二层 stacker 或训练期正则。

## 8. 2026-06-09 追加：set-level sufficiency 结构复核

### SURE-RAG / evidence sufficiency 的可落地边界

- 来源：arXiv 2026  
  https://arxiv.org/abs/2605.03534
- 关键点：evidence sufficiency 是集合属性，缺失证据、冲突证据和检索不确定性不能靠独立 passage relevance 分数解决；需要 coverage、relation strength、disagreement/conflict、retrieval uncertainty 等显式信号。
- 对 CLAIMARC 的启发：我们按这个思路实现了 `cv_set_sufficiency_meta.py`，用 BGE/current/adaptive score disagreement、source_count、evidence_combo、confidence 训练 fold-safe 小 LR head。结果为负：0.4861 / 0.6280 / 0.5859，明显弱于 fixed evidence-type adapter 0.5052 / 0.6412 / 0.6084。解释是：在弱监督小样本中，sufficiency 的低维特征很容易把 source0/absent 组学成假阳性通道；因此“显式 sufficiency”要落成极窄规则或 relation prototype，而不是普通 meta classifier。

### FactLens / FIRE / half-truth 检验线索

- 来源：ACL/NAACL/EMNLP 2025  
  https://aclanthology.org/2025.findings-acl.929/  
  https://aclanthology.org/2025.findings-naacl.158/  
  https://aclanthology.org/2025.emnlp-main.1724.pdf
- 关键点：新近 fact verification 工作共同强调 fine-grained verification、iterative verification、omission/completeness reasoning。它们不是简单 reranking，而是把 claim 分解、证据覆盖与缺失信息显式接入最终判断。
- 对 CLAIMARC 的启发：当前 `PO:medium` decision repair 可以解释为“参数+OCR 有中等覆盖但缺 VLM 时，需要用 evidence posterior 修正二分类边界”；`source_count==0` 不适合直接正判，只适合作为 score-side adaptive repair。`cv_evidence_type_selector.py` 的 fold-safe 离散选择器进一步验证：decision 端几乎总是选择 `PO:medium`，而允许 score 端自由选择 `source_rich_medium` 会轻微伤 AP。这支持把主方法写成固定、先验化、可审计的 evidence-type score/decision adapter。

### 当前实验决策

1. 保留 `evtype_adapt_score_src0_po_medium_decision_po_medium` 为 pooled repeated-CV 主线候选；它等价于 `evtype_fixed_src0_po_medium__po_medium`，并相对 BGE 与旧 `CM+BGE` 三项显著。
2. 把 `cv_set_sufficiency_meta.py` 作为负结果：证明普通 set-sufficiency LR head 不够稳，避免后续继续堆二层 stacker。
3. 把 `cv_evidence_type_selector.py` 作为机制验证：fold-safe selector 接近固定规则但不超过固定规则，说明规则应预注册为结构先验，而不是让小验证折自由选。
4. 下一步若使用大模型/专用 verifier，重点是生成 relation/sufficiency prototype 或任务内 cross-encoder verifier；通用 BGE reranker 和普通 LR/HGB reliability head 已经排除。

## 9. 2026-06-09 追加：RACL prototype verifier

### Multi-instance evidence pooling 的 RACL 几何版本

- 相关线索：Claim verification / fact-checking 的新工作常把多个 evidence units 先转成 relation scores，再通过 pooling 或 verifier 得到最终判断，而不是直接拼接全文。
- 对 CLAIMARC 的启发：`cv_racl_prototype_verifier.py` 直接利用 CLAIMARC 的 retrieval embedding `g` 构造训练折正/负 prototypes，按 source_bin、attribute、combo 等 evidence strata 计算 query 到正/负 prototype 的 similarity gap。这是把 RACL 从“训练时对比约束”推进到“推理时 relation geometry”的低成本检验。

### 实验读数

- 单独 prototype decision 不稳：`rankavg_bge_cm_proto_source_bin` 为 0.5053 / 0.6391 / 0.5891，AP 可与 evidence-type adapter 打平，但 Macro-F1 远低于 0.6084。
- score-side calibration 有用：`diagnose_racl_proto_rankblend.py` 用 prototype source-bin score 与 evidence-type score 做 case+fold 内等权 rankblend，decision 仍沿用 evidence-type，得到 0.5071 / 0.6430 / 0.6084。
- 配对 bootstrap：该 rankblend 相对 BGE 与旧 CM+BGE 三项显著；相对 current guarded 的 Macro-F1 显著，AP/AUROC 未显著；相对 evidence-type adapter 的 AP/AUROC 点估计正向但不显著。

### 当前实验决策

1. RACL prototype 是目前少数能提高 pooled AP/AUROC 点估计、且不伤 Macro-F1 的新信号，应保留为下一版 score calibration 候选。
2. 不能把 prototype 当独立 verifier；它适合做 ranking relation source，decision 仍需要 evidence-type/source-sufficiency guard。
3. 下一步如果要训练结构，应考虑 auxiliary prototype/relation objective：让 `g` 更明确地区分同 source_bin / same evidence-type 下的 support/refute prototypes，而不是继续加普通 CE/logit view consistency。
4. 第一版训练期 `proto_aux_weight=0.02, proto_aux_group=source_bin` smoke 只把 fs1/drop-src0args 的 PCLS 从 0.4849 / 0.6195 / 0.5817 / 0.5959 微调到 0.4845 / 0.6196 / 0.5845 / 0.5992；同一 embedding 的 prototype verifier 可到 0.5022 / 0.6414 / 0.5942，但 p_cls decision 仍弱于 BGE。实验决策更新为：prototype 应先作为 score-side calibration 或 margin/ranking relation objective，而不是继续扩大简单 prototype CE 辅助损失。
5. 协议化 `cv_racl_proto_evtype_protocol.py` 已把该信号写成两个固定分支：`cal50` 为 0.5071 / 0.6430 / 0.6084，`raw25` 为 0.5070 / 0.6438 / 0.6084。二者均显著胜 BGE/旧 CM+BGE；相对 evidence-type adapter 仍未显著。实验决策更新为：prototype calibration 可写成 ranking/screening enhancement，但不能替代 evidence-type adapter 作为独立主结论。
6. `cv_racl_proto_decision_feature.py` 进一步验证 source-sufficiency decision：在 `source_count==0` 上用 raw prototype rank <0.20 veto 正判、>0.75 promote 负判，得到 0.5070 / 0.6438 / 0.6142；相对 evidence-type adapter 的 Macro-F1 sample p=0.0012、pair-level group p=0.0090，且 AP/AUROC 不降。source0-only nested threshold selector 同样为 0.5070 / 0.6438 / 0.6142，相对 evidence-type 的 Macro-F1 sample p=0.0010、group p=0.0050。实验决策更新为：prototype 已经不仅是 ranking signal，也能作为 source-poor sufficiency guard；下一步是新增 split 或预注册后复跑，而不是继续扩大 selector。
7. 新增 `fold_seed=3/4` 验证后，结论更稳但也更细：基础 CLAIMARC pcls 在 fs3/fs4 仍弱于 BGE，说明端到端模型不是主张；`rankavg_bge_cm_proto_source_bin` 在两个新 split 上稳定改善 BGE 的 AP/AUROC，fs3+fs4 pooled 为 0.5009 / 0.6448 / 0.5893；BGE-base prototype decision `proto_decision_cvselect_macro_rankavg_bge_cm_proto_source_bin` 进一步到 0.5009 / 0.6448 / 0.5981，相对 BGE 的 Macro-F1 sample p=0.0012、room-level p=0.0064。实验决策更新为：主线应写成 **RACL prototype relation score + source/evidence sufficiency decision protocol**；score-only prototype、fixed guard、cross-fit selector 必须拆开报告，不能把 decision edit 相对 score-only 的未显著增益说成已闭环。
