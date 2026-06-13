# CLAIMARC Final Protocol Status (2026-06-09)

This file separates confirmatory claims from exploratory mechanisms.

## Current Main Candidate

Primary protocol:

`RACL raw25 score-router + fixed RACL source0 guard + fold-safe Qwen3 disagreement router`

Definition:

1. Score head: use a fixed case/fold rank blend of the evidence-type/CM-NLI guarded score with the raw RACL `proto_source_bin` relation rank (`raw25`: 25% prototype rank). No selector or validation-tuned score weight is used.
2. Decision head: start from the CM/NLI guarded binary decision.
3. RACL edit: only for `source_count==0`, use raw `proto_source_bin` rank. Veto positive predictions when rank < 0.20; promote negative predictions when rank > 0.75.
4. Conservative modern-embedding disagreement router: inside each repeated-CV case/fold, select a small Qwen3-Embedding disagreement switch on the other folds only, then apply it to the held-out fold. The selected rule must improve validation Macro-F1 by at least 0.004, non-veto rules require at least 0.015 validation gain, and validation flip rate is capped at 10%. The Qwen3 router changes decisions only; ranking gains come from the RACL raw25 score head.

Five-split pooled results (`fs0`-`fs4`, n=8470):

| Method | AUPRC | AUROC | Macro-F1 | wF1 | Role |
|---|---:|---:|---:|---:|---|
| BGE+LR | 0.4780 | 0.6321 | 0.5834 | 0.6127 | Strong baseline |
| BGE-large repro LR | 0.4816 | 0.6315 | 0.5845 | 0.6206 | Same evaluator frozen embedding baseline |
| Qwen3-Embedding-0.6B LR | 0.4845 | 0.6376 | 0.5900 | 0.6111 | Modern 2025 embedding baseline |
| Qwen3-Embedding-0.6B query-prompt LR | 0.4813 | 0.6351 | 0.5835 | 0.6093 | Prompted embedding check; weaker than no-prompt |
| CM/NLI guarded | 0.5029 | 0.6422 | 0.6010 | 0.6281 | Previous robust main |
| Evidence-type adapter | 0.5039 | 0.6424 | 0.6005 | 0.6277 | Interpretable three-split repair; not better on fs3/fs4 |
| Score-only RACL prototype | 0.5032 | 0.6414 | 0.5893 | 0.6143 | Ranking/prototype mechanism |
| CM/NLI guarded + fixed RACL source0 guard | 0.5029 | 0.6422 | 0.6053 | 0.6359 | Previous main candidate |
| + conservative fold-safe Qwen3 disagreement router | 0.5029 | 0.6422 | 0.6109 | 0.6411 | Previous decision-only candidate |
| RACL raw25 score + conservative Qwen3 router decision | **0.5084** | **0.6456** | **0.6109** | **0.6411** | Current top-line candidate |

Room-level group bootstrap for the current main candidate:

| Comparison | dAP | dAUROC | dMacro-F1 | p(AP/AUROC/Macro) |
|---|---:|---:|---:|---|
| vs BGE+LR | +0.0298 | +0.0134 | +0.0276 | 0.0004 / 0.0316 / 0.0000 |
| vs BGE-large repro LR | +0.0265 | +0.0143 | +0.0265 | 0.0034 / 0.0290 / 0.0000 |
| vs Qwen3-Embedding-0.6B LR | +0.0227 | +0.0075 | +0.0209 | 0.0274 / 0.1942 / 0.0002 |
| vs Qwen3-Embedding-0.6B query-prompt LR | +0.0265 | +0.0103 | +0.0272 | 0.0110 / 0.1022 / 0.0002 |
| vs conservative Qwen3 router score | +0.0053 | +0.0034 | 0.0000 | 0.1650 / 0.1558 / 1.0000 |
| vs fixed RACL source0 guard | +0.0053 | +0.0033 | +0.0055 | 0.1658 / 0.1564 / 0.0098 |
| vs CM/NLI guarded | +0.0052 | +0.0034 | +0.0098 | 0.1540 / 0.1492 / 0.0006 |
| vs evidence-type adapter | +0.0043 | +0.0031 | +0.0103 | 0.1992 / 0.1418 / 0.0004 |

Split-level Macro-F1:

| Split | BGE | Fixed RACL source0 guard | Conservative router decision |
|---|---:|---:|---:|
| fs0 | 0.5716 | 0.6075 | 0.6106 |
| fs1 | 0.6000 | 0.6072 | 0.6069 |
| fs2 | 0.5765 | 0.6190 | 0.6206 |
| fs3 | 0.5783 | 0.5912 | 0.6145 |
| fs4 | 0.5885 | 0.5988 | 0.6007 |

Independent fs3/fs4 validation:

| Scope | Score-router main | vs BGE p(AP/AUROC/Macro) | vs previous conservative router p(AP/AUROC/Macro) |
|---|---|---|---|
| sample bootstrap | 0.5114 / 0.6482 / 0.6076 / 0.6392 | 0.0002 / 0.0028 / 0.0000 | 0.0580 / 0.0430 / 1.0000 |
| pair-level bootstrap | 0.5114 / 0.6482 / 0.6076 / 0.6392 | 0.0012 / 0.0174 / 0.0000 | 0.1036 / 0.0920 / 1.0000 |
| room-level bootstrap | 0.5114 / 0.6482 / 0.6076 / 0.6392 | 0.0036 / 0.0370 / 0.0002 | 0.1372 / 0.1452 / 1.0000 |

The new splits independently support the score-router over BGE on AP/AUROC/Macro-F1. Against the previous conservative router score, the raw25 RACL score head is positive but not room-level significant; use it as the current top-line point estimate and mechanism candidate, not as a confirmatory claim that the old score is beaten under grouped resampling.

No-selector confirmation:

| Method | Five-split result | vs fixed RACL source0 guard p(sample/pair/room Macro) | fs3/fs4 result | fs3/fs4 vs fixed guard p(sample/pair/room Macro) |
|---|---|---|---|---|
| Fixed Qwen3 veto `base_conf<=0.20, qwen_conf>=0.10` | 0.5029 / 0.6422 / 0.6092 / 0.6383 | 0.0016 / 0.0106 / 0.0046 | 0.5051 / 0.6446 / 0.5975 / 0.6261 | 0.0890 / 0.1216 / 0.0576 |

This fixed rule performs no fold-level rule selection. It supports the existence of a robust Qwen3 disagreement-veto effect, while the conservative selected router remains the main decision module because it also captures source-rich bidirectional corrections on fs3/fs4.

Flip diagnosis:

- Five splits: 161 flips / 8470 rows, all in `source_count==0`; net correct +25.
- Five splits: fixed 38 false positives and 55 false negatives; introduced 11 false negatives and 57 false positives.
- fs3/fs4: 58 flips / 3388 rows; net correct +2.
- Conservative router over fixed RACL source0 guard: 270 flips / 8470 rows; net correct +62. It fixes 119 false positives and 47 false negatives while introducing 69 false negatives and 35 false positives.
- Router fs3/fs4: 152 flips / 3388 rows; net correct +38.
- Fixed Qwen3 veto check: 145 flips / 8470 rows, all veto, net correct +55. fs3/fs4: 36 flips / 3388 rows, net correct +14.

## What We Can Claim

- A retrieval-augmented contrastive relation score is useful beyond ranking when used as a narrow source-poor sufficiency guard.
- A fixed RACL prototype score blend (`raw25`) improves the top-line AP/AUROC point estimate while preserving the conservative router's Macro-F1/wF1. It is significant over BGE and modern embedding baselines at room level where noted above.
- The final protocol significantly improves Macro-F1 over the previous CM/NLI guarded protocol under sample, pair-level, and room-level bootstraps.
- The same final protocol significantly improves AUPRC, AUROC, and Macro-F1 over BGE+LR under room-level group bootstrap.
- Against modern Qwen3 embedding baselines, the final protocol significantly improves Macro-F1 at sample, pair, and room levels. AUPRC is now room-level significant against no-prompt Qwen3; AUROC should be treated as positive but not a core significant claim.
- The conservative Qwen3 disagreement router is a decision-only late module: it improves Macro-F1 without changing the score. This should be framed as a calibrated dispute-resolution layer rather than a replacement for the retrieval-contrastive mechanism.
- A no-selector fixed Qwen3 veto also significantly improves five-split Macro-F1 over the fixed RACL source0 guard under sample, pair, and room bootstraps. Use it as a robustness check for the router mechanism, not as the top-line model.

## What We Should Not Claim

- Do not claim evidence-type adapter is better than CM/NLI guarded on five splits. It is not.
- Do not claim broad cross-fit selector is the primary method. It reaches 0.6065 Macro-F1, but the fixed source0 rule is cleaner and already significant.
- Do not claim train-time prototype auxiliary solved classification. The margin/CE auxiliary smoke preserves prototype ranking value but does not close Macro-F1.
- Do not claim the Qwen3 router improves ranking. Ranking improvement comes from the RACL raw25 score head.
- Do not claim raw25 significantly beats the previous conservative router score under room-level group bootstrap. It is sample-level significant and positive at pair/room levels, but pair/room CIs still cross zero.

## Key Artifacts

- OOF builder: `scripts/build_guarded_proto_fs0_fs4_oof.py`
- Combined OOF: `data/final/cleancl/oof_guarded_proto_fs0_fs4_room_20260609.npz`
- Guarded-base decision output: `data/final/cleancl/racl_proto_decision_feature_fs0_fs4_guardedbase_20260609.json`
- Main OOF: `data/final/cleancl/oof_racl_proto_decision_feature_fs0_fs4_guardedbase_20260609.npz`
- Router script: `src/models/cv_oof_disagreement_router.py`
- Router output: `data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs0_fs4_20260609.json`
- Router OOF: `data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs0_fs4_20260609.npz`
- Router bootstrap: `data/final/cleancl/oof_disagreement_router_qwen3_conservative_room_bootstrap_5k_20260609.json`
- Router fs3/fs4 bootstrap: `data/final/cleancl/oof_disagreement_router_qwen3_conservative_fs3_fs4_room_bootstrap_5k_20260609.json`
- Score-router script: `src/models/cv_racl_proto_evtype_protocol.py`
- Score-router output: `data/final/cleancl/racl_proto_evtype_score_router_decision_fs0_fs4_20260609.json`
- Score-router OOF: `data/final/cleancl/oof_racl_proto_evtype_score_router_decision_fs0_fs4_20260609.npz`
- Score-router bootstrap: `data/final/cleancl/racl_proto_evtype_score_router_extra_room_bootstrap_5k_20260609.json`
- Score-router fs3/fs4 OOF: `data/final/cleancl/oof_racl_proto_evtype_score_router_decision_fs3_fs4_20260609.npz`
- Score-router fs3/fs4 bootstrap: `data/final/cleancl/racl_proto_evtype_score_router_fs3_fs4_room_bootstrap_5k_20260609.json`
- No-selector fixed veto OOF: `data/final/cleancl/oof_disagreement_router_qwen3_fixedveto_fs0_fs4_20260609.npz`
- No-selector fixed veto bootstrap: `data/final/cleancl/oof_disagreement_router_qwen3_fixedveto_room_bootstrap_5k_20260609.json`
- Modern embedding OOF: `data/final/cleancl/oof_guardedbase_plus_modern_emb_fs0_fs4_20260609.npz`
- Qwen3 baseline: `data/final/cleancl/cv_embedding_qwen3emb06b_srcargs_drop_fs0_fs4_20260609.json`
- Prompted Qwen3 baseline: `data/final/cleancl/cv_embedding_qwen3emb06b_query_srcargs_drop_fs0_fs4_20260609.json`
- Modern embedding bootstrap: `data/final/cleancl/guarded_source0_vs_modern_emb_room_bootstrap_5k_20260609.json`
- Modern embedding prompt-check bootstrap: `data/final/cleancl/guarded_source0_vs_modern_emb_prompt_room_bootstrap_5k_20260609.json`
- Sample bootstrap: `data/final/cleancl/racl_proto_decision_feature_fs0_fs4_guardedscore_fixed_sample_bootstrap_5k_20260609.json`
- Pair bootstrap: `data/final/cleancl/racl_proto_decision_feature_fs0_fs4_guardedscore_fixed_pair_bootstrap_5k_20260609.json`
- Room bootstrap: `data/final/cleancl/racl_proto_decision_feature_fs0_fs4_guardedscore_fixed_room_bootstrap_5k_20260609.json`
- fs3/fs4 OOF: `data/final/cleancl/oof_racl_proto_decision_feature_fs3_fs4_guardedbase_20260609.npz`
- Flip diagnosis: `data/final/cleancl/guarded_proto_source0_flip_diagnosis_fs0_fs4_20260609.json`

## Literature Alignment

The current direction matches recent fact-checking work that treats retrieval, evidence sufficiency, and verification as separate decisions rather than a single monolithic classifier:

- [FIRE: Fact-checking with Iterative Retrieval and Verification](https://aclanthology.org/2025.findings-naacl.158/) supports explicitly separating evidence discovery from verification.
- [DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts](https://proceedings.mlr.press/v267/braun25b.html) supports choosing evidence operations conditionally rather than using one static evidence view.
- [A Reality Check on Context Utilisation for Retrieval-Augmented Generation](https://www.copenlu.com/publication/2025_acl_hagstr%C3%B6m/) supports auditing whether retrieved context is sufficient and actually used.
- [RAFTS: Retrieval Augmented Fact Verification by Synthesizing Contrastive Arguments](https://aclanthology.org/2024.acl-long.556/) supports retaining contrastive argument/relation geometry while using a guarded decision layer for final classification.
