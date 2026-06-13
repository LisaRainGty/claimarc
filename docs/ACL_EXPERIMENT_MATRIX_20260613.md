# ACL/EMNLP/AAAI Experiment Matrix

This file tracks the experiment evidence needed for a publishable CLAIMARC
paper.  It separates benchmark rows, train-only auxiliary rows, and
recall-oriented candidate rows so that data construction does not leak into
evaluation.

## Current Anchors

Main grouped-CV evaluation dataset:

- `data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_source_recovered_v3_dropunresolved.jsonl`
- n=2,093, positive=686, grouped by `room_id`

Current trusted auxiliary dataset:

- `data/final/repaired_v1/dataset_atomic_productv2_v6clean_hardclean_v1_20260613.jsonl`
- n=2,888, positive=757

Current best completed CLAIMARC anchor:

| setting | AP | AUROC | Macro-F1 | wF1 | status |
|---|---:|---:|---:|---:|---|
| hardclean auxiliary | 0.8389 | 0.9440 | 0.8840 | 0.7764 | main anchor |
| hardclean BGE-LR | 0.8547 | 0.9492 | 0.8882 | 0.7722 | strongest compact baseline |
| hardclean + negbonus003 | 0.8310 | 0.9426 | 0.8781 | 0.7736 | rejected |
| source/conf prototype | fold0 only: 0.8746 | 0.9524 | 0.8982 | 0.8009 | stopped; below fold0 anchor on F1 |

Current statistical status:

- Paired bootstrap and room-grouped bootstrap against BGE-LR show that the
  current hardclean CLAIMARC anchor is not yet significantly better on AP,
  AUROC, or Macro-F1.  Its advantage is calibration and confidence-weighted
  behavior: ECE10 0.1440 vs BGE-LR 0.1742, and pooled wF1 0.7764 vs 0.7722,
  but `dWF1` is not significant yet (room-grouped CI includes 0).
- Mechanism diagnosis file:
  `data/final/cleancl/claimarc_hardclean_mechanisms_vs_bge_20260613.json`.
  Significance file:
  `data/final/cleancl/claimarc_vs_bge_hardclean_grouped_bootstrap_20260613.json`.

## Main Result Table

Minimum required methods:

| family | method | purpose | status |
|---|---|---|---|
| lexical/embedding | BGE-LR | strongest compact baseline | completed for hardclean anchor |
| supervised FT | BERT/RoBERTa classifiers | same-data neural baselines | completed in previous CVs; rerun only for final chosen setting |
| LLM baseline | zero/few-shot risk classifier | external large-model baseline | keep as appendix if API budget allows |
| CLAIMARC | PCLS | parametric head after RACL training | active |
| CLAIMARC | selectiveRKC | retrieval-corrected inference head | active |
| CLAIMARC | v2 blend | legacy fusion head | diagnostic only |

Final table should report AP, AUROC, Macro-F1, confidence-weighted F1, ECE, and
paired-bootstrap deltas against BGE-LR and the strongest fine-tuned baseline.

## Data Ablations

| ablation | dataset / condition | question | current result |
|---|---|---|---|
| hardclean vs raw v6 | hardclean v1 | does removing shortcuts improve main CV? | yes; current anchor |
| valuegate recall | v3 valuegate hp3 | does more short claim recall help? | no; learnability drops |
| candidate hpmerge | hardclean + candidate hpmerge | does recall expansion help? | AP/AUROC up in diagnostic, Macro down |
| teacher-consistent expansion | new selector | can external teacher clean candidates? | useful for negatives, over-prunes positives |
| coverage-only filter | coverage >= 2 | is fact coverage alone enough? | no; removes consumer-signal positives |
| source-family slices | numeric/material/visual/direct | which evidence families are reliable? | needs final mechanism table |

Decision rule:

- Main model should train on hardclean unless a full grouped-CV expansion run
  improves AP/AUROC without materially hurting Macro-F1/wF1.
- Recall-oriented rows can support robustness, teacher, or low-weight auxiliary
  ablations, but should not become the primary clean data claim.

## Model Ablations

Core ablations for CLAIMARC:

| ablation | change | expected claim |
|---|---|---|
| no RACL | `--lambda_cl 0` / no contrastive training | retrieval contrastive geometry matters |
| global negatives | `--global_neg` | attribute blocking is necessary |
| no fusion | no two-stream fusion | claim/evidence token interaction matters |
| no retrieval head | PCLS only vs selectiveRKC | external memory improves adaptable inference |
| source0 weighting | source0 CE/CL scales | evidence sufficiency affects supervision reliability |
| BGE distillation | high-confidence disagreement-only teacher | teacher helps AP only if it does not overwrite RACL |
| evidence view consistency | params/OCR/VLM view dropout | source invariance helps when evidence is incomplete |

Completed negative ablations:

- `cl_neg_filter=same_evtype_conf`: fold0 F1/wF1 loss; rejected.
- `cl_neg_bonus=0.03`: full CV below hardclean anchor; rejected.
- `proto_aux_weight=0.03, source_conf, margin`: fold0 below hardclean anchor on
  Macro-F1/wF1; stopped.
- Online fold-local BGE distillation (`distill_bge_weight=0.05`) stalled after
  teacher construction and produced unstable validation AP; rejected as a
  training-loop component.
- Low-weight hpmerge auxiliary (`weight_scale=0.15`) improved neither fold-0
  AP/AUROC nor F1 relative to hardclean; keep hpmerge as recall/QA pool only.
- Evidence-view consistency was too memory/compute expensive in the current
  implementation and did not improve warmup validation; redesign offline if
  revisited.
- Source auxiliary heads (`combo/conf/count`) hurt fold-0 primary metrics; keep
  source metadata for diagnosis and reporting, not as a regularizer.
- `lambda_cl=0.25` underperformed the hardclean fold-0 anchor; the default
  `lambda_cl=0.5` remains the best completed RACL setting.
- `lambda_cl=0.75` also underperformed the hardclean fold-0 anchor:
  PCLS AP 0.8689, AUROC 0.9527, Macro-F1 0.9024, wF1 0.8097;
  selectiveRKC AP 0.8743, AUROC 0.9543, Macro-F1 0.8897, wF1 0.8148.
  It was stopped after fold 0.

Active:

- Full400 pair-aligned LLM/VLM review has completed and passed audit
  (400/400 coverage, issue_rate=0.01).  The dominant data issues are generic
  evidence, bad claim spans, missing evidence, and value mismatch.
- `softdropbad full400 v3` is the current best data candidate by lightweight
  learnability (AP 0.8856, AUROC 0.9529, Macro-F1 0.9278).  It drops bad claim
  spans but retains insufficient/not-verifiable rows with lower confidence.
- RACL-U mask (`cl_c_min=0.2`, `cl_neg_c_min=0.2`) is the current best model
  pilot.  On fold 0, it improves CLAIMARC from AP 0.8759 / AUROC 0.9612 /
  Macro-F1 0.9155 to AP 0.9136 / AUROC 0.9675 / Macro-F1 0.9304.  It beats
  BGE-LR on AP/AUROC and nearly matches Macro-F1 (BGE-LR 0.9315), while wF1
  remains lower (0.8607 vs 0.8703).
- Full 5-fold RACL-U CV has completed.  Pooled OOF selectiveRKC improves
  Macro-F1/wF1 over BGE-LR (0.9076/0.8221 vs 0.9006/0.8143), but the paired
  bootstrap CI crosses zero and BGE-LR retains a significant AUROC advantage
  (0.9606 vs 0.9490).  This is not yet the final "clearly best" result.
- Fold-safe hybrid diagnostics show the methods are complementary:
  `hybrid_valblend` reaches Macro-F1 0.9156 but loses AP, while
  `hybrid_rankavg` reaches AP 0.8971 but loses Macro-F1.  The next step is
  multi-objective, evidence-conditioned calibration.
- RACL-U+C fold-safe OOF calibration is the new top-line candidate.  On the
  same softdropbad full400 v3 OOF, it reaches AP 0.8953 / AUROC 0.9678 /
  Macro-F1 0.9209 / wF1 0.8491.  Against BGE-LR, AUROC, Macro-F1, and wF1 are
  significant by paired bootstrap; AP is positive but not significant.
- The score-only calibration ablation reaches Macro-F1 0.9230, while the
  source-conditioned ablation reaches AP 0.9091 and wF1 0.8576.  This supports
  a compact "utility-masked RACL + evidence-conditioned calibration" method
  rather than a larger encoder stack.
- `src/models/diagnose_oof_thresholds.py` formalizes saved-vs-oracle OOF
  threshold analysis.  On the RACL-U fold-0 OOF, CLAIMARC's oracle Macro-F1 gap
  is only +0.0027, so the remaining deficit is subgroup/source reliability
  rather than global threshold selection.

## Robustness Experiments

Required robustness table:

| protocol | split / subgroup | purpose |
|---|---|---|
| cross-category | leave major category out where feasible | category transfer |
| cross-room/anchor | grouped CV by room_id | streamer/topic leakage control |
| evidence sparsity | coverage 0/1/2/3 and confidence bins | behavior under incomplete product facts |
| source type | params / OCR / VLM / mixed | source reliability |
| label reliability | c quantiles | sensitivity to measurement confidence |
| threshold stability | val-macro vs prior-stable | deployment calibration |
| OOF oracle gap | saved threshold vs oracle macro/wF1 | ranking-vs-calibration separation |

## Mechanism Checks

Mechanism evidence to report:

- Retrieval geometry:
  - label-match@10 and attribute-mAP@10.
  - same-attribute opposite-label neighbor distance before/after RACL.
- Boundary cases:
  - examples where BGE-LR predicts lexical similarity but CLAIMARC separates
    risk due to claim-evidence mismatch.
- Source sufficiency:
  - performance by evidence source count and confidence.
  - failure cases where OCR/VLM evidence is generic or title-derived.
- Consumer-signal validity:
  - compare positives with explicit expectation-gap comments vs weak alignment.
  - show that valuegate rows fail mainly because consumer alignment is weak, not
    because product evidence is impossible to find.

Current mechanism findings from hardclean OOF:

- CLAIMARC corrects 67 rows that BGE-LR misses; BGE-LR corrects 74 rows that
  CLAIMARC misses.  The methods are complementary but CLAIMARC still has more
  false positives on parameter-exact rows.
- CLAIMARC is better than BGE-LR when exactly two evidence sources are present
  (`dAP=+0.0475`, `dAUROC=+0.0154`, `d_wF1=+0.0255`) and in low-confidence
  quantile slices (`q1 dAP=+0.0482`, `q3 dMacro-F1=+0.0199`).
- CLAIMARC underperforms on digital/electronics, jewelry, shoes/bags, and some
  OCR-only rows.  Error examples show overfiring on exact attribute mentions
  when the evidence merely repeats the attribute name or gives a compatible
  parameter.  This points to a data fix: exact parameter/value alignment and
  negative claim-span veto, not more generic auxiliary rows.

Current mechanism findings from mechanism-repair v2 fold-0:

- CLAIMARC v2 now has stronger ranking than BGE-LR on the repaired candidate
  (AP 0.9182 vs 0.8820), which confirms the repair queue is not noise.
- Its deployed decision rule is still worse than BGE-LR
  (Macro-F1 0.8688 vs 0.8892; wF1 0.7828 vs 0.7926), and switching from
  `prior_stable` to `val_macro` does not fix it.
- Oracle threshold diagnosis gives CLAIMARC v2 Macro-F1 0.8921, close to or
  slightly above BGE-LR's oracle 0.8945.  The practical bottleneck is therefore
  calibration/threshold transfer under evidence-source shifts, not pure
  representation ranking.

Current mechanism findings from softdropbad full400 v3 + RACL-U fold-0:

- Full400 review shows that 219/400 queued rows need more evidence and 134/400
  contain bad claim spans.  This validates data reconstruction as a first-class
  contribution rather than a minor preprocessing detail.
- Masking low-confidence rows from contrastive anchors/negatives fixes the
  default CL instability.  CLAIMARC PCLS AP rises from 0.8759 to 0.9136 and
  AUROC from 0.9612 to 0.9675 on the same fold/test rows.
- Remaining error is concentrated in jewelry, shoes/bags, and confidence-
  weighted slices.  BGE-LR still has a small wF1 advantage, so full-CV results
  must report both ordinary and confidence-weighted classification.

Current full-CV finding:

- RACL-U is useful but not sufficient.  It gives the desired direction on
  Macro-F1 and confidence-weighted F1, but the AP/AUROC gap to BGE-LR remains.
- Slice diagnostics show CLAIMARC gains in beauty, food, smart home,
  digital/electronics, and mixed evidence combinations, while losing in general
  and jewelry categories.  The next repair queue should focus on those two
  categories plus high-weight rows where BGE is correct and CLAIMARC is wrong.
- RACL-U+C fixes the main full-CV bottleneck without changing the base encoder:
  it combines the RACL decision geometry with BGE's ranking signal and
  source/confidence reliability.  This makes calibration a method contribution,
  not a post-hoc reporting trick.

Current residual data finding:

- A 300-row RACL-U residual repair queue has been built from full-CV OOF
  predictions.  It is pair-aligned with the current dataset; 221 rows reuse the
  full400 blinded review and 79 rows require new review.
- The new uncovered rows are dominated by exact value/material hints, P/O/PO
  evidence combinations, BGE-correct/CLAIMARC-wrong errors, and CLAIMARC
  high-confidence false positives.  This is exactly the boundary that motivates
  source-conditioned calibration and exact-value negative repair.
- The 79 newly uncovered rows have now been reviewed, giving full residual
  coverage 300/300 with audit status pass and issue_rate 0.01.  The residual
  distribution is 172 insufficient, 59 supports, 45 not-verifiable, and 24
  contradicts; likely issues are mainly generic evidence (136), missing
  evidence (79), and value mismatch (27).
- Applying these reviews yields two candidates.  `residual conservative v1`
  has 1,783 rows and improves lightweight AP from 0.8856 to 0.9130.
  `residual candidate v1` also has 1,783 rows and improves lightweight
  Macro-F1 from 0.9278 to 0.9305.  Conservative v1 is the first end-to-end
  RACL-U screen because it gives the cleaner ranking signal.
- Residual conservative v1 fold-0 end-to-end RACL-U screen is strongly
  positive: CLAIMARC selectiveRKC reaches AP 0.9651 / AUROC 0.9792 /
  Macro-F1 0.9477 / wF1 0.8451, versus BGE-LR 0.9461 / 0.9716 / 0.9201 /
  0.8110 on the same fold.  This justifies the full 5-fold run now in progress.
- Residual conservative v1 full 5-fold RACL-U CV has completed.  Base RACL-U
  matches or slightly exceeds BGE-LR on AP/AUROC and beats it on wF1, but BGE
  still has a small Macro-F1 edge before calibration:
  CLAIMARC selectiveRKC 0.9309 / 0.9665 / 0.9138 / 0.8440 vs BGE-LR
  0.9305 / 0.9651 / 0.9165 / 0.8263.
- Source-conditioned RACL-U+C is now the best top-line candidate:
  AP 0.9419 / AUROC 0.9750 / Macro-F1 0.9296 / wF1 0.8652 / ECE10 0.0204.
  Against BGE-LR, AUROC, Macro-F1, and wF1 are significant by paired bootstrap;
  AP is positive but still not significant.  This is the cleanest current
  ACL/EMNLP-style main table result.

## Immediate Queue

1. Re-anchor the main benchmark on a no-drop stateful data view.  The
   conservative residual dataset is useful as a diagnostic upper-bound, but it
   drops 173 weak-evidence rows and should not be the final main result.
   Current no-drop artifact:
   `data/final/repaired_v1/dataset_attrpol_hq_mechanism_repaired_softdropbad_full400_v3_raclu_residual_stateful_nodrop_v1_20260613.jsonl`.
2. Promote source-conditioned RACL-U+C into the formal method variant and rerun
   confirmation diagnostics with fixed pre-registered hyperparameters on the
   no-drop stateful view: score-only, source-conditioned, and selected
   source/full.
3. Add a narrow robustness ablation around the mask threshold:
   `(cl_c_min, cl_neg_c_min) in {(0.1,0.1), (0.2,0.2), (0.3,0.3)}` only after
   the calibration diagnostic identifies whether AP or F1 is the binding
   constraint.
4. Use the full400/residual review states to build a formal RACL-U data artifact:
   utility-positive support/contradiction evidence, low-utility ignore masks,
   and bad-claim exclusion, without adding an LLM at inference time.
5. Build the paper tables around four layers: BGE-LR, base RACL-U, score-only
   calibration, source-conditioned RACL-U+C.  Mechanism checks should emphasize
   ECE reduction, source_count/evidence_combo gains, and corrected residual
   boundary cases.

## Data Validity Guardrail

The data line must not treat lower difficulty as better quality.  A row can be
removed from the main benchmark only when it is outside the task definition
(for example, transaction-only/service-only leakage after the product-attribute
scope is fixed).  Hard but valid rows should stay in the benchmark with one of
three explicit states:

- low-reliability consumer signal: keep label, lower `c`;
- insufficient product evidence: keep row, mask hard contrastive role, rerun
  Stage C evidence extraction;
- invalid or weak claim span: keep audit row, route to Stage B re-extraction
  before using it as a strong supervised example.

This guardrail comes directly from the proposal's measurement logic:
CLAIMARC predicts consumer-perception risk, not only objective contradiction.
Therefore `supports` product evidence must not automatically relabel a
consumer-risk positive as clean, and `contradicts` product evidence must not
automatically create a risk label without aligned negative consumer signal.
