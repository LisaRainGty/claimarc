# ACL/EMNLP/AAAI Experiment Matrix

This file tracks the experiment evidence needed for a publishable CLAIMARC
paper.  It separates benchmark rows, train-only auxiliary rows, and
recall-oriented candidate rows so that data construction does not leak into
evaluation.

## Current Anchors

### 2026-06-14 full-pair reconstruction reset

The main data construction line has been reset again after the upstream failure
mode became clear: the corpus already contains 13,769 product-attribute pairs,
but most were filtered because the old pipeline failed to recover streamer
claims and/or product evidence.  The correct response is not to keep optimizing
the 910/481-row diagnostic pools.  The paper-scale dataset must be rebuilt from
the full product-attribute population.

Current full-pair artifacts:

- reconstruction protocol:
  `docs/FULL_PAIR_RECONSTRUCTION_PROTOCOL_20260614.md`
- literature-to-design notes:
  `docs/LITERATURE_NOTES_FULLPAIR_RACL_20260614.md`
- full queue:
  `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- queue report:
  `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.report.json`
- SRT claim prefilter:
  `docs/FULL_PAIR_CLAIM_SRT_PREFILTER_20260614.md`
- stratified LLM pilot queue:
  `docs/FULL_PAIR_LLM_PILOT_QUEUE_20260614.md`
- builder:
  `src/data_quality/build_full_pair_reconstruction_queue_v1.py`
- LLM/VLM runner:
  `src/data_quality/llm_full_pair_reconstruct_v1.py`
- Stage-C/VLM evidence repair queue:
  `docs/FULL_PAIR_EVIDENCE_REPAIR_QUEUE_20260614.md`
- evidence-repair8 VLM audit:
  `docs/FULL_PAIR_EVIDENCE_REPAIR8_VLM_AUDIT_20260614.md`
- Stage-B claim repair queue:
  `docs/FULL_PAIR_CLAIM_REPAIR_QUEUE_20260614.md`
- full P0 claim repair seed batch:
  `docs/FULL_PAIR_CLAIM_REPAIR_QUEUE_FULL_P0_STRONGWEAK120_20260614.md`
- claim-reextract joint review audit:
  `docs/FULL_PAIR_CLAIM_REEXTRACT23_NOIMG_AUDIT_V2_20260614.md`

Queue summary:

| item | count |
|---|---:|
| product-attribute pairs | 13,769 |
| P0 / P1 / P2 / P3 | 6,336 / 5,202 / 2,016 / 215 |
| full claim/evidence/label rebuild | 10,394 |
| claim re-extract + label rebuild | 1,918 |
| evidence refresh + label rebuild | 1,117 |
| label rebuild on existing triplet | 340 |

SRT prefilter result over claim-missing/review rows:

| state | count |
|---|---:|
| strong candidate | 3,322 |
| weak candidate | 6,769 |
| very weak candidate | 2,141 |
| no candidate | 80 |

Stratified LLM pilot queue:

| stratum | rows |
|---|---:|
| strong SRT candidate | 24 |
| weak SRT candidate | 24 |
| very weak SRT candidate | 16 |
| no SRT candidate | 8 |

The 72-row pilot is intentionally diagnostic: it includes 10 categories,
58 `claim_missing` rows, and 14 `claim_present_review_needed` rows.

Pilot72 no-image review and Stage-C/VLM repair:

| item | count |
|---|---:|
| pilot72 reviews matched | 72 |
| recovered claims | 28 |
| conservative main candidates before VLM repair | 15 |
| positive silver/insufficient-evidence rows routed to VLM repair | 8 |
| evidence-repair8 rows promoted to `main_positive_refute` | 3 |
| evidence-repair8 rows still silver/repair | 5 |

The VLM repair result is encouraging but deliberately not overclaimed.  It
shows that the earlier evidence gap is partly a Stage-C extraction problem, not
evidence absence.  It also shows why the final benchmark must keep stateful
promotion gates: subjective or visually grounded evidence can help the consumer
perception task, but rows without source-backed product evidence remain outside
the main supervised candidate.

Pilot72 claim repair and joint review:

| item | count |
|---|---:|
| claim-missing/error rows routed to claim repair | 44 |
| pair-targeted SRT re-extraction rows with any exact claim | 23 |
| pair-targeted SRT re-extraction rows with no exact claim | 21 |
| claim-found rows sent back to joint review | 23 |
| joint review accepted claim_found | 6 |
| promotion main rows after identity gate | 1 |
| identity-attribute false promotion blocked | 1 |

This confirms a two-layer Stage-B design: claim re-extraction should maximize
recall from raw SRT, but the joint reviewer and promotion gate must remain
strict.  In particular, brand/model/SKU/barcode attributes now require the SRT
claim itself to contain the identity value; the model may not infer a brand
claim from product parameters or consumer comments.

The next expansion batch is the full-queue P0 strong/weak claim repair seed
queue with 120 rows.  It is not a selected easy set: all rows have strong SRT
candidates and strong consumer triggers, but they still must pass exact SRT
claim re-extraction, full joint review, audit flags, and promotion gates before
entering the main benchmark.

Full P0 strong/weak seed20 result:

| item | count |
|---|---:|
| claim-only processed rows | 20 |
| exact SRT claim found | 17 |
| joint review rows | 17 |
| joint reviewer `claim_found` | 12 |
| main rows after identity gate | 6 |
| silver evidence-repair rows | 4 |
| repair missing-claim rows | 5 |
| identity claim-value repairs | 2 |

The yield is substantially better than the original stratified pilot because it
targets strong SRT candidates with strong consumer triggers, while still using
strict post-retrieval gates.  This is now the preferred route for scaling from
pilot data to a paper-sized full-pair benchmark.

Full P0 strong/weak seed120 and low-noise40 result:

| item | count |
|---|---:|
| claim-repair seed rows | 120 |
| exact SRT claim found | 99 |
| no exact SRT claim | 21 |
| recovered claim candidates | 580 |
| low-noise joint-review rows | 40 |
| conservative main rows after all gates | 8 |
| main positives / negatives | 6 / 2 |
| evidence-silver rows | 12 |
| missing-claim repairs | 11 |
| identity/numeric/commercial-promise repairs | 2 / 2 / 1 |

Two gate corrections were added after manual inspection of the low-noise40
main candidates:

- Numeric and price-like claims require actual value/specification conflict;
  value judgments such as "too expensive" or "too little" remain repair/silver.
- Commercial-promise attributes are preserved but routed to silver unless the
  same promise is directly verified.

Low-noise40 silver12 Stage-C/VLM evidence repair:

| item | count |
|---|---:|
| evidence-silver rows sent to VLM repair | 12 |
| VLM reviews matched | 12 |
| pre-correction apparent main promotions | 3 |
| final conservative main rows after enum/family gates | 1 |
| enumeration-evidence extra-value silver rows | 1 |
| duplicate claim-family silver rows | 1 |
| remaining repair/silver rows | 9 |

This batch exposed two important data-quality failure modes that are now
first-class gates rather than manual notes.  First, an exhaustive color claim
cannot be promoted if product images list extra colors beyond the streamer
enumeration.  Second, the same product-room-claim should not enter the main
supervised view under both a specific attribute and a generic `描述` attribute.
Both rows remain in stateful outputs for repair and mechanism analysis; they
are not deleted as hard samples.

Label policy for this reset:

- old `y/c` are audit-only fields;
- claim extraction must recover a minimal continuous SRT claim for the target
  attribute;
- product evidence must come from title, params, detail OCR, or detail-image
  VLM, not from comments or SRT;
- final `new_y=1` requires at least one attribute-level consumer comment aligned
  to the repaired claim and refuting it;
- product-evidence contradiction alone is a mechanism/evidence relation state,
  not a positive consumer-perception label.
- exhaustive enumerated claims require product-side evidence that supports the
  same enumerated value set, otherwise the row is silver;
- main supervised rows are de-duplicated by product, room, and recovered claim
  family while duplicate hard cases remain in stateful outputs.

The 910-row complete candidate, the 459/481 triplet-aligned diagnostic views,
and all over-cleaned high-AUROC experiments remain useful for error analysis and
model debugging, but they are no longer the main paper benchmark.

### 2026-06-13 methodological correction

The main paper benchmark is being reset to a proposal-faithful supervised
definition.  The user-facing task is not "make AUROC high by cleaning away hard
or incomplete rows"; it is to learn consumer-perceived misleading risk from
complete `(claim, product evidence, consumer label)` triplets constructed from
the raw pipeline described in the proposal.

Current proposal-faithful data artifacts:

- audit view:
  `data/final/repaired_v1/proposal_quality_audit_all_v1_20260613.jsonl`
- current complete claim/evidence main candidate:
  `data/final/repaired_v1/dataset_attrpol_proposal_complete_claim_evidence_v1_20260613.jsonl`
  - n=910, positive=289, positive rate=31.8%
  - every row is product-scope, has a specific SRT claim, and has at least one
    product evidence source
  - labels are still the proposal weak labels; low-confidence negatives remain
    with their original sample weight `c`
- reviewable complete-after-claim-review candidate:
  `data/final/repaired_v1/dataset_attrpol_proposal_complete_after_claim_review_v1_20260613.jsonl`
  - n=1,911, positive=594, positive rate=31.1%
  - requires claim-specificity review before becoming a main benchmark
- prompt-ready completion queue:
  `data/final/repaired_v1/proposal_llm_completion_queue_v1_20260613.jsonl`
  - P0=379, P1=979, P2=11,501
  - queue types: joint claim/evidence completion 6,414; claim completion
    5,898; product-evidence completion 547

Lightweight learnability on the corrected complete candidates:

| dataset | AUPRC | AUROC | Macro-F1 | interpretation |
|---|---:|---:|---:|---|
| complete claim/evidence main | 0.6367 | 0.8540 | 0.7578 | realistic current benchmark |
| complete after claim review | 0.6834 | 0.8663 | 0.7710 | diagnostic until reviewed |

Initial GPU sanity check on the corrected complete main candidate, fold 0
only, single CLAIMARC seed:

| setting | AP | AUROC | Macro-F1 | wF1 | interpretation |
|---|---:|---:|---:|---:|---|
| CLAIMARC default RACL | 0.6265 | 0.8052 | 0.7192 | 0.6324 | below BGE; realistic difficulty |
| CLAIMARC masked RACL `c>=0.15` | 0.6517 | 0.8139 | 0.6804 | 0.5651 | ranking slightly up, threshold/F1 down |
| BGE-LR | 0.6742 | 0.8473 | 0.7847 | 0.7208 | current strongest fold-0 baseline |

P0 LLM/VLM completion pilot on the prompt-ready queue:

- top-50 P0 rows verified with `Qwen3-VL-Plus`, max 4 detail images.
- curation actions: `keep_clean=2`, `keep_risk=2`, `rerun_more_evidence=43`,
  `drop=3`.
- relation states: `claim_only=24`, `evidence_only=15`, `supports_claim=4`,
  `contradicts_claim=2`, `insufficient=5`.
- Interpretation: the queue is useful but most rows need stronger targeted
  repair, not direct promotion.  The next data step is pair-targeted full-SRT
  claim re-extraction for `evidence_only` rows and expanded product evidence
  recovery for `claim_only` rows.
- Pair-level full-SRT re-extraction pilot on 10 `evidence_only` rows confirmed
  that exact substring grounding alone is not enough: after adding product
  evidence hints, noisy recall fell, but the remaining recovered claim still
  drifted to a neighboring attribute.  Promotion must therefore require a
  separate claim-attribute validation gate before any row enters the main
  supervised benchmark.
- Triplet-alignment P0 verifier outputs have been split into second-stage
  repair queues:
  `data/final/repaired_v1/proposal_second_stage_repair_queues_v1_20260613/`.
  Counts are product-evidence refresh 52, full-SRT claim re-extraction 12,
  joint raw rescan 48, and manual/silver review 28.  These queues preserve the
  proposal label and target Stage B/C provenance repair only.
- Product-evidence refresh now has a dedicated runner:
  `src/data_quality/llm_product_evidence_refresh_v1.py`.  It keeps the claim
  fixed, searches only title/params/OCR/detail-image VLM, and routes medium
  confidence to silver rather than main training.
- VLM evidence coverage is a major data bottleneck: only 29/910 complete rows
  and 19/481 triplet-aligned-plus-repair rows have non-empty VLM evidence,
  despite abundant raw detail images.  The next data expansion should rerun
  attribute-targeted detail-image evidence extraction for repair rows before
  treating evidence absence as factual absence.

481-row triplet-aligned-plus-P0-repair fold-0 sanity check:

| setting | AP | AUROC | Macro-F1 | wF1 | interpretation |
|---|---:|---:|---:|---:|---|
| CLAIMARC PCLS | 0.7102 | 0.7720 | 0.6505 | 0.6220 | AP above BGE, but threshold/ranking stability weak |
| CLAIMARC selectiveRKC | 0.7035 | 0.7593 | 0.6711 | 0.6292 | retrieval head helps F1 slightly but remains below BGE |
| BGE-LR | 0.6749 | 0.7953 | 0.6932 | 0.6624 | still stronger on AUROC/F1 |

`lambda_cl=0` selected the same early checkpoint as default RACL on this fold,
so this run is not valid evidence that RACL helps or hurts.  It is a diagnostic
signal that the current 481-row pool is too small and too sparse in same-attribute
opposite-label neighbors for stable contrastive training.  The next paper-scale
experiment should follow the proposal-faithful repair queues rather than
optimizing this interim benchmark.

The previous `softdropbad`, `residual conservative`, and source-conditioned
RACL-U+C results are retained as diagnostic upper bounds and mechanism probes
only.  They should not be reported as the main paper result unless regenerated
from complete claim/evidence/label triplets without dropping hard-but-valid
examples.

Legacy grouped-CV evaluation dataset:

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

The full-pair reconstruction path now has a separate LLM review audit gate:

- script: `src/data_quality/audit_full_pair_llm_reviews_v1.py`
- report:
  `data/final/repaired_v1/full_pair_reconstruction_llm_audit_v1_20260614.report.json`
- flagged rows:
  `data/final/repaired_v1/full_pair_reconstruction_llm_audit_flags_v1_20260614.jsonl`
- manual audit packet:
  `data/final/repaired_v1/full_pair_manual_audit_packet_v1_20260614.csv`

This gate is intentionally upstream of promotion.  It blocks schema and label
logic errors, routes uncertain rows to silver/repair, and records
`mechanism_contradiction_without_consumer_refute` separately from positive
consumer-perception labels.  The manual packet exposes the SRT candidates,
product evidence previews, consumer snippets, LLM review fields, and reviewer
decision columns, so data quality is improved through traceable adjudication
rather than deletion of hard-but-valid rows.

Remote pilot20 no-image run on 2026-06-14:

- reviews:
  `data/final/repaired_v1/full_pair_reconstruction_llm_pilot20_noimg_v1_20260614.jsonl`
- audit:
  `data/final/repaired_v1/full_pair_reconstruction_llm_pilot20_noimg_audit_v2_20260614.report.json`
- manual packet:
  `data/final/repaired_v1/full_pair_manual_audit_packet_pilot20_noimg_v2_20260614.csv`

Result: 20 reviewed rows, 8 recovered claims, 5 main positives, 2 main
negatives, 1 silver positive with insufficient product evidence, and 12
claim/evidence repairs.  The key methodological value is the caught failure
case: one row had consumer refute comments but `claim_evidence_relation` was
`insufficient`; the audit gate now blocks such rows from main promotion.  This
confirms the data line is enforcing claim-evidence-comment alignment rather
than making the AUROC task cleaner by deletion.

Remote pilot72 no-image audit v3:

- reviews:
  `data/final/repaired_v1/full_pair_reconstruction_llm_pilot72_noimg_v1_20260614.jsonl`
- audit:
  `data/final/repaired_v1/full_pair_reconstruction_llm_pilot72_noimg_audit_v3_20260614.report.json`
- manual packet:
  `data/final/repaired_v1/full_pair_manual_audit_packet_pilot72_noimg_v3_20260614.csv`

Result: 72/72 matched reviews, 28 recovered claims, 12 main positives, 3 main
negatives, 8 silver positives needing product-evidence repair, 43
missing-claim repairs, and 1 pair-aware `llm_error`.  This is not yet a final
training set.  It shows that strong/weak deterministic SRT prefiltering can
recover useful claim-comment pairs, but text/OCR-only evidence is insufficient
for a nontrivial set of consumer-refute positives.  The next data step is
targeted Stage C/VLM evidence repair for silver positives plus SRT retrieval
refinement for missing-claim rows, before scaling the same audit gate to the
13,769 full pairs.

## 2026-06-13 Triplet-Alignment Correction

The proposal-complete candidate is now audited with an explicit
claim-attribute-evidence gate:

- script: `src/data_quality/audit_proposal_triplet_alignment_v2.py`
- audit: `data/final/repaired_v1/proposal_triplet_alignment_audit_v2_20260613.jsonl`
- aligned pool: `data/final/repaired_v1/dataset_attrpol_proposal_triplet_aligned_v2_20260613.jsonl`
- repair queue:
  `data/final/repaired_v1/proposal_triplet_alignment_repair_queue_v2_20260613.jsonl`

This gate preserves the proposal labels and sample weights.  It does not drop
rows for being difficult; it sends rows back to Stage B/C repair when the claim
or product evidence is not actually about the target attribute.  On the 910
complete rows, 459 pass the triplet gate and 451 require repair.

The new diagnostic baseline confirms that this is not an over-cleaning route:

| view | n | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|---:|
| complete claim/evidence main | 910 | 0.6021 | 0.8508 | 0.7685 | 0.7051 |
| triplet-aligned pool | 459 | 0.5286 | 0.7810 | 0.7085 | 0.6626 |

Research implication: the final ACL/EMNLP/AAAI experiment should not use the
459-row pool as the final dataset.  It should use it as a controlled sanity
benchmark while repairing the 451 rows from raw materials and continuing the
larger P0/P1 completion path.  Any future high AUROC result must be checked
against this gate to ensure it is not caused by deleting ambiguous but valid
consumer-perception samples.

The first triplet-repair pilot confirms that expansion should be conservative:

- prompt-ready repair queue:
  `data/final/repaired_v1/proposal_triplet_alignment_llm_repair_queue_v2_20260613.jsonl`
- queue size: 451 rows; P0=140, P1=124, P2=187
- first 30 P0 rows:
  - broad verifier v1: 8 keep_clean, but manual review showed over-promotion
    when evidence was merely related to the same broad attribute
  - strict verifier v2: 0 main promotions, 27 rerun_more_evidence, 3 drop
  - minimal-span verifier v3: 0 main promotions, 1 silver, 27 rerun_more_evidence,
    2 drop

The default verifier now uses the v3 policy: return the minimal continuous SRT
claim span, require same-attribute same-proposition support/contradiction, and
route medium-confidence material to silver rather than main training.  This
keeps the repair path faithful to the consumer-perception task and avoids
manufacturing separability by broadening evidence relations.

Full P0 v3 repair has completed:

- output:
  `data/final/repaired_v1/proposal_triplet_alignment_llm_repair_p0_v3_withlabel_20260613.jsonl`
- n=140; keep_clean=15, keep_risk=9, keep_silver=3,
  rerun_more_evidence=110, drop=3
- relation split: supports_claim=19, contradicts_claim=9, insufficient=56,
  claim_only=44, evidence_only=12

The 24 high-confidence keep rows are candidates for a provenance-preserving
promotion step, not yet a new benchmark.  Before training on them, the merge
must map each minimal claim back to an original SRT segment and add product
title as an explicit evidence source where used by the verifier.  This keeps
CLAIMARC's dual-flow input contract intact.

The provenance-preserving merge is now implemented:

- script: `src/data_quality/apply_triplet_alignment_repairs_v2.py`
- merged view:
  `data/final/repaired_v1/dataset_attrpol_proposal_triplet_aligned_plus_p0repair_v2_20260613.jsonl`
- size: 481 = 459 aligned base + 22 P0 high-confidence repairs
- deterministic veto: question-like claim spans are not promoted
- product-title evidence is represented as `evidence_params` with
  `param_key=product_title` so existing model inputs remain explicit.

Diagnostic BGE-LR over three grouped 5-fold seeds:

| view | n | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|---:|
| triplet-aligned pool | 459 | 0.5286 | 0.7810 | 0.7085 | 0.6626 |
| + P0 v3 high-confidence repairs | 481 | 0.6205 | 0.8127 | 0.7074 | 0.6506 |

This is the right direction for data repair: ranking signal improves, but AUROC
does not become implausibly high.  The next model-side run should compare
CLAIMARC/RACL against this 481-row sanity benchmark, while the data-side run
continues P1 repair and targeted re-extraction for P0 rerun_more_evidence rows.
