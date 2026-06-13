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
| hardclean + negbonus003 | 0.8310 | 0.9426 | 0.8781 | 0.7736 | rejected |
| source/conf prototype | fold0 only: 0.8746 | 0.9524 | 0.8982 | 0.8009 | stopped; below fold0 anchor on F1 |

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

Active:

- `distill_bge_weight=0.05`, high-confidence disagreement-only.

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

## Immediate Queue

1. Finish disagreement-only BGE distillation CV and compare fold0 with hardclean
   fold0 anchor before allowing all five folds to run.
2. If distillation fails, run one low-weight recall-pool auxiliary experiment
   using full hpmerge or stricter teacher-consistent negatives, but only as a
   data ablation.
3. After selecting the strongest main setting, rerun full baselines and paired
   bootstrap with `n_boot=2000`.
4. Generate mechanism diagnostics from OOF files for the final chosen setting.
