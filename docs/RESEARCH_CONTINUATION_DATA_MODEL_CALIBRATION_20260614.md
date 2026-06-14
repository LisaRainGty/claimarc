# ClaimArc Continuation: Data and Model Calibration

Date: 2026-06-14

## Goal Calibration

This iteration follows the corrected research target: improve the proposal-faithful claim-evidence-comment chain, not benchmark separability. A valid training row should preserve:

1. product-attribute pair;
2. recoverable livestream claim from SRT;
3. product-side evidence from title, params, OCR, or detail-image VLM;
4. consumer comments judged against the same claim;
5. perception label from consumer refutation of the claim.

Hard or ambiguous rows are retained as repair/silver states instead of being deleted to inflate AUROC.

## Data Decisions

Current primary dataset remains:

- `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_vlm120_supervised_20260614.jsonl`
- supervised rows: 469
- labels: 303 positive, 166 negative
- strict main rows: 282
- contrastive rows: 266

Guarded silver rows are now reliability-capped in `src/data_quality/build_stateful_proposal_dataset_v2.py`. This affects schema/meta, subjective-evaluation, commercial-promise, semantic-drift, and extra-enumeration states. The rows remain supervised when label-observed, but no longer carry near-main weights.

Fast learnability after the guarded cap:

| Data view | Supervised rows | AP | AUROC | Macro-F1 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| VLM120 guarded-cap | 469 | 0.8137 | 0.6684 | 0.5885 | current main data |
| VLM120 + weak194 guarded-cap | 526 | 0.7818 | 0.6204 | 0.5884 | expansion candidate, not main |
| VLM120 + weak194 + VLM7 | 524 | 0.7909 | 0.6219 | 0.4889 | audit record only |

The weak194 path is useful for recall analysis but not yet suitable as the primary supervised dataset. It adds mostly positives and degrades ranking, so it should be used as silver/auxiliary only after stronger claim and evidence gates.

## Weak-Claim Audit

The P0 weak SRT pool after the previous 1,500 high-priority rows still contains 1,000 candidates. A 300-row exact re-extraction pilot found 194 candidate claims. The no-image joint review of those 194 produced:

- 37 strict main positive refute rows;
- 3 strict main negative support rows;
- 82 repair-missing-claim rows;
- 36 low-information no-aligned-comment rows;
- 17 silver-review rows;
- remaining rows routed to evidence/ambiguity/repair states.

Only 7 weak rows needed VLM evidence repair after applying the existing evidence-repair gate. VLM repair on these 7 produced:

- 1 strict main positive row;
- 4 evidence-incomplete positive silver rows;
- 2 repair-missing-claim rows.

This supports a conservative rule: continue weak recall, but route weak candidates through VLM/manual sampling before main promotion.

Additional subagent audit found recurring weak-pool failure modes:

- non-SRT sources, detail-image OCR, or pure time-span strings being accepted as livestream claim sources;
- attribute mismatch, especially price/quantity claims routed to net-content or package attributes;
- product evidence values being implicitly grafted into the streamer claim;
- broad subjective claims such as quality, good-looking, or cheap being treated as concrete refutable propositions;
- numeric-package granularity errors, such as package count versus per-package amount.

Next weak-recall batches should hard-gate claim provenance (`claim_source` must be SRT or prefilter SRT), claim specificity, and attribute-value compatibility before any row can enter strict main training.

## GPU CV Status

Remote 5-fold grouped CV on VLM120 with local BGE encoder:

| Method | AP | AUROC | Macro-F1 | wF1 |
| --- | ---: | ---: | ---: | ---: |
| CLAIMARC_pcls | 0.7580 | 0.6483 | 0.6128 | 0.6173 |
| CLAIMARC_selectiveRKC | 0.7580 | 0.6483 | 0.6128 | 0.6173 |
| CLAIMARC_v2 | 0.7142 | 0.5818 | 0.5998 | 0.6009 |
| bge_lr | 0.8142 | 0.6975 | 0.6024 | 0.6077 |

Interpretation: CLAIMARC currently has a small decision-boundary advantage on Macro-F1/wF1, but BGE remains stronger on ranking metrics. This is not yet a paper-ready result.

The `source_first` evidence-policy run produced identical OOF probabilities to the default run. This is not a meaningful negative result. The stateful rows currently lack objective argument blocks, so `args_first` and `source_first` collapse to the same source-only evidence text.

Objective argument generation was then run on all 469 VLM120 supervised rows with `src/models/argument_aug.py`. The prompt excludes consumer comments and labels. Coverage was 469/469 with zero generation errors. Lightweight learnability changed as follows:

| Data view | AP | AUROC | Macro-F1 |
| --- | ---: | ---: | ---: |
| VLM120 source evidence only | 0.8137 | 0.6684 | 0.5885 |
| VLM120 + objective arguments | 0.8090 | 0.6912 | 0.6042 |

This is a promising screen: argument text improves AUROC and Macro-F1 while slightly lowering AP. It should be evaluated as an auxiliary evidence view or retrieval expert, not a direct replacement for source evidence.

GPU early screen confirmed this caution. Directly appending arguments into the main evidence stream was stopped after two folds because it produced unstable, poor CLAIMARC generalization:

| Fold | CLAIMARC AP | CLAIMARC AUROC | CLAIMARC Macro-F1 | Note |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.5634 | 0.5316 | 0.5030 | much worse than no-argument VLM120 |
| 1 | 0.6482 | 0.6244 | 0.4753 | threshold drifted to 0.92 |

Log: `logs/gpu_cv_vlm120_args_l8_t10_lam10_fs14.log`.

Decision: do not use arguments as unconditional main input. Use them as a separate retrieval/ranking expert, a view-consistency target, or a low-dimensional objective relation feature.

## Model Direction

The next structure change should preserve retrieval-augmented contrastive learning but add a non-leaky objective argument view:

- generate claim-evidence relation arguments from claim + product evidence only, without consumer comments or labels;
- store `supporting_argument`, `refuting_argument`, and `evidence_gap` fields;
- train with evidence-view consistency and RACL contrastive masks only on strict triplets;
- keep weak/silver rows in supervised CE with capped reliability, not hard contrastive memory.

This aligns with recent evidence-centric fact-checking work, including RAFTS contrastive arguments ([ACL 2024](https://aclanthology.org/2024.acl-long.556/)), iterative retrieval/verification such as FIRE ([NAACL Findings 2025](https://aclanthology.org/2025.findings-naacl.158/)), and recent dynamic contrastive retrieval ideas such as DACLR ([arXiv 2026](https://arxiv.org/abs/2605.27449)).

## Immediate Next Steps

1. Build an objective claim-evidence argument generation queue for the VLM120 main rows, excluding comments and labels from the prompt.
2. Rebuild VLM120 with argument blocks and rerun a small CV screen with `view_consistency_mix` before full CV.
3. Continue weak claim recovery only under stricter gates: fewer claims per pair, high SRT lexical alignment, VLM evidence repair for positive/refute candidates, and silver routing for conflict/lowinfo cases.
4. Do not report the weak194 expansion as the main result until it improves both proposal-validity audits and held-out model behavior.
