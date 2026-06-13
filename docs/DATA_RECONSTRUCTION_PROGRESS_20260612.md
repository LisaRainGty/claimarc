# Data Reconstruction Progress 2026-06-12

## Core Finding

The current bottleneck is not primarily the CLAIMARC architecture. It is the
long tail of low-evidence product-attribute pairs. After strict raw-text
recovery and P0/P1 LLM/VLM curation, rows with at least one product-side
evidence source are substantially more learnable than the full dataset.

Lightweight grouped 5-fold TF-IDF diagnostic:

| Dataset view | n | labels | AUPRC | AUROC | Macro-F1 |
|---|---:|---|---:|---:|---:|
| `p0adjudicated` full | 2317 | 1370/947 | 0.7190 | 0.8102 | 0.7820 |
| `p0adjudicated_sourceful` | 1742 | 1070/672 | 0.8385 | 0.9303 | 0.8849 |
| `p0p1adjudicated` full | 2315 | 1373/942 | 0.7739 | 0.8189 | 0.7755 |
| `p0p1adjudicated_sourceful` | 1756 | 1073/683 | 0.8566 | 0.9346 | 0.8751 |

Interpretation: adding P1 VLM evidence improves ranking signal in the full
dataset and improves the high-purity sourceful view. However, the remaining
source-zero rows still destabilize threshold-based Macro-F1.

## Generated Artifacts

- `data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_p0p1adjudicated_v1.jsonl`
- `data/final/repaired_v1/p0p1_adjudicated_v1_report.json`
- `data/final/repaired_v1/filtered_views_p0p1_v1/dataset_attrpol_hq_product_rawtext_llmcurated_p0adjudicated_sourceful_v1.jsonl`
- `data/final/repaired_v1/filtered_views_p0p1_v1/learnability_filtered_views_p0p1_v1.json`
- `data/final/repaired_v1/regeneration_queue_v2_llm_verify_p1_direct_merged_v1.jsonl`
- `data/final/repaired_v1/regeneration_queue_v2_missing_claim_verify_p0_v1.jsonl`

## P1 Direct Evidence Repair

P1 direct product-source-zero rows were verified with image-aware VLM prompts.
After removing error rows and keeping latest records by `pair_id`:

- verified rows: 23
- high-confidence promoted evidence: 16
- `keep_clean`: 9
- `keep_risk`: 7
- source0 count after promotion: 575 -> 559

LLM adjudication over the promoted rows produced atomic states:

- `product_evidence_state`: supported 8, contradicted 7, insufficient 1
- `consumer_signal`: refutes_claim 7, supports_claim 3, irrelevant 3, mixed 3
- fixed-rule result: keep 14, drop 2, flip 1->0 for 3 rows

As before, final labels were derived only from atomic fields, not from the
LLM's direct `label_recommendation`, which showed inconsistencies.

## P2 Direct Evidence Repair

P2 direct product-source-zero rows were also probed after P1:

- verified rows: 10
- high-confidence promoted evidence: 3
- `keep_clean`: 2
- `keep_risk`: 1
- source0 count after promotion: 559 -> 556

After deterministic adjudication, the full dataset changed from 2315 to 2315
rows, with one `0 -> 1` flip. The lightweight diagnostic did not improve:

| Dataset view | AUPRC | AUROC | Macro-F1 |
|---|---:|---:|---:|
| `p0p1adjudicated` full | 0.7739 | 0.8189 | 0.7755 |
| `p0p1p2adjudicated` full | 0.7625 | 0.8192 | 0.7483 |

Therefore P2 is kept as an audit trail but is not the current main training
branch.

## Missing-Claim P0 Audit

The missing-claim queue is not a simple source of additional training rows for
the current livestream-claim task.

P0 missing-claim verification:

- total: 14
- `live_claim_found`: 3
- `risk_candidate`: 2
- `rerun_more_evidence`: 1
- `drop`: 11

Important pattern: many rows contain real product-side evidence and consumer
refutation but no livestream claim. Examples include title/parameter conflicts
such as a product title implying `20000` capacity while structured parameters
show `10000`. These should be treated as a separate merchant-claim or product
listing contradiction audit, not injected into the current `(livestream claim,
product evidence, consumer perception)` training set.

## Methodological Implication

The data-generation logic should separate three sample families:

1. `livestream_claim_sourceful`: SRT claim exists and product-side evidence is
   available. This is the clean main benchmark/training family.
2. `livestream_claim_source0`: SRT claim exists but product-side evidence is
   absent. These rows should be down-weighted, queued for VLM/OCR recovery, or
   excluded from high-purity experiments.
3. `merchant_claim_no_srt`: no SRT claim, but title/params/detail page and
   consumer comments indicate possible listing-level contradiction. This is
   valuable, but it should be modeled as a separate auxiliary task or future
   extension rather than mixed into the core CLAIMARC label space.

## Next GPU Priority

Run CLAIMARC on:

`data/final/repaired_v1/filtered_views_p0p1_v1/dataset_attrpol_hq_product_rawtext_llmcurated_p0adjudicated_sourceful_v1.jsonl`

This is currently the strongest data branch by diagnostic learnability while
retaining enough scale for 5-fold room-grouped validation.
