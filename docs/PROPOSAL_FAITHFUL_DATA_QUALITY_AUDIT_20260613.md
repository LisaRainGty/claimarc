# Proposal-Faithful Data Quality Audit v1

## Principle
The main supervised dataset should contain complete claim, evidence, and consumer-perception labels. Incomplete rows are not relabeled for training; they are routed to completion queues.

## Outputs
- `audit_all`: `data/final/repaired_v1/proposal_quality_audit_all_v1_20260613.jsonl`
- `complete_claim_evidence_main`: `data/final/repaired_v1/dataset_attrpol_proposal_complete_claim_evidence_v1_20260613.jsonl`
- `complete_after_claim_review`: `data/final/repaired_v1/dataset_attrpol_proposal_complete_after_claim_review_v1_20260613.jsonl`
- `queue_claim_completion`: `data/final/repaired_v1/proposal_completion_queues_v1_20260613/claim_completion.jsonl`
- `queue_evidence_completion`: `data/final/repaired_v1/proposal_completion_queues_v1_20260613/evidence_completion.jsonl`
- `queue_label_alignment_review`: `data/final/repaired_v1/proposal_completion_queues_v1_20260613/label_alignment_review.jsonl`
- `queue_attribute_schema_review`: `data/final/repaired_v1/proposal_completion_queues_v1_20260613/attribute_schema_review.jsonl`
- `markdown`: `docs/PROPOSAL_FAITHFUL_DATA_QUALITY_AUDIT_20260613.md`

## Summaries
### all
- `n`: `16679`
- `labels`: `{0: 15620, 1: 1059}`
- `pos_rate`: `0.0635`
- `claim_state`: `{'claim_present_review_needed': 1832, 'claim_missing': 13165, 'claim_present_specific': 1682}`
- `evidence_state`: `{'evidence_missing': 9296, 'evidence_single_source': 5060, 'evidence_multi_source': 2323}`
- `label_state`: `{'label_negative_no_aligned_review': 14985, 'label_positive_claim_aligned_neg': 1059, 'label_negative_claim_aligned_nonneg': 635}`
- `issues`: `{'claim_specificity_review': 1832, 'missing_product_evidence': 9296, 'weak_or_unsupported_consumer_label': 15450, 'missing_claim': 13165, 'no_claim_but_direct_consumer_claim_reference': 945}`
- `coverage`: `{'0': 9296, '1': 5060, '2': 2174, '3': 149}`
- `attribute_scope`: `{'subjective_or_personal_eval': 445, 'product_attribute': 13769, 'service_or_process': 2465}`
- `split`: `{'train': 11573, 'test': 3326, 'val': 1780}`

### complete_claim_evidence_main
- `n`: `910`
- `labels`: `{0: 621, 1: 289}`
- `pos_rate`: `0.3176`
- `claim_state`: `{'claim_present_specific': 910}`
- `evidence_state`: `{'evidence_single_source': 570, 'evidence_multi_source': 340}`
- `label_state`: `{'label_negative_no_aligned_review': 444, 'label_positive_claim_aligned_neg': 289, 'label_negative_claim_aligned_nonneg': 177}`
- `issues`: `{'weak_or_unsupported_consumer_label': 568}`
- `coverage`: `{'1': 570, '2': 329, '3': 11}`
- `attribute_scope`: `{'product_attribute': 910}`
- `split`: `{'train': 633, 'test': 230, 'val': 47}`

### complete_after_claim_review
- `n`: `1911`
- `labels`: `{0: 1317, 1: 594}`
- `pos_rate`: `0.3108`
- `claim_state`: `{'claim_present_specific': 910, 'claim_present_review_needed': 1001}`
- `evidence_state`: `{'evidence_single_source': 1234, 'evidence_multi_source': 677}`
- `label_state`: `{'label_negative_no_aligned_review': 940, 'label_positive_claim_aligned_neg': 594, 'label_negative_claim_aligned_nonneg': 377}`
- `issues`: `{'weak_or_unsupported_consumer_label': 1214, 'claim_specificity_review': 1001}`
- `coverage`: `{'1': 1234, '2': 648, '3': 29}`
- `attribute_scope`: `{'product_attribute': 1911}`
- `split`: `{'train': 1340, 'test': 414, 'val': 157}`

## Queue Sizes
- `claim_completion`: `{'n': 14997, 'priority': {1: 1087, 2: 13910}, 'label': {0: 14464, 1: 533}, 'claim_state': {'claim_missing': 13165, 'claim_present_review_needed': 1832}}`
- `evidence_completion`: `{'n': 1469, 'priority': {1: 440, 2: 1029}, 'label': {1: 440, 0: 1029}, 'claim_state': {'claim_present_review_needed': 781, 'claim_present_specific': 688}}`
- `label_alignment_review`: `{'n': 2285, 'priority': {2: 2285}, 'label': {0: 2285}, 'claim_state': {'claim_present_review_needed': 1213, 'claim_present_specific': 1072}}`
- `attribute_schema_review`: `{'n': 2910, 'priority': {3: 2910}, 'label': {0: 2817, 1: 93}, 'claim_state': {'claim_missing': 2499, 'claim_present_specific': 225, 'claim_present_review_needed': 186}}`

## Prompt-Ready Completion Queue

`src/data_quality/build_proposal_llm_completion_queue_v1.py` converts the
quality audit into LLM/VLM repair tasks that include raw SRT paths, raw product
params, detail-image paths, current claim snippets, and consumer trigger
examples. The verifier is instructed to repair claim/evidence/alignment
materials only; it must not directly relabel consumer perception.

- output: `data/final/repaired_v1/proposal_llm_completion_queue_v1_20260613.jsonl`
- total repair tasks: 12,859
- priority: P0=379, P1=979, P2=11,501
- task type:
  - joint claim/evidence completion: 6,414
  - claim completion from raw SRT: 5,898
  - product-evidence completion from raw details/images: 547

## Lightweight Difficulty Check

The corrected complete candidates no longer show the suspiciously easy
0.95+ AUROC behavior seen in over-cleaned diagnostic views.

| dataset | AUPRC | AUROC | Macro-F1 |
|---|---:|---:|---:|
| complete claim/evidence main | 0.6367 | 0.8540 | 0.7578 |
| complete after claim review | 0.6834 | 0.8663 | 0.7710 |

Interpretation: the current bottleneck is upstream completion, especially
attribute-grounded claim extraction and product-evidence recovery. Model
experiments should use the complete claim/evidence main candidate as the
current supervised benchmark, while P0/P1 queue items are repaired from raw
data before promotion.

## P0 Completion Pilot

First P0 batch:

- verified rows: 50
- actions: `keep_clean=2`, `keep_risk=2`, `rerun_more_evidence=43`, `drop=3`
- relation states: `claim_only=24`, `evidence_only=15`,
  `supports_claim=4`, `contradicts_claim=2`, `insufficient=5`

Promotable examples include:

- `APPAREL_是否加绒`: claim "加绒加厚的,一体绒" supported by product title.
- `SHOEBAG_鞋底工艺`: claim "我这实心橡胶要越穿越软" contradicted by product
  param "鞋底材质: 橡胶发泡".
- `APPAREL_电源容量`: claim contains "20000移动电源" contradicted by product
  param "电源容量: 10000".

Most non-promotable rows are not useless; they reveal which raw-stage repair is
needed. `claim_only` rows need stronger product-evidence retrieval from
params/OCR/VLM, while `evidence_only` rows need pair-targeted full-SRT claim
re-extraction rather than short keyword windows.

## Pair-Level Claim Re-Extraction Pilot

`src/data_quality/llm_pair_claim_reextract_v1.py` was added to test a stricter
repair path for `evidence_only` rows: scan full raw SRT for one target
`(product, attribute)` pair, require exact source substrings, and map them back
to timestamps.

Pilot on the first 10 `evidence_only` rows from the P0 top-50 verification:

- with no product-evidence hint, 2/10 rows returned exact SRT spans, but manual
  inspection showed attribute drift such as assigning unrelated "乌鸡骨架小" to
  size or "双层海洋毛" to waterproof/breathability.
- after adding the product-evidence hint to the prompt, noisy recall dropped:
  1/10 returned a claim, 6/10 lacked SRT files, and 3/10 had no matching claim.
- the remaining returned claim was still not promotable because "双层寒牙毛"
  describes material/structure, not waterproof/breathability index.

Conclusion: high-quality expansion requires a three-stage repair gate:

1. pair-targeted full-SRT claim re-extraction;
2. claim-attribute validation against aliases, value type, and product evidence;
3. evidence relation validation before promotion.

Rows that fail any gate remain repair/auxiliary candidates, not main training
samples.

## Triplet-Alignment Gate v2

`src/data_quality/audit_proposal_triplet_alignment_v2.py` adds a stricter
proposal-faithful gate on top of the 910 complete rows.  The goal is not to
make the task easier; it checks whether the row is a natural supervised triplet:
at least one SRT claim must be about the target attribute, and at least one
product-side evidence item must support that same target attribute.

Outputs:

- audit: `data/final/repaired_v1/proposal_triplet_alignment_audit_v2_20260613.jsonl`
- aligned training pool:
  `data/final/repaired_v1/dataset_attrpol_proposal_triplet_aligned_v2_20260613.jsonl`
- label-supported audit core:
  `data/final/repaired_v1/dataset_attrpol_proposal_triplet_aligned_label_supported_v2_20260613.jsonl`
- repair queue:
  `data/final/repaired_v1/proposal_triplet_alignment_repair_queue_v2_20260613.jsonl`
- markdown:
  `docs/PROPOSAL_TRIPLET_ALIGNMENT_AUDIT_V2_20260613.md`

Summary on the 910 complete rows:

- triplet aligned with aligned-review-supported label: 259
- triplet aligned with low-confidence proposal negative label: 200
- needs repair before training: 451
- main repair reasons: claim-attribute alignment review 320; product-evidence
  alignment review 253; low-confidence proposal negative 444

This gate caught concrete Stage B/C defects such as:

- inventory/order talk being treated as a `件数` product claim;
- isolated OCR digits being treated as product evidence;
- price or promotion OCR snippets supporting unrelated attributes;
- brand/model/product-name claims drifting into celebrity, store, or generic
  product talk.

Crucially, the gate did not create an easier benchmark.  Frozen BGE-LR over
three grouped 5-fold seeds gives:

| view | n | AUPRC | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|---:|
| complete claim/evidence main | 910 | 0.6021 | 0.8508 | 0.7685 | 0.7051 |
| triplet-aligned pool | 459 | 0.5286 | 0.7810 | 0.7085 | 0.6626 |

Interpretation: stricter alignment removes logically invalid triplets, not hard
valid rows.  The 459-row pool is therefore only a controlled interim benchmark;
the 451-row repair queue should be repaired from raw SRT, params, OCR, and VLM
materials before any paper-scale final experiment.

## Triplet Repair Queue v2

`src/data_quality/build_triplet_alignment_repair_queue_v2.py` converts the
451 triplet-alignment failures into prompt-ready LLM/VLM repair tasks.

- queue:
  `data/final/repaired_v1/proposal_triplet_alignment_llm_repair_queue_v2_20260613.jsonl`
- report:
  `data/final/repaired_v1/proposal_triplet_alignment_llm_repair_queue_v2_20260613.report.json`

Queue distribution:

- total: 451
- priority: P0=140, P1=124, P2=187
- task type: claim+evidence realignment=122; claim-attribute realignment=198;
  product-evidence realignment=131
- labels: positive=138, negative=313

Verifier prompt evolution on the first 30 P0 rows:

| verifier | keep_clean | keep_silver | rerun_more_evidence | drop | note |
|---|---:|---:|---:|---:|---|
| v1 broad | 8 | 0 | 19 | 3 | over-promoted related but not entailing evidence |
| v2 strict relation | 0 | 0 | 27 | 3 | high precision but lost recoverable partial claims |
| v3 minimal claim span | 0 | 1 | 27 | 2 | best current gate; promotes no main-training rows without exact support |

The v3 rule asks the verifier to return the minimal continuous SRT substring
that can be compared with product evidence.  Medium-confidence results become
silver/auxiliary only.  This preserves the paper logic: expansion must come from
better raw-material repair, not from loosening the relation label.
