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
