# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_negative_control_queue_claimnonneg_rest30_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg_rest30_noimg_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg_rest30_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg_rest30_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg_rest30_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg_rest30_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `30`
- `label_observed_rows`: `12`
- `contrastive_rows`: `10`
- `repair_rows`: `18`
- `missing_queue_rows`: `0`
- `y_perception`: `{'0': 9, 'None': 18, '1': 3}`
- `sample_role`: `{'supervised_main': 10, 'repair_unlabeled': 8, 'lowinfo_unlabeled': 10, 'supervised_silver_repair_needed': 2}`
- `promotion_state`: `{'main_negative_support': 7, 'repair_insufficient_product_evidence': 3, 'main_positive_refute': 3, 'lowinfo_no_aligned_comment': 10, 'repair_missing_claim': 6, 'repair_missing_evidence': 1}`
- `claim_found`: `{True: 24, False: 6}`
- `product_evidence_found`: `{True: 29, False: 1}`
- `category_observed`: `{'digital_and_electronics': 4, 'shoes_and_bags': 3, 'apparel_and_underwear': 3, 'general': 1, 'beauty_and_personal_care': 1}`
- `duplicate_claim_family_groups`: `0`
- `conflicting_claim_family_groups`: `0`
- `contrastive_masked_by_claim_family`: `0`
- `split`: `{'train': 8, 'test': 3, 'val': 1}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
