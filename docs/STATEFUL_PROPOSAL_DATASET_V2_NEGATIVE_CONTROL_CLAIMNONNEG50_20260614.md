# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_negative_control_queue_claimnonneg80_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg50_noimg_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg50_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg50_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg50_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg50_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `50`
- `label_observed_rows`: `26`
- `contrastive_rows`: `17`
- `repair_rows`: `24`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 24, '1': 9, '0': 17}`
- `sample_role`: `{'lowinfo_unlabeled': 12, 'supervised_silver_ambiguous': 4, 'supervised_silver_evidence_incomplete': 1, 'repair_unlabeled': 12, 'supervised_main': 17, 'supervised_silver_repair_needed': 4}`
- `promotion_state`: `{'lowinfo_no_aligned_comment': 12, 'silver_conflicting_comment_relation': 4, 'silver_refute_insufficient_product_evidence': 1, 'repair_insufficient_product_evidence': 9, 'main_positive_refute': 4, 'repair_identity_claim_value': 1, 'main_negative_support': 13, 'repair_missing_claim': 5, 'repair_missing_evidence': 1}`
- `claim_found`: `{True: 45, False: 5}`
- `product_evidence_found`: `{True: 49, False: 1}`
- `category_observed`: `{'beauty_and_personal_care': 8, 'baby_kids_and_pets': 5, 'shoes_and_bags': 3, 'food_and_beverages': 1, 'digital_and_electronics': 3, 'general': 2, 'apparel_and_underwear': 4}`
- `duplicate_claim_family_groups`: `0`
- `conflicting_claim_family_groups`: `0`
- `contrastive_masked_by_claim_family`: `0`
- `split`: `{'test': 5, 'val': 3, 'train': 18}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
