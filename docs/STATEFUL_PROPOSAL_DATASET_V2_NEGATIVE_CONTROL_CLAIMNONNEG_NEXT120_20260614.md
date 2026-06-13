# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_negative_control_queue_claimnonneg_next120_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg_next120_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg_next120_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg_next120_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg_next120_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg_next120_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `120`
- `label_observed_rows`: `47`
- `contrastive_rows`: `34`
- `repair_rows`: `73`
- `missing_queue_rows`: `0`
- `y_perception`: `{'0': 42, 'None': 73, '1': 5}`
- `sample_role`: `{'supervised_main': 34, 'lowinfo_unlabeled': 40, 'repair_unlabeled': 33, 'supervised_silver_repair_needed': 11, 'supervised_silver_ambiguous': 1, 'supervised_silver_evidence_incomplete': 1}`
- `promotion_state`: `{'main_negative_support': 31, 'lowinfo_no_aligned_comment': 40, 'repair_missing_claim': 17, 'repair_insufficient_product_evidence': 22, 'main_positive_refute': 3, 'repair_missing_evidence': 5, 'silver_conflicting_comment_relation': 1, 'silver_refute_missing_product_evidence': 1}`
- `claim_found`: `{True: 103, False: 17}`
- `product_evidence_found`: `{True: 111, False: 9}`
- `category_observed`: `{'beauty_and_personal_care': 3, 'baby_kids_and_pets': 16, 'apparel_and_underwear': 12, 'shoes_and_bags': 6, 'digital_and_electronics': 1, 'food_and_beverages': 1, 'general': 6, 'sports_and_outdoor': 2}`
- `duplicate_claim_family_groups`: `0`
- `conflicting_claim_family_groups`: `0`
- `contrastive_masked_by_claim_family`: `0`
- `split`: `{'train': 33, 'val': 5, 'test': 9}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
