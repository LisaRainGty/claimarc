# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_negative_control_queue_n0_500_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_300_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_n0_300_noimg_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_n0_300_noimg_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_n0_300_noimg_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_n0_300_noimg_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `300`
- `label_observed_rows`: `122`
- `contrastive_rows`: `91`
- `repair_rows`: `178`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 178, '0': 107, '1': 15}`
- `sample_role`: `{'lowinfo_unlabeled': 97, 'repair_unlabeled': 81, 'supervised_silver_repair_needed': 25, 'supervised_main': 91, 'supervised_silver_ambiguous': 3, 'supervised_silver_evidence_incomplete': 3}`
- `promotion_state`: `{'lowinfo_no_aligned_comment': 96, 'repair_insufficient_product_evidence': 55, 'repair_missing_claim': 39, 'main_negative_support': 82, 'main_positive_refute': 9, 'repair_missing_evidence': 11, 'repair_identity_claim_value': 1, 'silver_conflicting_comment_relation': 3, 'silver_refute_missing_product_evidence': 2, 'silver_refute_insufficient_product_evidence': 1, 'silver_enumeration_evidence_extra_values': 1}`
- `claim_found`: `{True: 261, False: 39}`
- `product_evidence_found`: `{True: 281, False: 19}`
- `category_observed`: `{'beauty_and_personal_care': 7, 'baby_kids_and_pets': 33, 'digital_and_electronics': 11, 'shoes_and_bags': 15, 'apparel_and_underwear': 30, 'food_and_beverages': 3, 'general': 16, 'sports_and_outdoor': 3, 'jewelry_and_collectibles': 2, 'smart_home': 2}`
- `duplicate_claim_family_groups`: `1`
- `conflicting_claim_family_groups`: `0`
- `contrastive_masked_by_claim_family`: `0`
- `split`: `{'train': 84, 'test': 26, 'val': 12}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
