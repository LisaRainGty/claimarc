# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_negative_control_queue_n0_500_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_aligned377_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_n0_aligned377_noimg_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_n0_aligned377_noimg_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_n0_aligned377_noimg_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_n0_aligned377_noimg_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `377`
- `label_observed_rows`: `155`
- `contrastive_rows`: `115`
- `repair_rows`: `222`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 222, '0': 137, '1': 18}`
- `sample_role`: `{'lowinfo_unlabeled': 110, 'repair_unlabeled': 112, 'supervised_silver_repair_needed': 33, 'supervised_main': 115, 'supervised_silver_ambiguous': 4, 'supervised_silver_evidence_incomplete': 3}`
- `promotion_state`: `{'lowinfo_no_aligned_comment': 108, 'repair_insufficient_product_evidence': 67, 'repair_missing_claim': 63, 'main_negative_support': 104, 'main_positive_refute': 11, 'repair_missing_evidence': 14, 'repair_identity_claim_value': 1, 'silver_conflicting_comment_relation': 4, 'silver_refute_missing_product_evidence': 2, 'silver_refute_insufficient_product_evidence': 1, 'silver_enumeration_evidence_extra_values': 2}`
- `claim_found`: `{True: 314, False: 63}`
- `product_evidence_found`: `{True: 351, False: 26}`
- `category_observed`: `{'beauty_and_personal_care': 7, 'baby_kids_and_pets': 36, 'digital_and_electronics': 12, 'shoes_and_bags': 21, 'apparel_and_underwear': 40, 'food_and_beverages': 7, 'general': 21, 'sports_and_outdoor': 5, 'jewelry_and_collectibles': 3, 'smart_home': 3}`
- `duplicate_claim_family_groups`: `1`
- `conflicting_claim_family_groups`: `0`
- `contrastive_masked_by_claim_family`: `0`
- `split`: `{'train': 107, 'val': 17, 'test': 31}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
