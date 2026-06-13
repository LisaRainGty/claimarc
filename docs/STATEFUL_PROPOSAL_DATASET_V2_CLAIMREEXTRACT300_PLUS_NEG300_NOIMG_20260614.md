# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_joint_review_queue_claimreextract300_plus_neg300_reviewed_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims300_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_300_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract300_plus_neg300_noimg_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract300_plus_neg300_noimg_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract300_plus_neg300_noimg_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract300_plus_neg300_noimg_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `600`
- `label_observed_rows`: `251`
- `contrastive_rows`: `138`
- `repair_rows`: `349`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 349, '1': 139, '0': 112}`
- `sample_role`: `{'repair_unlabeled': 218, 'supervised_main': 141, 'lowinfo_unlabeled': 131, 'supervised_silver_evidence_incomplete': 63, 'supervised_silver_ambiguous': 10, 'supervised_silver_repair_needed': 29, 'supervised_silver_guarded': 8}`
- `promotion_state`: `{'repair_missing_claim': 127, 'main_positive_refute': 57, 'lowinfo_no_aligned_comment': 129, 'repair_missing_evidence': 42, 'silver_refute_insufficient_product_evidence': 26, 'silver_conflicting_comment_relation': 10, 'silver_refute_missing_product_evidence': 37, 'repair_insufficient_product_evidence': 74, 'repair_numeric_value_judgment': 1, 'silver_subjective_eval_attribute': 6, 'silver_schema_meta_attribute': 3, 'main_negative_support': 84, 'repair_identity_claim_value': 3, 'silver_enumeration_evidence_extra_values': 1}`
- `claim_found`: `{False: 127, True: 473}`
- `product_evidence_found`: `{False: 151, True: 449}`
- `category_observed`: `{'apparel_and_underwear': 37, 'baby_kids_and_pets': 51, 'beauty_and_personal_care': 10, 'digital_and_electronics': 13, 'food_and_beverages': 16, 'general': 59, 'jewelry_and_collectibles': 3, 'shoes_and_bags': 19, 'smart_home': 26, 'sports_and_outdoor': 17}`
- `duplicate_claim_family_groups`: `25`
- `conflicting_claim_family_groups`: `1`
- `contrastive_masked_by_claim_family`: `3`
- `split`: `{'train': 176, 'test': 50, 'val': 25}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
