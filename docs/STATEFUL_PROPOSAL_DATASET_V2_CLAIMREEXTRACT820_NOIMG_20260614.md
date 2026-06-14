# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_fewerclaims_joint_review_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims820_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_noimg_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_noimg_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_noimg_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_noimg_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `820`
- `label_observed_rows`: `349`
- `contrastive_rows`: `134`
- `repair_rows`: `471`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 471, '1': 320, '0': 29}`
- `sample_role`: `{'repair_unlabeled': 360, 'supervised_main': 145, 'lowinfo_unlabeled': 111, 'supervised_silver_evidence_incomplete': 151, 'supervised_silver_ambiguous': 19, 'supervised_silver_repair_needed': 19, 'supervised_silver_guarded': 15}`
- `promotion_state`: `{'repair_missing_claim': 247, 'main_positive_refute': 132, 'lowinfo_no_aligned_comment': 97, 'repair_missing_evidence': 59, 'silver_refute_insufficient_product_evidence': 71, 'silver_conflicting_comment_relation': 18, 'silver_refute_missing_product_evidence': 80, 'repair_insufficient_product_evidence': 66, 'repair_numeric_value_judgment': 3, 'silver_subjective_eval_attribute': 11, 'silver_schema_meta_attribute': 9, 'main_negative_support': 13, 'repair_identity_claim_value': 5, 'silver_commercial_promise_attribute': 7, 'silver_attribute_semantic_drift': 1, 'silver_enumeration_evidence_extra_values': 1}`
- `claim_found`: `{False: 247, True: 573}`
- `product_evidence_found`: `{False: 333, True: 487}`
- `category_observed`: `{'apparel_and_underwear': 31, 'baby_kids_and_pets': 62, 'beauty_and_personal_care': 10, 'digital_and_electronics': 5, 'food_and_beverages': 92, 'general': 64, 'jewelry_and_collectibles': 5, 'shoes_and_bags': 12, 'smart_home': 35, 'sports_and_outdoor': 33}`
- `duplicate_claim_family_groups`: `57`
- `conflicting_claim_family_groups`: `3`
- `contrastive_masked_by_claim_family`: `11`
- `split`: `{'train': 245, 'test': 69, 'val': 35}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
