# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_joint_review_queue_claimreextract820_plus_negaligned377_reviewed_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims820_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_aligned377_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_claimreextract820_plus_negaligned377_vlm120_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_vlm120_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_vlm120_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_vlm120_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_vlm120_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `1196`
- `label_observed_rows`: `469`
- `contrastive_rows`: `266`
- `repair_rows`: `727`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 727, '1': 303, '0': 166}`
- `sample_role`: `{'repair_unlabeled': 505, 'supervised_main': 282, 'lowinfo_unlabeled': 222, 'supervised_silver_ambiguous': 28, 'supervised_silver_evidence_incomplete': 91, 'supervised_silver_repair_needed': 53, 'supervised_silver_guarded': 15}`
- `promotion_state`: `{'repair_missing_claim': 333, 'main_positive_refute': 165, 'lowinfo_no_aligned_comment': 206, 'repair_missing_evidence': 77, 'repair_insufficient_product_evidence': 139, 'silver_conflicting_comment_relation': 27, 'silver_refute_insufficient_product_evidence': 51, 'silver_refute_missing_product_evidence': 40, 'repair_numeric_value_judgment': 4, 'silver_subjective_eval_attribute': 11, 'silver_schema_meta_attribute': 9, 'main_negative_support': 117, 'repair_identity_claim_value': 6, 'silver_commercial_promise_attribute': 7, 'silver_attribute_semantic_drift': 1, 'silver_enumeration_evidence_extra_values': 3}`
- `claim_found`: `{False: 333, True: 863}`
- `product_evidence_found`: `{False: 324, True: 872}`
- `category_observed`: `{'apparel_and_underwear': 68, 'baby_kids_and_pets': 89, 'beauty_and_personal_care': 16, 'digital_and_electronics': 17, 'food_and_beverages': 88, 'general': 82, 'jewelry_and_collectibles': 8, 'shoes_and_bags': 33, 'smart_home': 32, 'sports_and_outdoor': 36}`
- `duplicate_claim_family_groups`: `56`
- `conflicting_claim_family_groups`: `4`
- `contrastive_masked_by_claim_family`: `16`
- `split`: `{'train': 328, 'test': 94, 'val': 47}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
