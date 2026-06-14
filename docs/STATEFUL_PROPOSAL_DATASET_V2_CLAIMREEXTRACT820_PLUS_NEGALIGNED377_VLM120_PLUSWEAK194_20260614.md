# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_joint_review_queue_claimreextract820_plus_negaligned377_plusweak194_reviewed_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims820_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_aligned377_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_claimreextract820_plus_negaligned377_vlm120_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_weak_next1000_after1500_pilot300_fewerclaims194_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_vlm120_plusweak194_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_vlm120_plusweak194_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_vlm120_plusweak194_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_claimreextract820_plus_negaligned377_vlm120_plusweak194_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `1389`
- `label_observed_rows`: `526`
- `contrastive_rows`: `304`
- `repair_rows`: `863`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 863, '1': 355, '0': 171}`
- `sample_role`: `{'repair_unlabeled': 604, 'supervised_main': 322, 'lowinfo_unlabeled': 259, 'supervised_silver_ambiguous': 34, 'supervised_silver_evidence_incomplete': 99, 'supervised_silver_repair_needed': 56, 'supervised_silver_guarded': 15}`
- `promotion_state`: `{'repair_missing_claim': 414, 'main_positive_refute': 202, 'lowinfo_no_aligned_comment': 242, 'repair_missing_evidence': 82, 'repair_insufficient_product_evidence': 152, 'silver_conflicting_comment_relation': 32, 'silver_refute_insufficient_product_evidence': 55, 'silver_refute_missing_product_evidence': 44, 'repair_numeric_value_judgment': 6, 'silver_subjective_eval_attribute': 11, 'silver_schema_meta_attribute': 10, 'main_negative_support': 120, 'repair_identity_claim_value': 8, 'silver_commercial_promise_attribute': 7, 'silver_attribute_semantic_drift': 1, 'silver_enumeration_evidence_extra_values': 3}`
- `claim_found`: `{False: 414, True: 975}`
- `product_evidence_found`: `{False: 366, True: 1023}`
- `category_observed`: `{'apparel_and_underwear': 74, 'baby_kids_and_pets': 96, 'beauty_and_personal_care': 22, 'digital_and_electronics': 18, 'food_and_beverages': 97, 'general': 98, 'jewelry_and_collectibles': 8, 'shoes_and_bags': 40, 'smart_home': 35, 'sports_and_outdoor': 38}`
- `duplicate_claim_family_groups`: `65`
- `conflicting_claim_family_groups`: `4`
- `contrastive_masked_by_claim_family`: `18`
- `split`: `{'train': 367, 'val': 54, 'test': 105}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
