# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_fewerclaims_joint_review_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims300_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next1000_after500_fewerclaims300_noimg_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next1000_after500_fewerclaims300_noimg_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next1000_after500_fewerclaims300_noimg_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next1000_after500_fewerclaims300_noimg_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `300`
- `label_observed_rows`: `129`
- `contrastive_rows`: `48`
- `repair_rows`: `171`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 171, '1': 124, '0': 5}`
- `sample_role`: `{'repair_unlabeled': 137, 'supervised_main': 50, 'lowinfo_unlabeled': 34, 'supervised_silver_evidence_incomplete': 60, 'supervised_silver_ambiguous': 7, 'supervised_silver_repair_needed': 4, 'supervised_silver_guarded': 8}`
- `promotion_state`: `{'repair_missing_claim': 88, 'main_positive_refute': 48, 'lowinfo_no_aligned_comment': 33, 'repair_missing_evidence': 31, 'silver_refute_insufficient_product_evidence': 25, 'silver_conflicting_comment_relation': 7, 'silver_refute_missing_product_evidence': 35, 'repair_insufficient_product_evidence': 19, 'repair_numeric_value_judgment': 1, 'silver_subjective_eval_attribute': 6, 'silver_schema_meta_attribute': 3, 'main_negative_support': 2, 'repair_identity_claim_value': 2}`
- `claim_found`: `{False: 88, True: 212}`
- `product_evidence_found`: `{False: 132, True: 168}`
- `category_observed`: `{'apparel_and_underwear': 7, 'baby_kids_and_pets': 18, 'beauty_and_personal_care': 3, 'digital_and_electronics': 2, 'food_and_beverages': 13, 'general': 43, 'jewelry_and_collectibles': 1, 'shoes_and_bags': 4, 'smart_home': 24, 'sports_and_outdoor': 14}`
- `duplicate_claim_family_groups`: `24`
- `conflicting_claim_family_groups`: `0`
- `contrastive_masked_by_claim_family`: `2`
- `split`: `{'train': 87, 'test': 26, 'val': 16}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
