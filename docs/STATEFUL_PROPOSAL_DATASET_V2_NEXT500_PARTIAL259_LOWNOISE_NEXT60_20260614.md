# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_partial259_lownoise_next60_joint_review_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next500_partial259_lownoise_next60_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_partial259_lownoise_next60_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_partial259_lownoise_next60_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_partial259_lownoise_next60_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_partial259_lownoise_next60_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `60`
- `label_observed_rows`: `25`
- `contrastive_rows`: `12`
- `repair_rows`: `35`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 35, '1': 24, '0': 1}`
- `sample_role`: `{'lowinfo_unlabeled': 10, 'supervised_main': 12, 'repair_unlabeled': 25, 'supervised_silver_evidence_incomplete': 10, 'supervised_silver_ambiguous': 3}`
- `promotion_state`: `{'lowinfo_no_aligned_comment': 10, 'main_positive_refute': 11, 'repair_missing_claim': 22, 'silver_refute_missing_product_evidence': 6, 'silver_refute_insufficient_product_evidence': 4, 'repair_insufficient_product_evidence': 2, 'silver_conflicting_comment_relation': 2, 'repair_missing_evidence': 1, 'main_negative_support': 1, 'repair_numeric_value_judgment': 1}`
- `claim_found`: `{True: 38, False: 22}`
- `product_evidence_found`: `{True: 42, False: 18}`
- `category_observed`: `{'apparel_and_underwear': 4, 'baby_kids_and_pets': 8, 'beauty_and_personal_care': 1, 'food_and_beverages': 6, 'general': 3, 'smart_home': 1, 'sports_and_outdoor': 2}`
- `duplicate_claim_family_groups`: `0`
- `conflicting_claim_family_groups`: `0`
- `contrastive_masked_by_claim_family`: `0`
- `split`: `{'test': 5, 'train': 18, 'val': 2}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
