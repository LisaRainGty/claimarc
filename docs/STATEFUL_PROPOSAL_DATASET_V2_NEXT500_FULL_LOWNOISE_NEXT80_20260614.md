# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_full_lownoise_next80_joint_review_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next500_full_lownoise_next80_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_full_lownoise_next80_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_full_lownoise_next80_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_full_lownoise_next80_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_full_lownoise_next80_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `80`
- `label_observed_rows`: `34`
- `contrastive_rows`: `18`
- `repair_rows`: `46`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 46, '0': 10, '1': 24}`
- `sample_role`: `{'lowinfo_unlabeled': 10, 'repair_unlabeled': 36, 'supervised_main': 19, 'supervised_silver_evidence_incomplete': 10, 'supervised_silver_guarded': 1, 'supervised_silver_ambiguous': 1, 'supervised_silver_repair_needed': 3}`
- `promotion_state`: `{'lowinfo_no_aligned_comment': 10, 'repair_missing_evidence': 4, 'main_negative_support': 7, 'repair_missing_claim': 27, 'repair_insufficient_product_evidence': 7, 'main_positive_refute': 12, 'silver_refute_insufficient_product_evidence': 5, 'silver_refute_missing_product_evidence': 5, 'silver_commercial_promise_attribute': 1, 'silver_conflicting_comment_relation': 1, 'repair_numeric_value_judgment': 1}`
- `claim_found`: `{True: 53, False: 27}`
- `product_evidence_found`: `{True: 58, False: 22}`
- `category_observed`: `{'apparel_and_underwear': 3, 'baby_kids_and_pets': 6, 'digital_and_electronics': 1, 'general': 12, 'shoes_and_bags': 1, 'smart_home': 7, 'sports_and_outdoor': 3, 'beauty_and_personal_care': 1}`
- `duplicate_claim_family_groups`: `3`
- `conflicting_claim_family_groups`: `1`
- `contrastive_masked_by_claim_family`: `1`
- `split`: `{'train': 24, 'test': 7, 'val': 3}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
