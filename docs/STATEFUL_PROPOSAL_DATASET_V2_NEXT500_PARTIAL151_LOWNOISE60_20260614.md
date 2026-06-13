# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_partial151_lownoise60_joint_review_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next500_partial151_lownoise60_noimg_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_partial151_lownoise60_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_partial151_lownoise60_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_partial151_lownoise60_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next500_partial151_lownoise60_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `60`
- `label_observed_rows`: `36`
- `contrastive_rows`: `10`
- `repair_rows`: `24`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 24, '1': 28, '0': 8}`
- `sample_role`: `{'repair_unlabeled': 19, 'supervised_silver_evidence_incomplete': 15, 'supervised_silver_ambiguous': 8, 'lowinfo_unlabeled': 5, 'supervised_main': 10, 'supervised_silver_repair_needed': 3}`
- `promotion_state`: `{'repair_missing_evidence': 4, 'repair_missing_claim': 16, 'silver_refute_missing_product_evidence': 9, 'silver_conflicting_comment_relation': 7, 'silver_refute_insufficient_product_evidence': 6, 'lowinfo_no_aligned_comment': 5, 'repair_insufficient_product_evidence': 1, 'main_negative_support': 5, 'silver_attribute_semantic_drift': 1, 'repair_identity_claim_value': 1, 'main_positive_refute': 5}`
- `claim_found`: `{True: 44, False: 16}`
- `product_evidence_found`: `{False: 14, True: 46}`
- `category_observed`: `{'apparel_and_underwear': 3, 'baby_kids_and_pets': 12, 'beauty_and_personal_care': 2, 'digital_and_electronics': 1, 'food_and_beverages': 6, 'general': 4, 'shoes_and_bags': 2, 'smart_home': 4, 'sports_and_outdoor': 2}`
- `duplicate_claim_family_groups`: `0`
- `conflicting_claim_family_groups`: `0`
- `contrastive_masked_by_claim_family`: `0`
- `split`: `{'train': 25, 'val': 4, 'test': 7}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
