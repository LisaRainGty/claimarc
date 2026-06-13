# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_lownoise60_joint_review_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise60_noimg_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_lownoise60_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_lownoise60_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_lownoise60_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_lownoise60_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `60`
- `label_observed_rows`: `32`
- `contrastive_rows`: `8`
- `repair_rows`: `28`
- `missing_queue_rows`: `0`
- `y_perception`: `{'1': 31, 'None': 28, '0': 1}`
- `sample_role`: `{'supervised_silver_ambiguous': 7, 'repair_unlabeled': 23, 'supervised_main': 8, 'supervised_silver_evidence_incomplete': 16, 'supervised_silver_repair_needed': 1, 'lowinfo_unlabeled': 5}`
- `promotion_state`: `{'silver_conflicting_comment_relation': 7, 'repair_missing_evidence': 2, 'repair_missing_claim': 19, 'main_positive_refute': 8, 'silver_refute_missing_product_evidence': 9, 'silver_refute_insufficient_product_evidence': 7, 'repair_insufficient_product_evidence': 3, 'lowinfo_no_aligned_comment': 5}`
- `claim_found`: `{True: 41, False: 19}`
- `product_evidence_found`: `{True: 48, False: 12}`
- `category_observed`: `{'apparel_and_underwear': 2, 'baby_kids_and_pets': 6, 'beauty_and_personal_care': 4, 'digital_and_electronics': 1, 'food_and_beverages': 5, 'general': 12, 'smart_home': 1, 'sports_and_outdoor': 1}`
- `duplicate_claim_family_groups`: `1`
- `conflicting_claim_family_groups`: `0`
- `contrastive_masked_by_claim_family`: `0`
- `split`: `{'train': 22, 'test': 6, 'val': 4}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
