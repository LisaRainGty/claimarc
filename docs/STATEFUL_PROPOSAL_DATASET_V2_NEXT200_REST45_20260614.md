# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_rest45_joint_review_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_rest45_noimg_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_rest45_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_rest45_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_rest45_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_rest45_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `45`
- `label_observed_rows`: `23`
- `contrastive_rows`: `12`
- `repair_rows`: `22`
- `missing_queue_rows`: `0`
- `y_perception`: `{'1': 20, 'None': 22, '0': 3}`
- `sample_role`: `{'supervised_silver_ambiguous': 3, 'repair_unlabeled': 17, 'lowinfo_unlabeled': 5, 'supervised_silver_evidence_incomplete': 6, 'supervised_main': 12, 'supervised_silver_repair_needed': 2}`
- `promotion_state`: `{'silver_conflicting_comment_relation': 3, 'repair_missing_claim': 13, 'lowinfo_no_aligned_comment': 5, 'repair_insufficient_product_evidence': 2, 'silver_refute_missing_product_evidence': 3, 'main_positive_refute': 10, 'main_negative_support': 2, 'repair_numeric_value_judgment': 1, 'repair_missing_evidence': 3, 'silver_refute_insufficient_product_evidence': 3}`
- `claim_found`: `{True: 32, False: 13}`
- `product_evidence_found`: `{True: 37, False: 8}`
- `category_observed`: `{'smart_home': 1, 'baby_kids_and_pets': 5, 'food_and_beverages': 11, 'general': 2, 'shoes_and_bags': 3, 'sports_and_outdoor': 1}`
- `duplicate_claim_family_groups`: `0`
- `conflicting_claim_family_groups`: `0`
- `contrastive_masked_by_claim_family`: `0`
- `split`: `{'train': 15, 'test': 5, 'val': 3}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
