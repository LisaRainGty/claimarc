# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_lownoise_next60_joint_review_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise_next60_noimg_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_lownoise_next60_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_lownoise_next60_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_lownoise_next60_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_lownoise_next60_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `60`
- `label_observed_rows`: `35`
- `contrastive_rows`: `16`
- `repair_rows`: `25`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 25, '1': 30, '0': 5}`
- `sample_role`: `{'repair_unlabeled': 20, 'supervised_main': 16, 'lowinfo_unlabeled': 5, 'supervised_silver_evidence_incomplete': 10, 'supervised_silver_repair_needed': 2, 'supervised_silver_ambiguous': 7}`
- `promotion_state`: `{'repair_missing_claim': 14, 'main_positive_refute': 12, 'lowinfo_no_aligned_comment': 5, 'silver_refute_missing_product_evidence': 2, 'main_negative_support': 4, 'repair_numeric_value_judgment': 1, 'repair_identity_claim_value': 2, 'silver_conflicting_comment_relation': 7, 'repair_missing_evidence': 2, 'silver_refute_insufficient_product_evidence': 8, 'repair_insufficient_product_evidence': 3}`
- `claim_found`: `{False: 14, True: 46}`
- `product_evidence_found`: `{True: 55, False: 5}`
- `category_observed`: `{'food_and_beverages': 17, 'general': 7, 'shoes_and_bags': 3, 'smart_home': 2, 'baby_kids_and_pets': 2, 'beauty_and_personal_care': 3, 'apparel_and_underwear': 1}`
- `duplicate_claim_family_groups`: `1`
- `conflicting_claim_family_groups`: `0`
- `contrastive_masked_by_claim_family`: `0`
- `split`: `{'train': 24, 'val': 4, 'test': 7}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
