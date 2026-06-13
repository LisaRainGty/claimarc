# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_pilot40_joint_review_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_pilot40_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next1000_after500_pilot40_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next1000_after500_pilot40_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next1000_after500_pilot40_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_next1000_after500_pilot40_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `35`
- `label_observed_rows`: `18`
- `contrastive_rows`: `6`
- `repair_rows`: `17`
- `missing_queue_rows`: `0`
- `y_perception`: `{'1': 15, 'None': 17, '0': 3}`
- `sample_role`: `{'supervised_main': 6, 'supervised_silver_evidence_incomplete': 6, 'supervised_silver_ambiguous': 4, 'lowinfo_unlabeled': 6, 'repair_unlabeled': 11, 'supervised_silver_repair_needed': 2}`
- `promotion_state`: `{'main_positive_refute': 5, 'silver_refute_insufficient_product_evidence': 5, 'silver_conflicting_comment_relation': 4, 'silver_commercial_promise_attribute': 1, 'lowinfo_no_aligned_comment': 5, 'repair_missing_claim': 6, 'repair_insufficient_product_evidence': 6, 'repair_missing_evidence': 1, 'silver_refute_missing_product_evidence': 1, 'main_negative_support': 1}`
- `claim_found`: `{True: 29, False: 6}`
- `product_evidence_found`: `{True: 32, False: 3}`
- `category_observed`: `{'baby_kids_and_pets': 8, 'food_and_beverages': 9, 'beauty_and_personal_care': 1}`
- `duplicate_claim_family_groups`: `1`
- `conflicting_claim_family_groups`: `1`
- `contrastive_masked_by_claim_family`: `0`
- `split`: `{'train': 12, 'test': 4, 'val': 2}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
