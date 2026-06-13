# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `99`
- `label_observed_rows`: `57`
- `contrastive_rows`: `20`
- `repair_rows`: `42`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 42, '0': 8, '1': 49}`
- `sample_role`: `{'repair_unlabeled': 37, 'supervised_silver_guarded': 5, 'supervised_silver_evidence_incomplete': 15, 'supervised_main': 22, 'supervised_silver_ambiguous': 8, 'supervised_silver_repair_needed': 7, 'lowinfo_unlabeled': 5}`
- `promotion_state`: `{'repair_missing_claim': 31, 'silver_attribute_semantic_drift': 3, 'silver_refute_insufficient_product_evidence': 7, 'main_positive_refute': 19, 'silver_conflicting_comment_relation': 5, 'repair_identity_claim_value': 4, 'silver_refute_missing_product_evidence': 8, 'repair_numeric_value_judgment': 2, 'repair_insufficient_product_evidence': 6, 'lowinfo_no_aligned_comment': 4, 'silver_commercial_promise_attribute': 1, 'main_negative_support': 3, 'repair_missing_evidence': 2, 'silver_consumer_expectation_mismatch': 1, 'silver_subjective_eval_attribute': 2, 'silver_price_value_not_direct_refute': 1}`
- `claim_found`: `{False: 31, True: 68}`
- `product_evidence_found`: `{True: 87, False: 12}`
- `category_observed`: `{'baby_kids_and_pets': 15, 'beauty_and_personal_care': 3, 'food_and_beverages': 18, 'general': 9, 'shoes_and_bags': 3, 'smart_home': 5, 'digital_and_electronics': 1, 'apparel_and_underwear': 1, 'sports_and_outdoor': 2}`
- `duplicate_claim_family_groups`: `4`
- `conflicting_claim_family_groups`: `1`
- `contrastive_masked_by_claim_family`: `2`
- `split`: `{'test': 12, 'train': 37, 'val': 8}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
