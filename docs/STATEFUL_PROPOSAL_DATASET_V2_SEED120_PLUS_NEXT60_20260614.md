# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise60_noimg_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_plus_next60_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_plus_next60_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_plus_next60_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_plus_next60_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `159`
- `label_observed_rows`: `89`
- `contrastive_rows`: `28`
- `repair_rows`: `70`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 70, '0': 9, '1': 80}`
- `sample_role`: `{'repair_unlabeled': 60, 'supervised_silver_guarded': 5, 'supervised_silver_evidence_incomplete': 31, 'supervised_main': 30, 'supervised_silver_ambiguous': 15, 'supervised_silver_repair_needed': 8, 'lowinfo_unlabeled': 10}`
- `promotion_state`: `{'repair_missing_claim': 50, 'silver_attribute_semantic_drift': 3, 'silver_refute_insufficient_product_evidence': 14, 'main_positive_refute': 27, 'silver_conflicting_comment_relation': 12, 'repair_identity_claim_value': 4, 'silver_refute_missing_product_evidence': 17, 'repair_numeric_value_judgment': 2, 'repair_insufficient_product_evidence': 9, 'lowinfo_no_aligned_comment': 9, 'silver_commercial_promise_attribute': 1, 'main_negative_support': 3, 'repair_missing_evidence': 4, 'silver_consumer_expectation_mismatch': 1, 'silver_subjective_eval_attribute': 2, 'silver_price_value_not_direct_refute': 1}`
- `claim_found`: `{False: 50, True: 109}`
- `product_evidence_found`: `{True: 135, False: 24}`
- `category_observed`: `{'baby_kids_and_pets': 21, 'beauty_and_personal_care': 7, 'food_and_beverages': 23, 'general': 21, 'shoes_and_bags': 3, 'smart_home': 6, 'digital_and_electronics': 2, 'apparel_and_underwear': 3, 'sports_and_outdoor': 3}`
- `duplicate_claim_family_groups`: `5`
- `conflicting_claim_family_groups`: `1`
- `contrastive_masked_by_claim_family`: `2`
- `split`: `{'val': 13, 'train': 59, 'test': 17}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
