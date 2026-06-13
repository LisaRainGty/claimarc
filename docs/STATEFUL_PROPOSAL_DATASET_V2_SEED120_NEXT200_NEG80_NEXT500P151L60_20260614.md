# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise60_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg50_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise_next60_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg_rest30_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_rest45_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next500_partial151_lownoise60_noimg_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_next500p151l60_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_next500p151l60_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_next500p151l60_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_next500p151l60_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `404`
- `label_observed_rows`: `221`
- `contrastive_rows`: `88`
- `repair_rows`: `183`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 183, '0': 51, '1': 170}`
- `sample_role`: `{'repair_unlabeled': 136, 'supervised_silver_guarded': 5, 'supervised_silver_evidence_incomplete': 63, 'supervised_main': 95, 'supervised_silver_ambiguous': 37, 'supervised_silver_repair_needed': 21, 'lowinfo_unlabeled': 47}`
- `promotion_state`: `{'repair_missing_claim': 104, 'silver_attribute_semantic_drift': 4, 'silver_refute_insufficient_product_evidence': 32, 'main_positive_refute': 61, 'silver_conflicting_comment_relation': 33, 'repair_identity_claim_value': 8, 'silver_refute_missing_product_evidence': 31, 'repair_numeric_value_judgment': 4, 'repair_insufficient_product_evidence': 27, 'lowinfo_no_aligned_comment': 46, 'silver_commercial_promise_attribute': 1, 'main_negative_support': 34, 'repair_missing_evidence': 15, 'silver_consumer_expectation_mismatch': 1, 'silver_subjective_eval_attribute': 2, 'silver_price_value_not_direct_refute': 1}`
- `claim_found`: `{False: 104, True: 300}`
- `product_evidence_found`: `{True: 351, False: 53}`
- `category_observed`: `{'baby_kids_and_pets': 45, 'beauty_and_personal_care': 21, 'food_and_beverages': 58, 'general': 37, 'shoes_and_bags': 17, 'smart_home': 13, 'digital_and_electronics': 10, 'apparel_and_underwear': 14, 'sports_and_outdoor': 6}`
- `duplicate_claim_family_groups`: `20`
- `conflicting_claim_family_groups`: `3`
- `contrastive_masked_by_claim_family`: `7`
- `split`: `{'val': 23, 'test': 44, 'train': 154}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
