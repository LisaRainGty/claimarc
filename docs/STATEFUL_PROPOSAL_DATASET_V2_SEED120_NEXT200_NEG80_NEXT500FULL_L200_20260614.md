# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise60_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg50_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise_next60_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg_rest30_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_rest45_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next500_partial151_lownoise60_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next500_partial259_lownoise_next60_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next500_full_lownoise_next80_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_next500full_l200_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_next500full_l200_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_next500full_l200_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_next500full_l200_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `544`
- `label_observed_rows`: `280`
- `contrastive_rows`: `112`
- `repair_rows`: `264`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 264, '0': 62, '1': 218}`
- `sample_role`: `{'repair_unlabeled': 197, 'supervised_silver_guarded': 6, 'supervised_silver_evidence_incomplete': 83, 'supervised_main': 126, 'supervised_silver_ambiguous': 41, 'supervised_silver_repair_needed': 24, 'lowinfo_unlabeled': 67}`
- `promotion_state`: `{'repair_missing_claim': 153, 'silver_attribute_semantic_drift': 4, 'silver_refute_insufficient_product_evidence': 41, 'main_positive_refute': 84, 'silver_conflicting_comment_relation': 36, 'repair_identity_claim_value': 8, 'silver_refute_missing_product_evidence': 42, 'repair_numeric_value_judgment': 6, 'repair_insufficient_product_evidence': 36, 'lowinfo_no_aligned_comment': 66, 'silver_commercial_promise_attribute': 2, 'main_negative_support': 42, 'repair_missing_evidence': 20, 'silver_consumer_expectation_mismatch': 1, 'silver_subjective_eval_attribute': 2, 'silver_price_value_not_direct_refute': 1}`
- `claim_found`: `{False: 153, True: 391}`
- `product_evidence_found`: `{True: 451, False: 93}`
- `category_observed`: `{'baby_kids_and_pets': 59, 'beauty_and_personal_care': 23, 'food_and_beverages': 64, 'general': 52, 'shoes_and_bags': 18, 'smart_home': 21, 'digital_and_electronics': 11, 'apparel_and_underwear': 21, 'sports_and_outdoor': 11}`
- `duplicate_claim_family_groups`: `33`
- `conflicting_claim_family_groups`: `7`
- `contrastive_masked_by_claim_family`: `14`
- `split`: `{'val': 29, 'train': 195, 'test': 56}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
