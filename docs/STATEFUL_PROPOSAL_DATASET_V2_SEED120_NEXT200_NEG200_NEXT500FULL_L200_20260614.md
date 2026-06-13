# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise60_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg50_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise_next60_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg_rest30_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_rest45_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next500_partial151_lownoise60_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next500_partial259_lownoise_next60_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next500_full_lownoise_next80_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg_next120_noimg_flash_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg200_next500full_l200_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg200_next500full_l200_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg200_next500full_l200_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg200_next500full_l200_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `664`
- `label_observed_rows`: `327`
- `contrastive_rows`: `146`
- `repair_rows`: `337`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 337, '0': 104, '1': 223}`
- `sample_role`: `{'repair_unlabeled': 230, 'supervised_silver_guarded': 6, 'supervised_silver_evidence_incomplete': 84, 'supervised_main': 160, 'supervised_silver_ambiguous': 42, 'supervised_silver_repair_needed': 35, 'lowinfo_unlabeled': 107}`
- `promotion_state`: `{'repair_missing_claim': 170, 'silver_attribute_semantic_drift': 4, 'silver_refute_insufficient_product_evidence': 41, 'main_positive_refute': 87, 'silver_conflicting_comment_relation': 37, 'repair_identity_claim_value': 8, 'silver_refute_missing_product_evidence': 43, 'repair_numeric_value_judgment': 6, 'repair_insufficient_product_evidence': 58, 'lowinfo_no_aligned_comment': 106, 'silver_commercial_promise_attribute': 2, 'main_negative_support': 73, 'repair_missing_evidence': 25, 'silver_consumer_expectation_mismatch': 1, 'silver_subjective_eval_attribute': 2, 'silver_price_value_not_direct_refute': 1}`
- `claim_found`: `{False: 170, True: 494}`
- `product_evidence_found`: `{True: 562, False: 102}`
- `category_observed`: `{'baby_kids_and_pets': 75, 'beauty_and_personal_care': 26, 'food_and_beverages': 65, 'general': 58, 'shoes_and_bags': 24, 'smart_home': 21, 'digital_and_electronics': 12, 'apparel_and_underwear': 33, 'sports_and_outdoor': 13}`
- `duplicate_claim_family_groups`: `34`
- `conflicting_claim_family_groups`: `7`
- `contrastive_masked_by_claim_family`: `14`
- `split`: `{'val': 33, 'test': 65, 'train': 229}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
