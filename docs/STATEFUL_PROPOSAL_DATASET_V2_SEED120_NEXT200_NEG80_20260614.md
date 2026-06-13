# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_lownoise_next40_silver11_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_rest19_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise60_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg50_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise_next60_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg_rest30_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_rest45_noimg_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `344`
- `label_observed_rows`: `185`
- `contrastive_rows`: `80`
- `repair_rows`: `159`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 159, '0': 43, '1': 142}`
- `sample_role`: `{'repair_unlabeled': 117, 'supervised_silver_guarded': 5, 'supervised_silver_evidence_incomplete': 48, 'supervised_main': 85, 'supervised_silver_ambiguous': 29, 'supervised_silver_repair_needed': 18, 'lowinfo_unlabeled': 42}`
- `promotion_state`: `{'repair_missing_claim': 88, 'silver_attribute_semantic_drift': 3, 'silver_refute_insufficient_product_evidence': 26, 'main_positive_refute': 56, 'silver_conflicting_comment_relation': 26, 'repair_identity_claim_value': 7, 'silver_refute_missing_product_evidence': 22, 'repair_numeric_value_judgment': 4, 'repair_insufficient_product_evidence': 26, 'lowinfo_no_aligned_comment': 41, 'silver_commercial_promise_attribute': 1, 'main_negative_support': 29, 'repair_missing_evidence': 11, 'silver_consumer_expectation_mismatch': 1, 'silver_subjective_eval_attribute': 2, 'silver_price_value_not_direct_refute': 1}`
- `claim_found`: `{False: 88, True: 256}`
- `product_evidence_found`: `{True: 305, False: 39}`
- `category_observed`: `{'baby_kids_and_pets': 33, 'beauty_and_personal_care': 19, 'food_and_beverages': 52, 'general': 33, 'shoes_and_bags': 15, 'smart_home': 9, 'digital_and_electronics': 9, 'apparel_and_underwear': 11, 'sports_and_outdoor': 4}`
- `duplicate_claim_family_groups`: `13`
- `conflicting_claim_family_groups`: `1`
- `contrastive_masked_by_claim_family`: `5`
- `split`: `{'val': 18, 'train': 130, 'test': 37}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
