# Full Pair Promoted Dataset v1

This is the promotion report for LLM/VLM full-pair reconstruction reviews.
The main candidate is conservative; stateful rows preserve all reviewed hard cases.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_lownoise_next60_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise_next60_noimg_v1_20260614.jsonl`

## Outputs

- stateful reviewed rows: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_lownoise_next60_stateful_v1_20260614.jsonl`
- main supervised candidate: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_lownoise_next60_main_v1_20260614.jsonl`
- repair/silver rows: `data/final/repaired_v1/full_pair_claimreextract_next200_lownoise_next60_repair_silver_v1_20260614.jsonl`
- report json: `data/final/repaired_v1/full_pair_claimreextract_next200_lownoise_next60_promotion_v1_20260614.report.json`

## Summary

- `reviewed_rows`: `60`
- `main_rows`: `16`
- `missing_reviews`: `0`
- `all_labels`: `{0: 48, 1: 12}`
- `main_labels`: `{1: 12, 0: 4}`
- `promotion_state`: `{'repair_missing_claim': 14, 'main_positive_refute': 12, 'lowinfo_no_aligned_comment': 5, 'silver_refute_missing_product_evidence': 2, 'main_negative_support': 4, 'repair_numeric_value_judgment': 1, 'repair_identity_claim_value': 2, 'silver_conflicting_comment_relation': 7, 'repair_missing_evidence': 2, 'silver_refute_insufficient_product_evidence': 8, 'repair_insufficient_product_evidence': 3}`
- `confidence`: `{'high': 60}`
- `main_split`: `{'val': 2, 'train': 11, 'test': 3}`
- `main_split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
- `category`: `{'food_and_beverages': 9, 'general': 2, 'shoes_and_bags': 2, 'smart_home': 2, 'beauty_and_personal_care': 1}`
- `duplicate_claim_family_groups`: `0`
- `conflicting_claim_family_groups`: `0`
- `duplicate_claim_family_demoted`: `0`
- `duplicate_claim_family_examples`: `[]`
- `queue`: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_lownoise_next60_joint_review_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise_next60_noimg_v1_20260614.jsonl`
- `out_all`: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_lownoise_next60_stateful_v1_20260614.jsonl`
- `out_main`: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_lownoise_next60_main_v1_20260614.jsonl`
- `out_repair`: `data/final/repaired_v1/full_pair_claimreextract_next200_lownoise_next60_repair_silver_v1_20260614.jsonl`

## Promotion Rule

- `main_positive_refute`: claim found, product evidence found, and at least one aligned consumer comment refutes the same claim.
- `main_negative_support`: claim found, product evidence found, and aligned consumer comments support rather than refute the claim.
- Missing claim, missing/insufficient product evidence, mixed comments, and no aligned comments remain in stateful repair/silver outputs.
