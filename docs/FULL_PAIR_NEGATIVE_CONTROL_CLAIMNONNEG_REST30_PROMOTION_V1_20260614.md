# Full Pair Promoted Dataset v1

This is the promotion report for LLM/VLM full-pair reconstruction reviews.
The main candidate is conservative; stateful rows preserve all reviewed hard cases.

## Inputs

- queue: `data/final/repaired_v1/full_pair_negative_control_queue_claimnonneg_rest30_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg_rest30_noimg_v1_20260614.jsonl`

## Outputs

- stateful reviewed rows: `data/final/repaired_v1/dataset_full_pair_negative_control_claimnonneg_rest30_stateful_v1_20260614.jsonl`
- main supervised candidate: `data/final/repaired_v1/dataset_full_pair_negative_control_claimnonneg_rest30_main_v1_20260614.jsonl`
- repair/silver rows: `data/final/repaired_v1/full_pair_negative_control_claimnonneg_rest30_repair_silver_v1_20260614.jsonl`
- report json: `data/final/repaired_v1/full_pair_negative_control_claimnonneg_rest30_promotion_v1_20260614.report.json`

## Summary

- `reviewed_rows`: `30`
- `main_rows`: `10`
- `missing_reviews`: `0`
- `all_labels`: `{0: 27, 1: 3}`
- `main_labels`: `{0: 7, 1: 3}`
- `promotion_state`: `{'main_negative_support': 7, 'repair_insufficient_product_evidence': 3, 'main_positive_refute': 3, 'lowinfo_no_aligned_comment': 10, 'repair_missing_claim': 6, 'repair_missing_evidence': 1}`
- `confidence`: `{'high': 30}`
- `main_split`: `{'train': 7, 'test': 2, 'val': 1}`
- `main_split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
- `category`: `{'digital_and_electronics': 2, 'shoes_and_bags': 3, 'apparel_and_underwear': 3, 'general': 1, 'beauty_and_personal_care': 1}`
- `duplicate_claim_family_groups`: `0`
- `conflicting_claim_family_groups`: `0`
- `duplicate_claim_family_demoted`: `0`
- `duplicate_claim_family_examples`: `[]`
- `queue`: `data/final/repaired_v1/full_pair_negative_control_queue_claimnonneg_rest30_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg_rest30_noimg_v1_20260614.jsonl`
- `out_all`: `data/final/repaired_v1/dataset_full_pair_negative_control_claimnonneg_rest30_stateful_v1_20260614.jsonl`
- `out_main`: `data/final/repaired_v1/dataset_full_pair_negative_control_claimnonneg_rest30_main_v1_20260614.jsonl`
- `out_repair`: `data/final/repaired_v1/full_pair_negative_control_claimnonneg_rest30_repair_silver_v1_20260614.jsonl`

## Promotion Rule

- `main_positive_refute`: claim found, product evidence found, and at least one aligned consumer comment refutes the same claim.
- `main_negative_support`: claim found, product evidence found, and aligned consumer comments support rather than refute the claim.
- Missing claim, missing/insufficient product evidence, mixed comments, and no aligned comments remain in stateful repair/silver outputs.
