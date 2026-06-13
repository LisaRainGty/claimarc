# Full Pair Promoted Dataset v1

This is the promotion report for LLM/VLM full-pair reconstruction reviews.
The main candidate is conservative; stateful rows preserve all reviewed hard cases.

## Inputs

- queue: `data/final/repaired_v1/full_pair_negative_control_queue_claimnonneg80_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg50_noimg_v1_20260614.jsonl`

## Outputs

- stateful reviewed rows: `data/final/repaired_v1/dataset_full_pair_negative_control_claimnonneg50_stateful_v1_20260614.jsonl`
- main supervised candidate: `data/final/repaired_v1/dataset_full_pair_negative_control_claimnonneg50_main_v1_20260614.jsonl`
- repair/silver rows: `data/final/repaired_v1/full_pair_negative_control_claimnonneg50_repair_silver_v1_20260614.jsonl`
- report json: `data/final/repaired_v1/full_pair_negative_control_claimnonneg50_promotion_v1_20260614.report.json`

## Summary

- `reviewed_rows`: `50`
- `main_rows`: `17`
- `missing_reviews`: `30`
- `all_labels`: `{0: 46, 1: 4}`
- `main_labels`: `{1: 4, 0: 13}`
- `promotion_state`: `{'lowinfo_no_aligned_comment': 12, 'silver_conflicting_comment_relation': 4, 'silver_refute_insufficient_product_evidence': 1, 'repair_insufficient_product_evidence': 9, 'main_positive_refute': 4, 'repair_identity_claim_value': 1, 'main_negative_support': 13, 'repair_missing_claim': 5, 'repair_missing_evidence': 1}`
- `confidence`: `{'high': 50}`
- `main_split`: `{'test': 3, 'val': 2, 'train': 12}`
- `main_split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
- `category`: `{'beauty_and_personal_care': 5, 'shoes_and_bags': 3, 'baby_kids_and_pets': 2, 'digital_and_electronics': 3, 'general': 2, 'apparel_and_underwear': 2}`
- `duplicate_claim_family_groups`: `0`
- `conflicting_claim_family_groups`: `0`
- `duplicate_claim_family_demoted`: `0`
- `duplicate_claim_family_examples`: `[]`
- `queue`: `data/final/repaired_v1/full_pair_negative_control_queue_claimnonneg80_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_claimnonneg50_noimg_v1_20260614.jsonl`
- `out_all`: `data/final/repaired_v1/dataset_full_pair_negative_control_claimnonneg50_stateful_v1_20260614.jsonl`
- `out_main`: `data/final/repaired_v1/dataset_full_pair_negative_control_claimnonneg50_main_v1_20260614.jsonl`
- `out_repair`: `data/final/repaired_v1/full_pair_negative_control_claimnonneg50_repair_silver_v1_20260614.jsonl`

## Promotion Rule

- `main_positive_refute`: claim found, product evidence found, and at least one aligned consumer comment refutes the same claim.
- `main_negative_support`: claim found, product evidence found, and aligned consumer comments support rather than refute the claim.
- Missing claim, missing/insufficient product evidence, mixed comments, and no aligned comments remain in stateful repair/silver outputs.
