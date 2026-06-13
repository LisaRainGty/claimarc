# Full Pair Promoted Dataset v1

This is the promotion report for LLM/VLM full-pair reconstruction reviews.
The main candidate is conservative; stateful rows preserve all reviewed hard cases.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_rest45_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_rest45_noimg_v1_20260614.jsonl`

## Outputs

- stateful reviewed rows: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_rest45_stateful_v1_20260614.jsonl`
- main supervised candidate: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_rest45_main_v1_20260614.jsonl`
- repair/silver rows: `data/final/repaired_v1/full_pair_claimreextract_next200_rest45_repair_silver_v1_20260614.jsonl`
- report json: `data/final/repaired_v1/full_pair_claimreextract_next200_rest45_promotion_v1_20260614.report.json`

## Summary

- `reviewed_rows`: `45`
- `main_rows`: `12`
- `missing_reviews`: `0`
- `all_labels`: `{0: 35, 1: 10}`
- `main_labels`: `{1: 10, 0: 2}`
- `promotion_state`: `{'silver_conflicting_comment_relation': 3, 'repair_missing_claim': 13, 'lowinfo_no_aligned_comment': 5, 'repair_insufficient_product_evidence': 2, 'silver_refute_missing_product_evidence': 3, 'main_positive_refute': 10, 'main_negative_support': 2, 'repair_numeric_value_judgment': 1, 'repair_missing_evidence': 3, 'silver_refute_insufficient_product_evidence': 3}`
- `confidence`: `{'high': 45}`
- `main_split`: `{'test': 3, 'train': 7, 'val': 2}`
- `main_split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
- `category`: `{'food_and_beverages': 8, 'baby_kids_and_pets': 3, 'shoes_and_bags': 1}`
- `duplicate_claim_family_groups`: `0`
- `conflicting_claim_family_groups`: `0`
- `duplicate_claim_family_demoted`: `0`
- `duplicate_claim_family_examples`: `[]`
- `queue`: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_rest45_joint_review_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_rest45_noimg_v1_20260614.jsonl`
- `out_all`: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_rest45_stateful_v1_20260614.jsonl`
- `out_main`: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_rest45_main_v1_20260614.jsonl`
- `out_repair`: `data/final/repaired_v1/full_pair_claimreextract_next200_rest45_repair_silver_v1_20260614.jsonl`

## Promotion Rule

- `main_positive_refute`: claim found, product evidence found, and at least one aligned consumer comment refutes the same claim.
- `main_negative_support`: claim found, product evidence found, and aligned consumer comments support rather than refute the claim.
- Missing claim, missing/insufficient product evidence, mixed comments, and no aligned comments remain in stateful repair/silver outputs.
