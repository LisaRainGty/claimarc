# Full Pair Promoted Dataset v1

This is the promotion report for LLM/VLM full-pair reconstruction reviews.
The main candidate is conservative; stateful rows preserve all reviewed hard cases.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_lownoise60_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise60_noimg_v1_20260614.jsonl`

## Outputs

- stateful reviewed rows: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_lownoise60_stateful_v1_20260614.jsonl`
- main supervised candidate: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_lownoise60_main_v1_20260614.jsonl`
- repair/silver rows: `data/final/repaired_v1/full_pair_claimreextract_next200_lownoise60_repair_silver_v1_20260614.jsonl`
- report json: `data/final/repaired_v1/full_pair_claimreextract_next200_lownoise60_promotion_v1_20260614.report.json`

## Summary

- `reviewed_rows`: `60`
- `main_rows`: `8`
- `missing_reviews`: `0`
- `all_labels`: `{0: 52, 1: 8}`
- `main_labels`: `{1: 8}`
- `promotion_state`: `{'silver_conflicting_comment_relation': 7, 'repair_missing_evidence': 2, 'repair_missing_claim': 19, 'main_positive_refute': 8, 'silver_refute_missing_product_evidence': 9, 'silver_refute_insufficient_product_evidence': 7, 'repair_insufficient_product_evidence': 3, 'lowinfo_no_aligned_comment': 5}`
- `confidence`: `{'high': 59, 'medium': 1}`
- `main_split`: `{'train': 5, 'val': 1, 'test': 2}`
- `main_split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
- `category`: `{'baby_kids_and_pets': 4, 'general': 3, 'beauty_and_personal_care': 1}`
- `duplicate_claim_family_groups`: `0`
- `conflicting_claim_family_groups`: `0`
- `duplicate_claim_family_demoted`: `0`
- `duplicate_claim_family_examples`: `[]`
- `queue`: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_lownoise60_joint_review_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak_next200_lownoise60_noimg_v1_20260614.jsonl`
- `out_all`: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_lownoise60_stateful_v1_20260614.jsonl`
- `out_main`: `data/final/repaired_v1/dataset_full_pair_claimreextract_next200_lownoise60_main_v1_20260614.jsonl`
- `out_repair`: `data/final/repaired_v1/full_pair_claimreextract_next200_lownoise60_repair_silver_v1_20260614.jsonl`

## Promotion Rule

- `main_positive_refute`: claim found, product evidence found, and at least one aligned consumer comment refutes the same claim.
- `main_negative_support`: claim found, product evidence found, and aligned consumer comments support rather than refute the claim.
- Missing claim, missing/insufficient product evidence, mixed comments, and no aligned comments remain in stateful repair/silver outputs.
