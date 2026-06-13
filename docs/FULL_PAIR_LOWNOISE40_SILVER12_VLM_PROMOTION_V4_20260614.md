# Full Pair Promoted Dataset v1

This is the promotion report for LLM/VLM full-pair reconstruction reviews.
The main candidate is conservative; stateful rows preserve all reviewed hard cases.

## Inputs

- queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_lownoise40_silver12_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_v1_20260614.jsonl`

## Outputs

- stateful reviewed rows: `data/final/repaired_v1/dataset_full_pair_lownoise40_silver12_vlm_stateful_v4_20260614.jsonl`
- main supervised candidate: `data/final/repaired_v1/dataset_full_pair_lownoise40_silver12_vlm_main_v4_20260614.jsonl`
- repair/silver rows: `data/final/repaired_v1/full_pair_lownoise40_silver12_vlm_repair_silver_v4_20260614.jsonl`
- report json: `data/final/repaired_v1/full_pair_lownoise40_silver12_vlm_promotion_v4_20260614.report.json`

## Summary

- `reviewed_rows`: `12`
- `main_rows`: `1`
- `missing_reviews`: `0`
- `all_labels`: `{0: 11, 1: 1}`
- `main_labels`: `{1: 1}`
- `promotion_state`: `{'repair_missing_evidence': 1, 'silver_refute_missing_product_evidence': 2, 'repair_missing_claim': 2, 'main_positive_refute': 1, 'silver_refute_insufficient_product_evidence': 3, 'repair_insufficient_product_evidence': 1, 'silver_conflicting_comment_relation': 1, 'silver_duplicate_claim_family': 1}`
- `confidence`: `{'high': 12}`
- `main_split`: `{'train': 1}`
- `main_split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
- `category`: `{'smart_home': 1}`
- `duplicate_claim_family_groups`: `1`
- `conflicting_claim_family_groups`: `0`
- `duplicate_claim_family_demoted`: `1`
- `duplicate_claim_family_examples`: `[{'demoted_pair_id': 'p3663574409718972261__HOME_描述', 'kept_pair_id': 'p3663574409718972261__HOME_功效', 'attribute_name': '描述', 'kept_attribute_name': '功效', 'state_before': 'main_positive_refute', 'reason': 'duplicate_same_label'}]`
- `queue`: `data/final/repaired_v1/full_pair_evidence_repair_queue_lownoise40_silver12_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_lownoise40_silver12_vlm_v1_20260614.jsonl`
- `out_all`: `data/final/repaired_v1/dataset_full_pair_lownoise40_silver12_vlm_stateful_v4_20260614.jsonl`
- `out_main`: `data/final/repaired_v1/dataset_full_pair_lownoise40_silver12_vlm_main_v4_20260614.jsonl`
- `out_repair`: `data/final/repaired_v1/full_pair_lownoise40_silver12_vlm_repair_silver_v4_20260614.jsonl`

## Promotion Rule

- `main_positive_refute`: claim found, product evidence found, and at least one aligned consumer comment refutes the same claim.
- `main_negative_support`: claim found, product evidence found, and aligned consumer comments support rather than refute the claim.
- Missing claim, missing/insufficient product evidence, mixed comments, and no aligned comments remain in stateful repair/silver outputs.
