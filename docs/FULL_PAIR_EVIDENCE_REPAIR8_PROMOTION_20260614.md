# Full Pair Promoted Dataset v1

This is the promotion report for LLM/VLM full-pair reconstruction reviews.
The main candidate is conservative; stateful rows preserve all reviewed hard cases.

## Inputs

- queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair8_vlm_v1_20260614.jsonl`

## Outputs

- stateful reviewed rows: `data/final/repaired_v1/dataset_full_pair_evidence_repair8_stateful_v1_20260614.jsonl`
- main supervised candidate: `data/final/repaired_v1/dataset_full_pair_evidence_repair8_main_v1_20260614.jsonl`
- repair/silver rows: `data/final/repaired_v1/full_pair_evidence_repair8_repair_silver_v1_20260614.jsonl`
- report json: `data/final/repaired_v1/full_pair_evidence_repair8_promotion_v1_20260614.report.json`

## Summary

- `reviewed_rows`: `8`
- `main_rows`: `3`
- `missing_reviews`: `0`
- `all_labels`: `{0: 5, 1: 3}`
- `main_labels`: `{1: 3}`
- `promotion_state`: `{'silver_refute_missing_product_evidence': 3, 'main_positive_refute': 3, 'silver_refute_insufficient_product_evidence': 2}`
- `confidence`: `{'high': 8}`
- `main_split`: `{'train': 2, 'test': 1}`
- `main_split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
- `category`: `{'sports_and_outdoor': 1, 'baby_kids_and_pets': 1, 'digital_and_electronics': 1}`
- `queue`: `data/final/repaired_v1/full_pair_evidence_repair_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair8_vlm_v1_20260614.jsonl`
- `out_all`: `data/final/repaired_v1/dataset_full_pair_evidence_repair8_stateful_v1_20260614.jsonl`
- `out_main`: `data/final/repaired_v1/dataset_full_pair_evidence_repair8_main_v1_20260614.jsonl`
- `out_repair`: `data/final/repaired_v1/full_pair_evidence_repair8_repair_silver_v1_20260614.jsonl`

## Promotion Rule

- `main_positive_refute`: claim found, product evidence found, and at least one aligned consumer comment refutes the same claim.
- `main_negative_support`: claim found, product evidence found, and aligned consumer comments support rather than refute the claim.
- Missing claim, missing/insufficient product evidence, mixed comments, and no aligned comments remain in stateful repair/silver outputs.
