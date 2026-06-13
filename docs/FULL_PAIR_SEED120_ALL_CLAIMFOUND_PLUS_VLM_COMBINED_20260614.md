# Combined Full Pair Main Candidates v1

This report combines batch-level promoted rows and reapplies cross-batch claim-family gates.

## Inputs

- `data/final/repaired_v1/dataset_full_pair_claimreextract_full_p0_strongweak120_lownoise40_noimg_main_v5_20260614.jsonl`
- `data/final/repaired_v1/dataset_full_pair_lownoise40_silver12_vlm_main_v4_20260614.jsonl`
- `data/final/repaired_v1/dataset_full_pair_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_main_v4_20260614.jsonl`
- `data/final/repaired_v1/dataset_full_pair_lownoise_next40_silver11_vlm_main_v3_20260614.jsonl`
- `data/final/repaired_v1/dataset_full_pair_claimreextract_full_p0_strongweak120_rest19_noimg_main_v2_20260614.jsonl`

## Outputs

- stateful combined rows: `data/final/repaired_v1/dataset_full_pair_seed120_all_claimfound_plus_vlm_stateful_v1_20260614.jsonl`
- main combined rows: `data/final/repaired_v1/dataset_full_pair_seed120_all_claimfound_plus_vlm_main_v1_20260614.jsonl`
- repair/silver combined rows: `data/final/repaired_v1/full_pair_seed120_all_claimfound_plus_vlm_repair_silver_v1_20260614.jsonl`
- report json: `data/final/repaired_v1/dataset_full_pair_seed120_all_claimfound_plus_vlm_v1_20260614.report.json`

## Summary

- `input_rows`: `21`
- `pair_id_duplicates_dropped`: `0`
- `stateful_rows`: `21`
- `main_rows`: `20`
- `main_labels`: `{1: 17, 0: 3}`
- `promotion_state`: `{'main_positive_refute': 17, 'main_negative_support': 3, 'silver_duplicate_claim_family': 1}`
- `main_split`: `{'train': 14, 'val': 2, 'test': 4}`
- `main_split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
- `category`: `{'beauty_and_personal_care': 3, 'general': 3, 'shoes_and_bags': 2, 'food_and_beverages': 7, 'smart_home': 2, 'apparel_and_underwear': 1, 'baby_kids_and_pets': 2}`
- `duplicate_claim_family_groups`: `1`
- `conflicting_claim_family_groups`: `0`
- `duplicate_claim_family_demoted`: `1`
- `duplicate_claim_family_examples`: `[{'demoted_pair_id': 'p3708680373845229842__GEN_面料材质', 'kept_pair_id': 'p3708680373845229842__GEN_面料成分含量', 'attribute_name': '面料材质', 'kept_attribute_name': '面料成分含量', 'state_before': 'main_positive_refute', 'reason': 'duplicate_same_label'}]`
- `inputs`: `['data/final/repaired_v1/dataset_full_pair_claimreextract_full_p0_strongweak120_lownoise40_noimg_main_v5_20260614.jsonl', 'data/final/repaired_v1/dataset_full_pair_lownoise40_silver12_vlm_main_v4_20260614.jsonl', 'data/final/repaired_v1/dataset_full_pair_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_main_v4_20260614.jsonl', 'data/final/repaired_v1/dataset_full_pair_lownoise_next40_silver11_vlm_main_v3_20260614.jsonl', 'data/final/repaired_v1/dataset_full_pair_claimreextract_full_p0_strongweak120_rest19_noimg_main_v2_20260614.jsonl']`
- `out_all`: `data/final/repaired_v1/dataset_full_pair_seed120_all_claimfound_plus_vlm_stateful_v1_20260614.jsonl`
- `out_main`: `data/final/repaired_v1/dataset_full_pair_seed120_all_claimfound_plus_vlm_main_v1_20260614.jsonl`
- `out_repair`: `data/final/repaired_v1/full_pair_seed120_all_claimfound_plus_vlm_repair_silver_v1_20260614.jsonl`
