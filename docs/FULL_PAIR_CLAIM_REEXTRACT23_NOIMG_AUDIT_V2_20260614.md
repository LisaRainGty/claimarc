# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract23_noimg_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract23_noimg_audit_v2_20260614.report.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract23_noimg_audit_flags_v2_20260614.jsonl`

## Summary

- `queue_rows`: `23`
- `review_rows`: `23`
- `matched_reviews`: `23`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `3`
- `high_flag_rows`: `1`
- `medium_or_high_flag_rows`: `3`
- `flag_severity`: `{'medium': 2, 'high': 1}`
- `flag_code`: `{'claim_not_in_top_srt_prefilter': 1, 'promote_action_but_not_main_ready': 1, 'identity_attribute_claim_lacks_value': 1}`
- `promotion_state`: `{'repair_missing_claim': 17, 'repair_insufficient_product_evidence': 2, 'main_positive_refute': 2, 'silver_refute_insufficient_product_evidence': 1, 'lowinfo_no_aligned_comment': 1}`
- `new_y`: `{'0': 20, '1': 3}`
- `confidence`: `{'high': 23}`
- `category`: `{'baby_kids_and_pets': 2, 'beauty_and_personal_care': 4, 'general': 3, 'jewelry_and_collectibles': 3, 'sports_and_outdoor': 1, 'digital_and_electronics': 2, 'shoes_and_bags': 4, 'smart_home': 3, 'food_and_beverages': 1}`
- `queue`: `data/final/repaired_v1/full_pair_claim_reextract_joint_review_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract23_noimg_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract23_noimg_audit_flags_v2_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract23_noimg_audit_v2_20260614.report.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
