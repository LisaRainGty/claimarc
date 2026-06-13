# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak20_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak17_noimg_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak17_noimg_audit_v1_20260614.report.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak17_noimg_audit_flags_v1_20260614.jsonl`

## Summary

- `queue_rows`: `17`
- `review_rows`: `17`
- `matched_reviews`: `17`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `4`
- `high_flag_rows`: `2`
- `medium_or_high_flag_rows`: `4`
- `flag_severity`: `{'medium': 2, 'high': 2}`
- `flag_code`: `{'claim_not_in_top_srt_prefilter': 1, 'identity_attribute_claim_lacks_value': 3}`
- `promotion_state`: `{'main_positive_refute': 7, 'silver_refute_insufficient_product_evidence': 4, 'repair_missing_claim': 5, 'main_negative_support': 1}`
- `new_y`: `{'1': 11, '0': 6}`
- `confidence`: `{'high': 17}`
- `category`: `{'baby_kids_and_pets': 6, 'food_and_beverages': 3, 'apparel_and_underwear': 1, 'beauty_and_personal_care': 3, 'digital_and_electronics': 2, 'general': 1, 'smart_home': 1}`
- `queue`: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak20_joint_review_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak17_noimg_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak17_noimg_audit_flags_v1_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak17_noimg_audit_v1_20260614.report.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
