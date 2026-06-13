# Full Pair LLM Review Audit v1

This report audits LLM/VLM reconstruction reviews before promotion.
It checks label-definition consistency rather than benchmark separability.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_pilot40_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_pilot40_noimg_flash_v1_20260614.jsonl`

## Outputs

- report json: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_pilot40_noimg_flash_audit_v1_20260614.report.json`
- flagged rows: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_pilot40_noimg_flash_audit_flags_v1_20260614.jsonl`

## Summary

- `queue_rows`: `35`
- `review_rows`: `35`
- `matched_reviews`: `35`
- `missing_reviews`: `0`
- `duplicate_review_pairs`: `0`
- `duplicate_review_events`: `0`
- `flagged_rows`: `13`
- `high_flag_rows`: `0`
- `medium_or_high_flag_rows`: `13`
- `flag_severity`: `{'medium': 18, 'low': 1}`
- `flag_code`: `{'conflicting_comment_relation_requires_silver': 5, 'promote_action_but_not_main_ready': 10, 'claim_not_in_top_srt_prefilter': 2, 'raw_llm_y_overridden_by_clean_rule': 1, 'positive_label_missing_product_evidence_for_main': 1}`
- `promotion_state`: `{'main_positive_refute': 5, 'silver_refute_insufficient_product_evidence': 5, 'silver_conflicting_comment_relation': 4, 'silver_commercial_promise_attribute': 1, 'lowinfo_no_aligned_comment': 5, 'repair_missing_claim': 6, 'repair_insufficient_product_evidence': 6, 'repair_missing_evidence': 1, 'silver_refute_missing_product_evidence': 1, 'main_negative_support': 1}`
- `new_y`: `{'1': 15, '0': 20}`
- `confidence`: `{'high': 35}`
- `category`: `{'baby_kids_and_pets': 11, 'food_and_beverages': 17, 'beauty_and_personal_care': 6, 'digital_and_electronics': 1}`
- `queue`: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_pilot40_joint_review_queue_v1_20260614.jsonl`
- `reviews`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_pilot40_noimg_flash_v1_20260614.jsonl`
- `flagged`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_pilot40_noimg_flash_audit_flags_v1_20260614.jsonl`
- `out`: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_pilot40_noimg_flash_audit_v1_20260614.report.json`

## Gate Interpretation

- `high` flags block main promotion until rerun or manual repair.
- `medium` flags require manual sampling or silver routing.
- `mechanism_contradiction_without_consumer_refute` is not a positive label by itself.
- Missing reviews are expected before the LLM pilot has been executed.
