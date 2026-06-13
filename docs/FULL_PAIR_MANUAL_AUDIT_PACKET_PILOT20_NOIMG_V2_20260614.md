# Full Pair Manual Audit Packet v1

This packet is for manual inspection of the full-pair reconstruction pilot.
It is not a training dataset and should not be used to select only easy rows.

## Outputs

- csv: `data/final/repaired_v1/full_pair_manual_audit_packet_pilot20_noimg_v2_20260614.csv`
- report: `data/final/repaired_v1/full_pair_manual_audit_packet_pilot20_noimg_v2_20260614.report.json`

## Summary

- rows: `72`
- reviews joined: `20`
- rows with audit flags: `53`
- prefilter state: `{'no_srt_candidate': 8, 'strong_srt_candidate': 24, 'very_weak_srt_candidate': 16, 'weak_srt_candidate': 24}`
- queue type: `{'full_claim_evidence_label_rebuild': 41, 'claim_reextract_label_rebuild': 31}`
- claim state: `{'claim_missing': 58, 'claim_present_review_needed': 14}`
- category: `{'apparel_and_underwear': 11, 'digital_and_electronics': 11, 'shoes_and_bags': 7, 'baby_kids_and_pets': 8, 'beauty_and_personal_care': 8, 'food_and_beverages': 6, 'general': 6, 'jewelry_and_collectibles': 5, 'smart_home': 5, 'sports_and_outdoor': 5}`

## Manual Columns

- `manual_claim_source_valid`: whether the claim is traceable to SRT.
- `manual_claim_attribute_specific`: whether the claim is about the target attribute.
- `manual_product_evidence_valid`: whether evidence comes from product-side material.
- `manual_comments_same_claim`: whether comments discuss the same atomic claim.
- `manual_new_y`: final human label under the consumer-perception definition.
- `manual_decision`: promote, silver, rerun_claim, rerun_evidence, rerun_joint, or reject_out_of_scope.
