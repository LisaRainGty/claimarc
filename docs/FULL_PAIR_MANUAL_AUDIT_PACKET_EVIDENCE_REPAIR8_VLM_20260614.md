# Full Pair Manual Audit Packet v1

This packet is for manual inspection of the full-pair reconstruction pilot.
It is not a training dataset and should not be used to select only easy rows.

## Outputs

- csv: `data/final/repaired_v1/full_pair_manual_audit_packet_evidence_repair8_vlm_v1_20260614.csv`
- report: `data/final/repaired_v1/full_pair_manual_audit_packet_evidence_repair8_vlm_v1_20260614.report.json`

## Summary

- rows: `8`
- reviews joined: `8`
- rows with audit flags: `4`
- prefilter state: `{'strong_srt_candidate': 2, 'weak_srt_candidate': 6}`
- queue type: `{'evidence_vlm_repair_from_llm_review': 8}`
- claim state: `{'claim_seeded_from_llm_review': 8}`
- category: `{'baby_kids_and_pets': 1, 'general': 1, 'apparel_and_underwear': 2, 'digital_and_electronics': 2, 'food_and_beverages': 1, 'sports_and_outdoor': 1}`

## Manual Columns

- `manual_claim_source_valid`: whether the claim is traceable to SRT.
- `manual_claim_attribute_specific`: whether the claim is about the target attribute.
- `manual_product_evidence_valid`: whether evidence comes from product-side material.
- `manual_comments_same_claim`: whether comments discuss the same atomic claim.
- `manual_new_y`: final human label under the consumer-perception definition.
- `manual_decision`: promote, silver, rerun_claim, rerun_evidence, rerun_joint, or reject_out_of_scope.
