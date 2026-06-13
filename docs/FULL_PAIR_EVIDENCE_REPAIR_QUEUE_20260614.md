# Full Pair Evidence Repair Queue v1

This queue routes claim-recovered positive consumer-refute rows back to Stage C/VLM evidence repair.
It does not drop hard rows, alter labels, or promote rows into the main benchmark.

## Inputs

- queue: `data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_pilot72_noimg_v1_20260614.jsonl`

## Outputs

- repair queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_evidence_repair_queue_v1_20260614.report.json`

## Summary

- `review_rows`: `72`
- `selected_rows`: `8`
- `limit`: `0`
- `missing_queue_rows`: `0`
- `skipped_no_detail_images`: `0`
- `source_claim_evidence_relation`: `{'insufficient': 8}`
- `source_product_evidence_found`: `{'False': 5, 'True': 3}`
- `category`: `{'food_and_beverages': 1, 'apparel_and_underwear': 2, 'general': 1, 'sports_and_outdoor': 1, 'digital_and_electronics': 2, 'baby_kids_and_pets': 1}`
- `attribute_objectivity`: `{'product_attribute': 8}`
- `image_count_bucket`: `{'13+': 7, '5-12': 1}`

## Examples

- `p3700674430352097415__FOOD_套餐详情` cat=food_and_beverages attr=套餐详情 relation=insufficient evidence_found=False images=39
- `p3731321152212172814__APPAREL_价格` cat=apparel_and_underwear attr=<价格> relation=insufficient evidence_found=False images=34
- `p3772877260806292222__GEN_价格` cat=general attr=<价格> relation=insufficient evidence_found=False images=15
- `p3538902193669330044__SPORT_风格` cat=sports_and_outdoor attr=风格 relation=insufficient evidence_found=False images=21
- `p3580184154987456894__DIGITAL_尺寸` cat=digital_and_electronics attr=尺寸 relation=insufficient evidence_found=False images=20
- `p3731321152212172814__APPAREL_附加功能` cat=apparel_and_underwear attr=附加功能 relation=insufficient evidence_found=True images=34
- `p3660066402737519648__BABY_功效` cat=baby_kids_and_pets attr=功效 relation=insufficient evidence_found=True images=8
- `p3679197849955992032__DIGITAL_安装方式` cat=digital_and_electronics attr=安装方式 relation=insufficient evidence_found=True images=26
