# Full Pair Evidence Repair Queue v1

This queue routes claim-recovered positive consumer-refute rows back to Stage C/VLM evidence repair.
It does not drop hard rows, alter labels, or promote rows into the main benchmark.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_lownoise_next40_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise_next40_noimg_v1_20260614.jsonl`

## Outputs

- repair queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_lownoise_next40_silver11_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_evidence_repair_queue_lownoise_next40_silver11_v1_20260614.report.json`

## Summary

- `review_rows`: `40`
- `selected_rows`: `11`
- `limit`: `0`
- `missing_queue_rows`: `0`
- `skipped_no_detail_images`: `0`
- `source_claim_evidence_relation`: `{'insufficient': 11}`
- `source_product_evidence_found`: `{'False': 5, 'True': 6}`
- `category`: `{'food_and_beverages': 7, 'baby_kids_and_pets': 3, 'general': 1}`
- `attribute_objectivity`: `{'product_attribute': 11}`
- `image_count_bucket`: `{'5-12': 7, '13+': 4}`

## Examples

- `p3753593497471550320__FOOD_充电功能` cat=food_and_beverages attr=充电功能 relation=insufficient evidence_found=False images=9
- `p3675112863778865361__BABY_价格` cat=baby_kids_and_pets attr=<价格> relation=insufficient evidence_found=False images=13
- `p3753593497471550320__FOOD_搅拌杯属性` cat=food_and_beverages attr=搅拌杯属性 relation=insufficient evidence_found=False images=9
- `p3753593497471550320__FOOD_价格` cat=food_and_beverages attr=价格 relation=insufficient evidence_found=False images=9
- `p3660066402737519648__BABY_价格` cat=baby_kids_and_pets attr=<价格> relation=insufficient evidence_found=False images=8
- `p3683596134795837638__FOOD_广告宣传` cat=food_and_beverages attr=<广告宣传> relation=insufficient evidence_found=True images=29
- `p3683596134795837638__FOOD_净含量` cat=food_and_beverages attr=净含量 relation=insufficient evidence_found=True images=29
- `p3708680373845229842__GEN_面料成分含量` cat=general attr=面料成分含量 relation=insufficient evidence_found=True images=13
- `p3753593497471550320__FOOD_食用方式` cat=food_and_beverages attr=食用方式 relation=insufficient evidence_found=True images=9
- `p3660066402737519648__BABY_功效` cat=baby_kids_and_pets attr=功效 relation=insufficient evidence_found=True images=8
- `p3753593497471550320__FOOD_风味` cat=food_and_beverages attr=风味 relation=insufficient evidence_found=True images=9
