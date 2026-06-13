# Full Pair Evidence Repair Queue v1

This queue routes claim-recovered positive consumer-refute rows back to Stage C/VLM evidence repair.
It does not drop hard rows, alter labels, or promote rows into the main benchmark.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_lownoise40_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_full_p0_strongweak120_lownoise40_noimg_v1_20260614.jsonl`

## Outputs

- repair queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_lownoise40_silver12_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_evidence_repair_queue_lownoise40_silver12_v1_20260614.report.json`

## Summary

- `review_rows`: `40`
- `selected_rows`: `12`
- `limit`: `0`
- `missing_queue_rows`: `0`
- `skipped_no_detail_images`: `0`
- `source_claim_evidence_relation`: `{'insufficient': 12}`
- `source_product_evidence_found`: `{'False': 4, 'True': 8}`
- `category`: `{'food_and_beverages': 4, 'smart_home': 5, 'general': 1, 'baby_kids_and_pets': 2}`
- `attribute_objectivity`: `{'product_attribute': 12}`
- `image_count_bucket`: `{'13+': 9, '0': 2, '5-12': 1}`

## Examples

- `p3683596134795837638__FOOD_价格` cat=food_and_beverages attr=价格 relation=insufficient evidence_found=False images=29
- `p3703129246668030235__HOME_价格` cat=smart_home attr=<价格> relation=insufficient evidence_found=False images=23
- `p3703129246668030235__HOME_安装方式` cat=smart_home attr=安装方式 relation=insufficient evidence_found=False images=23
- `p3708680373845229842__GEN_价格` cat=general attr=<价格> relation=insufficient evidence_found=False images=13
- `p3663574409718972261__HOME_功效` cat=smart_home attr=功效 relation=insufficient evidence_found=True images=25
- `p3784241694703223083__FOOD_甜度` cat=food_and_beverages attr=甜度 relation=insufficient evidence_found=True images=0
- `p3784241694703223083__FOOD_风味` cat=food_and_beverages attr=风味 relation=insufficient evidence_found=True images=0
- `p3494583142998751397__HOME_价格` cat=smart_home attr=<价格> relation=insufficient evidence_found=True images=54
- `p3753593497471550320__FOOD_包装颜色` cat=food_and_beverages attr=包装颜色 relation=insufficient evidence_found=True images=9
- `p3663574409718972261__HOME_描述` cat=smart_home attr=描述 relation=insufficient evidence_found=True images=25
- `p3809578084848500939__BABY_充绒量` cat=baby_kids_and_pets attr=充绒量 relation=insufficient evidence_found=True images=41
- `p3482085810386379885__BABY_品牌` cat=baby_kids_and_pets attr=品牌 relation=insufficient evidence_found=True images=19
