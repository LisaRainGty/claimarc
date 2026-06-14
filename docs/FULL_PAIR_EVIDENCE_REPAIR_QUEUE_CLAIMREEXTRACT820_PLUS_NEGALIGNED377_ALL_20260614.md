# Full Pair Evidence Repair Queue v1

This queue routes claim-recovered positive consumer-refute rows back to Stage C/VLM evidence repair.
It does not drop hard rows, alter labels, or promote rows into the main benchmark.

## Inputs

- queue: `data/final/repaired_v1/full_pair_joint_review_queue_claimreextract820_plus_negaligned377_reviewed_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract820_plus_negaligned377_noimg_flash_v1_20260614.jsonl`

## Outputs

- repair queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_claimreextract820_plus_negaligned377_all_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_evidence_repair_queue_claimreextract820_plus_negaligned377_all_v1_20260614.report.json`

## Summary

- `review_rows`: `1196`
- `selected_rows`: `139`
- `limit`: `0`
- `missing_queue_rows`: `0`
- `skipped_no_detail_images`: `0`
- `source_claim_evidence_relation`: `{'insufficient': 139}`
- `source_product_evidence_found`: `{'False': 73, 'True': 66}`
- `category`: `{'baby_kids_and_pets': 20, 'smart_home': 19, 'food_and_beverages': 39, 'general': 17, 'sports_and_outdoor': 22, 'apparel_and_underwear': 13, 'shoes_and_bags': 4, 'beauty_and_personal_care': 3, 'jewelry_and_collectibles': 1, 'digital_and_electronics': 1}`
- `attribute_objectivity`: `{'product_attribute': 139}`
- `image_count_bucket`: `{'13+': 82, '1-4': 5, '5-12': 52}`

## Examples

- `p3809773793598111808__BABY_服装版型` cat=baby_kids_and_pets attr=服装版型 relation=insufficient evidence_found=False images=22
- `p3699398204358525049__HOME_出水方式` cat=smart_home attr=出水方式 relation=insufficient evidence_found=False images=2
- `p3753593497471550320__FOOD_搅拌杯保温性` cat=food_and_beverages attr=搅拌杯保温性 relation=insufficient evidence_found=False images=9
- `p3753593497471550320__FOOD_搅拌部件结构稳定性` cat=food_and_beverages attr=搅拌部件结构稳定性 relation=insufficient evidence_found=False images=9
- `p3439082779739038362__GEN_干度` cat=general attr=干度 relation=insufficient evidence_found=False images=20
- `p3703129246668030235__HOME_次品零件` cat=smart_home attr=次品零件 relation=insufficient evidence_found=False images=23
- `p3716863370939465968__SPORT_尺码` cat=sports_and_outdoor attr=尺码 relation=insufficient evidence_found=False images=14
- `p3768382098625396975__SPORT_尺码` cat=sports_and_outdoor attr=尺码 relation=insufficient evidence_found=False images=22
- `p3751877580878381132__APPAREL_绒子含量` cat=apparel_and_underwear attr=绒子含量 relation=insufficient evidence_found=False images=21
- `p3729661121632796878__FOOD_生产日期` cat=food_and_beverages attr=生产日期 relation=insufficient evidence_found=False images=6
- `p3805186120120140090__SHOEBAG_品牌` cat=shoes_and_bags attr=品牌 relation=insufficient evidence_found=False images=22
- `p3703129246668030235__HOME_膨胀螺丝` cat=smart_home attr=膨胀螺丝 relation=insufficient evidence_found=False images=23
- `p3703129246668030235__HOME_钻墙能力` cat=smart_home attr=钻墙能力 relation=insufficient evidence_found=False images=23
- `p3703129246668030235__HOME_钻瓷片能力` cat=smart_home attr=钻瓷片能力 relation=insufficient evidence_found=False images=23
- `p3723511462895943927__HOME_出口品质` cat=smart_home attr=出口品质 relation=insufficient evidence_found=False images=32
- `p3581774877637633940__SPORT_礼包内容` cat=sports_and_outdoor attr=礼包内容 relation=insufficient evidence_found=False images=25
- `p3716503433512091689__APPAREL_随机帽子` cat=apparel_and_underwear attr=随机帽子 relation=insufficient evidence_found=False images=30
- `p3720975198850253111__APPAREL_服装版型` cat=apparel_and_underwear attr=服装版型 relation=insufficient evidence_found=False images=9
- `p3720975198850253111__APPAREL_翘边情况` cat=apparel_and_underwear attr=翘边情况 relation=insufficient evidence_found=False images=9
- `p3720975198850253111__APPAREL_适合肤质` cat=apparel_and_underwear attr=适合肤质 relation=insufficient evidence_found=False images=9
