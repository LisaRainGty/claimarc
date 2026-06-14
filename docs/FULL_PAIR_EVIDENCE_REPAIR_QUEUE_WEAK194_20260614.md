# Full Pair Evidence Repair Queue v1

This queue routes claim-recovered positive consumer-refute rows back to Stage C/VLM evidence repair.
It does not drop hard rows, alter labels, or promote rows into the main benchmark.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_weak_next1000_after1500_pilot300_fewerclaims_joint_review_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_weak_next1000_after1500_pilot300_fewerclaims194_noimg_flash_v1_20260614.jsonl`

## Outputs

- repair queue: `data/final/repaired_v1/full_pair_evidence_repair_queue_weak194_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_evidence_repair_queue_weak194_v1_20260614.report.json`

## Summary

- `review_rows`: `194`
- `selected_rows`: `7`
- `limit`: `0`
- `missing_queue_rows`: `0`
- `skipped_no_detail_images`: `1`
- `source_claim_evidence_relation`: `{'insufficient': 7}`
- `source_product_evidence_found`: `{'False': 4, 'True': 3}`
- `category`: `{'shoes_and_bags': 2, 'beauty_and_personal_care': 2, 'apparel_and_underwear': 1, 'smart_home': 1, 'sports_and_outdoor': 1}`
- `attribute_objectivity`: `{'product_attribute': 7}`
- `image_count_bucket`: `{'13+': 5, '1-4': 1, '5-12': 1}`

## Examples

- `p3785691042330837066__SHOEBAG_鞋底工艺` cat=shoes_and_bags attr=鞋底工艺 relation=insufficient evidence_found=False images=51
- `p3705157179901345903__BEAUTY_净含量` cat=beauty_and_personal_care attr=净含量 relation=insufficient evidence_found=False images=18
- `p3807592908006228107__BEAUTY_价格` cat=beauty_and_personal_care attr=<价格> relation=insufficient evidence_found=False images=2
- `p3702755032744198404__APPAREL_镜片尺寸` cat=apparel_and_underwear attr=镜片尺寸 relation=insufficient evidence_found=False images=13
- `p3738753685501640789__HOME_商品等级` cat=smart_home attr=商品等级 relation=insufficient evidence_found=True images=7
- `p3538902193669330044__SPORT_安装指导` cat=sports_and_outdoor attr=安装指导 relation=insufficient evidence_found=True images=21
- `p3707740106652778867__SHOEBAG_是否瑕疵` cat=shoes_and_bags attr=是否瑕疵 relation=insufficient evidence_found=True images=15
