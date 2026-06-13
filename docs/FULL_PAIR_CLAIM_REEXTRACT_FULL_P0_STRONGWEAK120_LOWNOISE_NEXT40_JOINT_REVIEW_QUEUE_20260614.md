# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak120_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_lownoise_next40_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_lownoise_next40_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `120`
- `selected_rows`: `40`
- `claim_found_pairs`: `99`
- `no_claim_pairs`: `21`
- `excluded_pair_ids`: `40`
- `excluded_rows`: `40`
- `seeded_claim_count_bucket`: `{'2-4': 12, '5-10': 28}`
- `category`: `{'food_and_beverages': 14, 'general': 3, 'smart_home': 2, 'baby_kids_and_pets': 8, 'beauty_and_personal_care': 5, 'digital_and_electronics': 3, 'shoes_and_bags': 2, 'apparel_and_underwear': 2, 'sports_and_outdoor': 1}`

## Examples

- `p3784789614443760279__FOOD_生产日期` cat=food_and_beverages attr=生产日期 raw_claims=3 seeded=3
- `p3657116391363977564__GEN_品牌` cat=general attr=品牌 raw_claims=3 seeded=3
- `p3708680373845229842__GEN_面料成分含量` cat=general attr=面料成分含量 raw_claims=3 seeded=3
- `p3708680373845229842__GEN_面料材质` cat=general attr=面料材质 raw_claims=3 seeded=3
- `p3723511462895943927__HOME_电池容量` cat=smart_home attr=电池容量 raw_claims=3 seeded=3
- `p3660066402737519648__BABY_包装方式` cat=baby_kids_and_pets attr=包装方式 raw_claims=4 seeded=4
- `p3660066402737519648__BABY_数量` cat=baby_kids_and_pets attr=数量 raw_claims=4 seeded=4
- `p3717436738369618131__BEAUTY_功效` cat=beauty_and_personal_care attr=功效 raw_claims=4 seeded=4
- `p3752312351777488914__DIGITAL_品牌` cat=digital_and_electronics attr=品牌 raw_claims=4 seeded=4
- `p3683596134795837638__FOOD_净含量` cat=food_and_beverages attr=净含量 raw_claims=4 seeded=4
- `p3683596134795837638__FOOD_广告宣传` cat=food_and_beverages attr=<广告宣传> raw_claims=4 seeded=4
- `p3703683175160086673__SHOEBAG_品牌` cat=shoes_and_bags attr=品牌 raw_claims=4 seeded=4
- `p3530933199612072457__BABY_价格` cat=baby_kids_and_pets attr=<价格> raw_claims=5 seeded=5
- `p3660066402737519648__BABY_是否进口` cat=baby_kids_and_pets attr=是否进口 raw_claims=5 seeded=5
- `p3675112863778865361__BABY_价格` cat=baby_kids_and_pets attr=<价格> raw_claims=5 seeded=5
- `p3571193789647317734__BEAUTY_产品名称` cat=beauty_and_personal_care attr=产品名称 raw_claims=5 seeded=5
- `p3753593497471550320__FOOD_充电功能` cat=food_and_beverages attr=充电功能 raw_claims=5 seeded=5
- `p3753593497471550320__FOOD_开关功能` cat=food_and_beverages attr=开关功能 raw_claims=5 seeded=5
- `p3753593497471550320__FOOD_风味` cat=food_and_beverages attr=风味 raw_claims=5 seeded=5
- `p3703129246668030235__HOME_功能` cat=smart_home attr=功能 raw_claims=5 seeded=5
