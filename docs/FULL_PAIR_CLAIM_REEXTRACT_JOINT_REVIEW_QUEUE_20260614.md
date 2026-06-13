# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_pilot44_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `44`
- `selected_rows`: `23`
- `claim_found_pairs`: `23`
- `no_claim_pairs`: `21`
- `seeded_claim_count_bucket`: `{'5-10': 9, '2-4': 7, '1': 7}`
- `category`: `{'baby_kids_and_pets': 2, 'beauty_and_personal_care': 4, 'general': 3, 'jewelry_and_collectibles': 3, 'sports_and_outdoor': 1, 'digital_and_electronics': 2, 'shoes_and_bags': 4, 'smart_home': 3, 'food_and_beverages': 1}`

## Examples

- `p3675112863778865361__BABY_是否进口` cat=baby_kids_and_pets attr=是否进口 raw_claims=9 seeded=8
- `p3649304698210641818__BEAUTY_使用方法` cat=beauty_and_personal_care attr=使用方法 raw_claims=9 seeded=8
- `p3772877260806292222__GEN_胀包问题` cat=general attr=胀包问题 raw_claims=10 seeded=8
- `p3772134166922133792__JEWEL_底面修整` cat=jewelry_and_collectibles attr=底面修整 raw_claims=20 seeded=8
- `p3772134166922133792__JEWEL_是否镶嵌` cat=jewelry_and_collectibles attr=是否镶嵌 raw_claims=35 seeded=8
- `p3768024980667891731__SPORT_风格` cat=sports_and_outdoor attr=风格 raw_claims=23 seeded=8
- `p3580184154987456894__DIGITAL_贴膜特点` cat=digital_and_electronics attr=贴膜特点 raw_claims=7 seeded=7
- `p3580184154987456894__DIGITAL_贴膜神器` cat=digital_and_electronics attr=贴膜神器 raw_claims=7 seeded=7
- `p3649304698210641818__BEAUTY_是否电动` cat=beauty_and_personal_care attr=是否电动 raw_claims=6 seeded=6
- `p3753596480485720540__BABY_是否进口` cat=baby_kids_and_pets attr=是否进口 raw_claims=4 seeded=4
- `p3717436738369618131__BEAUTY_功效` cat=beauty_and_personal_care attr=功效 raw_claims=4 seeded=4
- `p3725779871381717481__BEAUTY_功效` cat=beauty_and_personal_care attr=功效 raw_claims=4 seeded=4
- `p3703683175160086673__SHOEBAG_货号` cat=shoes_and_bags attr=货号 raw_claims=3 seeded=3
- `p3708680373845229842__GEN_厚度` cat=general attr=厚度 raw_claims=2 seeded=2
- `p3703683175160086673__SHOEBAG_是否商场同款` cat=shoes_and_bags attr=是否商场同款 raw_claims=2 seeded=2
- `p3663574409718972261__HOME_描述` cat=smart_home attr=描述 raw_claims=2 seeded=2
- `p3700674430352097415__FOOD_包装类型` cat=food_and_beverages attr=包装类型 raw_claims=1 seeded=1
- `p3708680373845229842__GEN_品牌` cat=general attr=品牌 raw_claims=1 seeded=1
- `p3681068645343166788__JEWEL_圈口尺寸` cat=jewelry_and_collectibles attr=圈口/尺寸 raw_claims=1 seeded=1
- `p3703683175160086673__SHOEBAG_材质` cat=shoes_and_bags attr=材质 raw_claims=1 seeded=1
