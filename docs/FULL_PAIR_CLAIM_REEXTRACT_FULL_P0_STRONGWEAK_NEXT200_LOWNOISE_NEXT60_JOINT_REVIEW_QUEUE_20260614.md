# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next200_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_lownoise_next60_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_lownoise_next60_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `200`
- `selected_rows`: `60`
- `claim_found_pairs`: `165`
- `no_claim_pairs`: `35`
- `excluded_pair_ids`: `60`
- `excluded_rows`: `60`
- `seeded_claim_count_bucket`: `{'2-4': 53, '5-10': 7}`
- `category`: `{'beauty_and_personal_care': 5, 'food_and_beverages': 27, 'general': 12, 'shoes_and_bags': 3, 'smart_home': 3, 'sports_and_outdoor': 1, 'apparel_and_underwear': 4, 'baby_kids_and_pets': 5}`

## Examples

- `p3717436738369618131__BEAUTY_适合肤质` cat=beauty_and_personal_care attr=适合肤质 raw_claims=2 seeded=2
- `p3719631374240579758__FOOD_风味` cat=food_and_beverages attr=风味 raw_claims=2 seeded=2
- `p3736181201837359332__FOOD_风味` cat=food_and_beverages attr=风味 raw_claims=2 seeded=2
- `p3743612721522933924__FOOD_是否临期` cat=food_and_beverages attr=是否临期 raw_claims=2 seeded=2
- `p3753593497471550320__FOOD_充电接口` cat=food_and_beverages attr=充电接口 raw_claims=2 seeded=2
- `p3753593497471550320__FOOD_是否临期` cat=food_and_beverages attr=是否临期 raw_claims=2 seeded=2
- `p3779880885608907125__FOOD_薯制品种类` cat=food_and_beverages attr=薯制品种类 raw_claims=2 seeded=2
- `p3784789614443760279__FOOD_净含量` cat=food_and_beverages attr=净含量 raw_claims=2 seeded=2
- `p3784789614443760279__FOOD_是否临期` cat=food_and_beverages attr=是否临期 raw_claims=2 seeded=2
- `p3784789614443760279__FOOD_食用方式` cat=food_and_beverages attr=食用方式 raw_claims=2 seeded=2
- `p3439082779739038362__GEN_净含量` cat=general attr=净含量 raw_claims=2 seeded=2
- `p3711948332873154684__GEN_品牌` cat=general attr=品牌 raw_claims=2 seeded=2
- `p3720772695445602597__GEN_品牌` cat=general attr=品牌 raw_claims=2 seeded=2
- `p3720772695445602597__GEN_鞋底材质` cat=general attr=鞋底材质 raw_claims=2 seeded=2
- `p3759321650240290980__GEN_大纸张` cat=general attr=大纸张 raw_claims=2 seeded=2
- `p3728761325975896542__SHOEBAG_价格` cat=shoes_and_bags attr=<价格> raw_claims=2 seeded=2
- `p3728761325975896542__SHOEBAG_面料材质成分含量` cat=shoes_and_bags attr=面料材质成分含量 raw_claims=2 seeded=2
- `p3663574409718972261__HOME_清洁效果` cat=smart_home attr=<清洁效果> raw_claims=2 seeded=2
- `p3758583403453218941__HOME_重量` cat=smart_home attr=重量 raw_claims=2 seeded=2
- `p3538902193669330044__SPORT_价格` cat=sports_and_outdoor attr=价格 raw_claims=2 seeded=2
