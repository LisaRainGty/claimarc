# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next500_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_partial259_lownoise_next60_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_partial259_lownoise_next60_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `259`
- `selected_rows`: `60`
- `claim_found_pairs`: `217`
- `no_claim_pairs`: `42`
- `excluded_pair_ids`: `60`
- `excluded_rows`: `60`
- `seeded_claim_count_bucket`: `{'1': 25, '2-4': 35}`
- `category`: `{'apparel_and_underwear': 6, 'baby_kids_and_pets': 16, 'beauty_and_personal_care': 6, 'food_and_beverages': 19, 'general': 7, 'smart_home': 4, 'sports_and_outdoor': 2}`

## Examples

- `p3720163847332561117__APPAREL_包装方式` cat=apparel_and_underwear attr=包装方式 raw_claims=1 seeded=1
- `p3720538529466548264__APPAREL_面料材质` cat=apparel_and_underwear attr=面料材质 raw_claims=1 seeded=1
- `p3756360813653393631__APPAREL_香味` cat=apparel_and_underwear attr=香味 raw_claims=1 seeded=1
- `p3482085810386379885__BABY_适用季节` cat=baby_kids_and_pets attr=适用季节 raw_claims=1 seeded=1
- `p3693461401977880924__BABY_异味` cat=baby_kids_and_pets attr=<异味> raw_claims=1 seeded=1
- `p3790945142245032045__BABY_保质期` cat=baby_kids_and_pets attr=保质期 raw_claims=1 seeded=1
- `p3809578084848500939__BABY_里料材质` cat=baby_kids_and_pets attr=里料材质 raw_claims=1 seeded=1
- `p3583766632415258115__BEAUTY_颜色分类` cat=beauty_and_personal_care attr=颜色分类 raw_claims=1 seeded=1
- `p3676842541531136387__BEAUTY_适合肤质` cat=beauty_and_personal_care attr=适合肤质 raw_claims=1 seeded=1
- `p3483616667710416469__FOOD_是否为有机食品` cat=food_and_beverages attr=是否为有机食品 raw_claims=1 seeded=1
- `p3719631374240579758__FOOD_包装类型` cat=food_and_beverages attr=包装类型 raw_claims=1 seeded=1
- `p3719815067852734494__FOOD_价格` cat=food_and_beverages attr=价格 raw_claims=1 seeded=1
- `p3742805896858828993__FOOD_套餐份量` cat=food_and_beverages attr=套餐份量 raw_claims=1 seeded=1
- `p3779880885608907125__FOOD_食品工艺` cat=food_and_beverages attr=食品工艺 raw_claims=1 seeded=1
- `p3784241694703223083__FOOD_售卖方式` cat=food_and_beverages attr=售卖方式 raw_claims=1 seeded=1
- `p3793465375434342476__FOOD_套餐详情` cat=food_and_beverages attr=套餐详情 raw_claims=1 seeded=1
- `p3439082779739038362__GEN_产地` cat=general attr=产地 raw_claims=1 seeded=1
- `p3657116391363977564__GEN_掉渣` cat=general attr=掉渣 raw_claims=1 seeded=1
- `p3720772695445602597__GEN_贴合度` cat=general attr=<贴合度> raw_claims=1 seeded=1
- `p3769484498568347935__GEN_保质期` cat=general attr=保质期 raw_claims=1 seeded=1
