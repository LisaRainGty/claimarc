# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next200_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_lownoise60_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_lownoise60_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `200`
- `selected_rows`: `60`
- `claim_found_pairs`: `165`
- `no_claim_pairs`: `35`
- `excluded_pair_ids`: `0`
- `excluded_rows`: `0`
- `seeded_claim_count_bucket`: `{'1': 54, '2-4': 6}`
- `category`: `{'apparel_and_underwear': 4, 'baby_kids_and_pets': 9, 'beauty_and_personal_care': 8, 'digital_and_electronics': 1, 'food_and_beverages': 12, 'general': 19, 'smart_home': 6, 'sports_and_outdoor': 1}`

## Examples

- `p3617221077064913906__APPAREL_香味` cat=apparel_and_underwear attr=香味 raw_claims=1 seeded=1
- `p3718004155214856462__APPAREL_价格` cat=apparel_and_underwear attr=<价格> raw_claims=1 seeded=1
- `p3720538529466548264__APPAREL_是否加绒` cat=apparel_and_underwear attr=是否加绒 raw_claims=1 seeded=1
- `p3730164951386554776__APPAREL_工艺` cat=apparel_and_underwear attr=工艺 raw_claims=1 seeded=1
- `p3693461401977880924__BABY_护发素效果` cat=baby_kids_and_pets attr=护发素效果 raw_claims=1 seeded=1
- `p3724404517601673447__BABY_面料材质` cat=baby_kids_and_pets attr=面料材质 raw_claims=1 seeded=1
- `p3724404517601673447__BABY_颜色分类` cat=baby_kids_and_pets attr=颜色分类 raw_claims=1 seeded=1
- `p3778410690419753155__BABY_尺码` cat=baby_kids_and_pets attr=尺码 raw_claims=1 seeded=1
- `p3796309837801980378__BABY_数量` cat=baby_kids_and_pets attr=数量 raw_claims=1 seeded=1
- `p3810008932236263827__BABY_面料材质` cat=baby_kids_and_pets attr=面料材质 raw_claims=1 seeded=1
- `p3705157179901345903__BEAUTY_价格` cat=beauty_and_personal_care attr=<价格> raw_claims=1 seeded=1
- `p3709009939570753771__BEAUTY_适合肤质` cat=beauty_and_personal_care attr=适合肤质 raw_claims=1 seeded=1
- `p3709009939570753771__BEAUTY_适用对象` cat=beauty_and_personal_care attr=适用对象 raw_claims=1 seeded=1
- `p3716328694839640329__BEAUTY_颜色` cat=beauty_and_personal_care attr=颜色 raw_claims=1 seeded=1
- `p3717408677242732721__BEAUTY_购买渠道` cat=beauty_and_personal_care attr=<购买渠道> raw_claims=1 seeded=1
- `p3734640811833426037__DIGITAL_颜色` cat=digital_and_electronics attr=颜色 raw_claims=1 seeded=1
- `p3483616667710416469__FOOD_套餐份量` cat=food_and_beverages attr=套餐份量 raw_claims=1 seeded=1
- `p3483616667710416469__FOOD_是否临期` cat=food_and_beverages attr=是否临期 raw_claims=1 seeded=1
- `p3483616667710416469__FOOD_辛辣程度` cat=food_and_beverages attr=辛辣程度 raw_claims=1 seeded=1
- `p3671415216744300934__FOOD_价格` cat=food_and_beverages attr=价格 raw_claims=1 seeded=1
