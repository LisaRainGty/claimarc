# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next200_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_rest45_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next200_rest45_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `200`
- `selected_rows`: `45`
- `claim_found_pairs`: `165`
- `no_claim_pairs`: `35`
- `excluded_pair_ids`: `120`
- `excluded_rows`: `120`
- `seeded_claim_count_bucket`: `{'5-10': 45}`
- `category`: `{'smart_home': 1, 'sports_and_outdoor': 5, 'apparel_and_underwear': 3, 'baby_kids_and_pets': 6, 'food_and_beverages': 24, 'general': 3, 'shoes_and_bags': 3}`

## Examples

- `p3703129246668030235__HOME_描述` cat=smart_home attr=描述 raw_claims=5 seeded=5
- `p3538902193669330044__SPORT_类型` cat=sports_and_outdoor attr=类型 raw_claims=5 seeded=5
- `p3718004155214856462__APPAREL_面料材质` cat=apparel_and_underwear attr=面料材质 raw_claims=6 seeded=6
- `p3720163847332561117__APPAREL_面料材质` cat=apparel_and_underwear attr=面料材质 raw_claims=6 seeded=6
- `p3482085810386379885__BABY_里料材质` cat=baby_kids_and_pets attr=里料材质 raw_claims=6 seeded=6
- `p3693461401977880924__BABY_包装方式` cat=baby_kids_and_pets attr=包装方式 raw_claims=6 seeded=6
- `p3585819184510160371__FOOD_食用方式` cat=food_and_beverages attr=食用方式 raw_claims=6 seeded=6
- `p3671347504429007019__FOOD_包装类型` cat=food_and_beverages attr=包装类型 raw_claims=6 seeded=6
- `p3743612721522933924__FOOD_食用方式` cat=food_and_beverages attr=食用方式 raw_claims=6 seeded=6
- `p3753593497471550320__FOOD_品牌` cat=food_and_beverages attr=品牌 raw_claims=6 seeded=6
- `p3753593497471550320__FOOD_按键功能` cat=food_and_beverages attr=按键功能 raw_claims=6 seeded=6
- `p3753593497471550320__FOOD_杯子旋转性能` cat=food_and_beverages attr=杯子旋转性能 raw_claims=6 seeded=6
- `p3779880885608907125__FOOD_风味` cat=food_and_beverages attr=风味 raw_claims=6 seeded=6
- `p3784789614443760279__FOOD_包装类型` cat=food_and_beverages attr=包装类型 raw_claims=6 seeded=6
- `p3538902193669330044__SPORT_鞋面材质` cat=sports_and_outdoor attr=鞋面材质 raw_claims=6 seeded=6
- `p3757291204451107454__BABY_面料材质` cat=baby_kids_and_pets attr=面料材质 raw_claims=7 seeded=7
- `p3809773793598111808__BABY_厚度` cat=baby_kids_and_pets attr=厚度 raw_claims=7 seeded=7
- `p3483616667710416469__FOOD_商品条形码` cat=food_and_beverages attr=商品条形码 raw_claims=7 seeded=7
- `p3784789614443760279__FOOD_套餐份量` cat=food_and_beverages attr=套餐份量 raw_claims=7 seeded=7
- `p3702755032744198404__APPAREL_镜框款式` cat=apparel_and_underwear attr=镜框款式 raw_claims=9 seeded=8
