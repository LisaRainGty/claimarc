# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak120_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_rest19_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_rest19_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `120`
- `selected_rows`: `19`
- `claim_found_pairs`: `99`
- `no_claim_pairs`: `21`
- `excluded_pair_ids`: `80`
- `excluded_rows`: `80`
- `seeded_claim_count_bucket`: `{'5-10': 19}`
- `category`: `{'baby_kids_and_pets': 5, 'food_and_beverages': 10, 'general': 2, 'sports_and_outdoor': 2}`

## Examples

- `p3660066402737519648__BABY_功能` cat=baby_kids_and_pets attr=功能 raw_claims=11 seeded=8
- `p3660066402737519648__BABY_安全等级` cat=baby_kids_and_pets attr=安全等级 raw_claims=9 seeded=8
- `p3660066402737519648__BABY_组合形式` cat=baby_kids_and_pets attr=组合形式 raw_claims=9 seeded=8
- `p3660066402737519648__BABY_适用人群` cat=baby_kids_and_pets attr=适用人群 raw_claims=13 seeded=8
- `p3660066402737519648__BABY_面料材质` cat=baby_kids_and_pets attr=面料材质 raw_claims=11 seeded=8
- `p3490652681037608601__FOOD_包装类型` cat=food_and_beverages attr=包装类型 raw_claims=31 seeded=8
- `p3490652681037608601__FOOD_薯制品种类` cat=food_and_beverages attr=薯制品种类 raw_claims=11 seeded=8
- `p3490652681037608601__FOOD_风味` cat=food_and_beverages attr=风味 raw_claims=19 seeded=8
- `p3683596134795837638__FOOD_产品名称` cat=food_and_beverages attr=产品名称 raw_claims=16 seeded=8
- `p3683596134795837638__FOOD_风味` cat=food_and_beverages attr=风味 raw_claims=8 seeded=8
- `p3683596134795837638__FOOD_食用方式` cat=food_and_beverages attr=食用方式 raw_claims=9 seeded=8
- `p3753593497471550320__FOOD_包装类型` cat=food_and_beverages attr=包装类型 raw_claims=8 seeded=8
- `p3753593497471550320__FOOD_商品条形码` cat=food_and_beverages attr=商品条形码 raw_claims=10 seeded=8
- `p3753593497471550320__FOOD_杯子` cat=food_and_beverages attr=杯子 raw_claims=48 seeded=8
- `p3753593497471550320__FOOD_溶解性` cat=food_and_beverages attr=溶解性 raw_claims=17 seeded=8
- `p3709840760133255169__GEN_品牌` cat=general attr=品牌 raw_claims=19 seeded=8
- `p3772877260806292222__GEN_包装方式` cat=general attr=包装方式 raw_claims=14 seeded=8
- `p3538902193669330044__SPORT_工艺` cat=sports_and_outdoor attr=工艺 raw_claims=11 seeded=8
- `p3768024980667891731__SPORT_是否瑕疵` cat=sports_and_outdoor attr=是否瑕疵 raw_claims=18 seeded=8
