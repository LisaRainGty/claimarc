# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next1000_after1500_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_weak_next1000_after1500_pilot300_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_weak_next1000_after1500_pilot300_fewerclaims_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_weak_next1000_after1500_pilot300_fewerclaims_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `300`
- `selected_rows`: `194`
- `claim_found_pairs`: `194`
- `no_claim_pairs`: `106`
- `excluded_pair_ids`: `0`
- `excluded_rows`: `0`
- `seeded_claim_count_bucket`: `{'1': 84, '2-4': 66, '5-10': 44}`
- `category`: `{'apparel_and_underwear': 20, 'baby_kids_and_pets': 29, 'beauty_and_personal_care': 34, 'digital_and_electronics': 5, 'food_and_beverages': 30, 'general': 38, 'shoes_and_bags': 15, 'smart_home': 10, 'sports_and_outdoor': 13}`

## Examples

- `p3750430700810141764__APPAREL_香味` cat=apparel_and_underwear attr=香味 raw_claims=1 seeded=1
- `p3756360813653393631__APPAREL_尺码` cat=apparel_and_underwear attr=尺码 raw_claims=1 seeded=1
- `p3756360813653393631__APPAREL_工艺` cat=apparel_and_underwear attr=工艺 raw_claims=1 seeded=1
- `p3756360813653393631__APPAREL_面料材质` cat=apparel_and_underwear attr=面料材质 raw_claims=1 seeded=1
- `p3756360813653393631__APPAREL_颜色分类` cat=apparel_and_underwear attr=颜色分类 raw_claims=1 seeded=1
- `p3768840820300579205__APPAREL_工艺` cat=apparel_and_underwear attr=工艺 raw_claims=1 seeded=1
- `p3769737523740409928__APPAREL_尺码` cat=apparel_and_underwear attr=尺码 raw_claims=1 seeded=1
- `p3795700145853694215__APPAREL_工艺` cat=apparel_and_underwear attr=工艺 raw_claims=1 seeded=1
- `p3530933199612072457__BABY_功能` cat=baby_kids_and_pets attr=功能 raw_claims=1 seeded=1
- `p3530933199612072457__BABY_口感` cat=baby_kids_and_pets attr=<口感> raw_claims=1 seeded=1
- `p3530933199612072457__BABY_掉毛` cat=baby_kids_and_pets attr=<掉毛> raw_claims=1 seeded=1
- `p3530933199612072457__BABY_数量` cat=baby_kids_and_pets attr=数量 raw_claims=1 seeded=1
- `p3530933199612072457__BABY_是否进口` cat=baby_kids_and_pets attr=是否进口 raw_claims=1 seeded=1
- `p3530933199612072457__BABY_适用人群` cat=baby_kids_and_pets attr=适用人群 raw_claims=1 seeded=1
- `p3625778776367080085__BABY_包装方式` cat=baby_kids_and_pets attr=包装方式 raw_claims=1 seeded=1
- `p3651530887260166475__BABY_旋转性能` cat=baby_kids_and_pets attr=旋转性能 raw_claims=1 seeded=1
- `p3653346505244421877__BABY_面料材质` cat=baby_kids_and_pets attr=面料材质 raw_claims=1 seeded=1
- `p3663621559769277407__BABY_套餐详情` cat=baby_kids_and_pets attr=套餐详情 raw_claims=1 seeded=1
- `p3675112863778865361__BABY_功能` cat=baby_kids_and_pets attr=功能 raw_claims=1 seeded=1
- `p3675112863778865361__BABY_猫主粮分类` cat=baby_kids_and_pets attr=猫主粮分类 raw_claims=1 seeded=1
