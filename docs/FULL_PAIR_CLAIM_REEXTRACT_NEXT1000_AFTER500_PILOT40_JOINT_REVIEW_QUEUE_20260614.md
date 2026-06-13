# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next1000_after500_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_pilot40_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_pilot40_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_pilot40_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `40`
- `selected_rows`: `35`
- `claim_found_pairs`: `35`
- `no_claim_pairs`: `5`
- `excluded_pair_ids`: `0`
- `excluded_rows`: `0`
- `seeded_claim_count_bucket`: `{'5-10': 13, '2-4': 16, '1': 6}`
- `category`: `{'baby_kids_and_pets': 11, 'food_and_beverages': 17, 'beauty_and_personal_care': 6, 'digital_and_electronics': 1}`

## Examples

- `p3660066402737519648__BABY_沐浴露` cat=baby_kids_and_pets attr=沐浴露 raw_claims=13 seeded=8
- `p3660066402737519648__BABY_添加剂` cat=baby_kids_and_pets attr=<添加剂> raw_claims=9 seeded=8
- `p3757291204451107454__BABY_适用季节` cat=baby_kids_and_pets attr=适用季节 raw_claims=9 seeded=8
- `p3671347504429007019__FOOD_功能` cat=food_and_beverages attr=功能 raw_claims=9 seeded=8
- `p3671347504429007019__FOOD_售卖方式` cat=food_and_beverages attr=售卖方式 raw_claims=13 seeded=8
- `p3683596134795837638__FOOD_异味去除效果` cat=food_and_beverages attr=异味去除效果 raw_claims=11 seeded=8
- `p3743612721522933924__FOOD_产品` cat=food_and_beverages attr=产品 raw_claims=15 seeded=8
- `p3743612721522933924__FOOD_规格` cat=food_and_beverages attr=规格 raw_claims=10 seeded=8
- `p3753593497471550320__FOOD_购买意图` cat=food_and_beverages attr=购买意图 raw_claims=12 seeded=8
- `p3675112863778865361__BABY_价格优惠` cat=baby_kids_and_pets attr=价格优惠 raw_claims=7 seeded=7
- `p3649304698210641818__BEAUTY_是否防水` cat=beauty_and_personal_care attr=是否防水 raw_claims=6 seeded=6
- `p3649304698210641818__BEAUTY_适用季节` cat=beauty_and_personal_care attr=适用季节 raw_claims=6 seeded=6
- `p3683596134795837638__FOOD_购买意图` cat=food_and_beverages attr=购买意图 raw_claims=6 seeded=6
- `p3683596134795837638__FOOD_包装颜色` cat=food_and_beverages attr=包装颜色 raw_claims=4 seeded=4
- `p3683596134795837638__FOOD_视频内容` cat=food_and_beverages attr=视频内容 raw_claims=4 seeded=4
- `p3683596134795837638__FOOD_适用人群` cat=food_and_beverages attr=适用人群 raw_claims=4 seeded=4
- `p3482085810386379885__BABY_组合件数` cat=baby_kids_and_pets attr=组合件数 raw_claims=3 seeded=3
- `p3482085810386379885__BABY_适用腰围最大值` cat=baby_kids_and_pets attr=适用腰围最大值 raw_claims=3 seeded=3
- `p3660066402737519648__BABY_泡沫特性` cat=baby_kids_and_pets attr=泡沫特性 raw_claims=3 seeded=3
- `p3660066402737519648__BABY_蓬松` cat=baby_kids_and_pets attr=<蓬松> raw_claims=3 seeded=3
