# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak120_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak20_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak20_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak20_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `20`
- `selected_rows`: `17`
- `claim_found_pairs`: `17`
- `no_claim_pairs`: `3`
- `seeded_claim_count_bucket`: `{'5-10': 8, '2-4': 7, '1': 2}`
- `category`: `{'baby_kids_and_pets': 6, 'food_and_beverages': 3, 'apparel_and_underwear': 1, 'beauty_and_personal_care': 3, 'digital_and_electronics': 2, 'general': 1, 'smart_home': 1}`

## Examples

- `p3660066402737519648__BABY_产品名称` cat=baby_kids_and_pets attr=产品名称 raw_claims=12 seeded=8
- `p3660066402737519648__BABY_功效` cat=baby_kids_and_pets attr=功效 raw_claims=11 seeded=8
- `p3660066402737519648__BABY_安全等级` cat=baby_kids_and_pets attr=安全等级 raw_claims=9 seeded=8
- `p3683596134795837638__FOOD_风味` cat=food_and_beverages attr=风味 raw_claims=8 seeded=8
- `p3702755032744198404__APPAREL_主要功能` cat=apparel_and_underwear attr=主要功能 raw_claims=7 seeded=7
- `p3571193789647317734__BEAUTY_功效` cat=beauty_and_personal_care attr=功效 raw_claims=7 seeded=7
- `p3580184154987456894__DIGITAL_贴膜特点` cat=digital_and_electronics attr=贴膜特点 raw_claims=7 seeded=7
- `p3649304698210641818__BEAUTY_是否电动` cat=beauty_and_personal_care attr=是否电动 raw_claims=6 seeded=6
- `p3717436738369618131__BEAUTY_功效` cat=beauty_and_personal_care attr=功效 raw_claims=4 seeded=4
- `p3752312351777488914__DIGITAL_品牌` cat=digital_and_electronics attr=品牌 raw_claims=4 seeded=4
- `p3683596134795837638__FOOD_广告宣传` cat=food_and_beverages attr=<广告宣传> raw_claims=4 seeded=4
- `p3482085810386379885__BABY_品牌` cat=baby_kids_and_pets attr=品牌 raw_claims=3 seeded=3
- `p3708680373845229842__GEN_面料材质` cat=general attr=面料材质 raw_claims=3 seeded=3
- `p3660066402737519648__BABY_品牌` cat=baby_kids_and_pets attr=品牌 raw_claims=2 seeded=2
- `p3663574409718972261__HOME_描述` cat=smart_home attr=描述 raw_claims=2 seeded=2
- `p3693461401977880924__BABY_品牌` cat=baby_kids_and_pets attr=品牌 raw_claims=1 seeded=1
- `p3753593497471550320__FOOD_搅拌性能` cat=food_and_beverages attr=搅拌性能 raw_claims=1 seeded=1
