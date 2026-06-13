# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next500_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_partial151_lownoise60_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_partial151_lownoise60_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `151`
- `selected_rows`: `60`
- `claim_found_pairs`: `130`
- `no_claim_pairs`: `21`
- `excluded_pair_ids`: `0`
- `excluded_rows`: `0`
- `seeded_claim_count_bucket`: `{'1': 46, '2-4': 14}`
- `category`: `{'apparel_and_underwear': 6, 'baby_kids_and_pets': 18, 'beauty_and_personal_care': 6, 'digital_and_electronics': 1, 'food_and_beverages': 10, 'general': 8, 'shoes_and_bags': 3, 'smart_home': 6, 'sports_and_outdoor': 2}`

## Examples

- `p3649510178069603603__APPAREL_价格` cat=apparel_and_underwear attr=<价格> raw_claims=1 seeded=1
- `p3702755032744198404__APPAREL_重量` cat=apparel_and_underwear attr=重量 raw_claims=1 seeded=1
- `p3731321152212172814__APPAREL_数据线类型` cat=apparel_and_underwear attr=数据线类型 raw_claims=1 seeded=1
- `p3769737523740409928__APPAREL_价格` cat=apparel_and_underwear attr=<价格> raw_claims=1 seeded=1
- `p3482085810386379885__BABY_侧漏` cat=baby_kids_and_pets attr=侧漏 raw_claims=1 seeded=1
- `p3630823105900996885__BABY_厚度` cat=baby_kids_and_pets attr=厚度 raw_claims=1 seeded=1
- `p3630823105900996885__BABY_适用季节` cat=baby_kids_and_pets attr=适用季节 raw_claims=1 seeded=1
- `p3693461401977880924__BABY_净含量` cat=baby_kids_and_pets attr=净含量 raw_claims=1 seeded=1
- `p3693461401977880924__BABY_发质状态` cat=baby_kids_and_pets attr=<发质状态> raw_claims=1 seeded=1
- `p3693461401977880924__BABY_掉发情况` cat=baby_kids_and_pets attr=掉发情况 raw_claims=1 seeded=1
- `p3693461401977880924__BABY_清爽` cat=baby_kids_and_pets attr=<清爽> raw_claims=1 seeded=1
- `p3693461401977880924__BABY_面料材质` cat=baby_kids_and_pets attr=面料材质 raw_claims=1 seeded=1
- `p3790945142245032045__BABY_品牌` cat=baby_kids_and_pets attr=品牌 raw_claims=1 seeded=1
- `p3790945142245032045__BABY_套餐详情` cat=baby_kids_and_pets attr=套餐详情 raw_claims=1 seeded=1
- `p3809578084848500939__BABY_异味` cat=baby_kids_and_pets attr=<异味> raw_claims=1 seeded=1
- `p3809578084848500939__BABY_绒子含量` cat=baby_kids_and_pets attr=绒子含量 raw_claims=1 seeded=1
- `p3809578084848500939__BABY_衣长` cat=baby_kids_and_pets attr=衣长 raw_claims=1 seeded=1
- `p3809578084848500939__BABY_适用季节` cat=baby_kids_and_pets attr=适用季节 raw_claims=1 seeded=1
- `p3705157179901345903__BEAUTY_是否临期` cat=beauty_and_personal_care attr=是否临期 raw_claims=1 seeded=1
- `p3705480440681988181__BEAUTY_质地` cat=beauty_and_personal_care attr=质地 raw_claims=1 seeded=1
