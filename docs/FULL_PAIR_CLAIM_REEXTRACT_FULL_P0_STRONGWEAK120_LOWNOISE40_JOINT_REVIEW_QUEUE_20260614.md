# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak120_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_lownoise40_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak120_lownoise40_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `120`
- `selected_rows`: `40`
- `claim_found_pairs`: `99`
- `no_claim_pairs`: `21`
- `seeded_claim_count_bucket`: `{'1': 20, '2-4': 20}`
- `category`: `{'baby_kids_and_pets': 9, 'beauty_and_personal_care': 2, 'food_and_beverages': 10, 'general': 10, 'shoes_and_bags': 3, 'smart_home': 6}`

## Examples

- `p3629782961358001865__BABY_充绒量` cat=baby_kids_and_pets attr=充绒量 raw_claims=1 seeded=1
- `p3693461401977880924__BABY_品牌` cat=baby_kids_and_pets attr=品牌 raw_claims=1 seeded=1
- `p3693461401977880924__BABY_面料成分含量` cat=baby_kids_and_pets attr=面料成分含量 raw_claims=1 seeded=1
- `p3809578084848500939__BABY_充绒量` cat=baby_kids_and_pets attr=充绒量 raw_claims=1 seeded=1
- `p3571193789647317734__BEAUTY_质地` cat=beauty_and_personal_care attr=质地 raw_claims=1 seeded=1
- `p3583766632415258115__BEAUTY_颜色` cat=beauty_and_personal_care attr=颜色 raw_claims=1 seeded=1
- `p3753593497471550320__FOOD_包装颜色` cat=food_and_beverages attr=包装颜色 raw_claims=1 seeded=1
- `p3753593497471550320__FOOD_搅拌性能` cat=food_and_beverages attr=搅拌性能 raw_claims=1 seeded=1
- `p3439082779739038362__GEN_品牌` cat=general attr=品牌 raw_claims=1 seeded=1
- `p3549451361445893022__GEN_包数` cat=general attr=包数 raw_claims=1 seeded=1
- `p3549451361445893022__GEN_品牌` cat=general attr=品牌 raw_claims=1 seeded=1
- `p3657116391363977564__GEN_品质` cat=general attr=<品质> raw_claims=1 seeded=1
- `p3708680373845229842__GEN_价格` cat=general attr=<价格> raw_claims=1 seeded=1
- `p3711948332873154684__GEN_包装方式` cat=general attr=包装方式 raw_claims=1 seeded=1
- `p3759321650240290980__GEN_价格` cat=general attr=<价格> raw_claims=1 seeded=1
- `p3759321650240290980__GEN_包数` cat=general attr=包数 raw_claims=1 seeded=1
- `p3703683175160086673__SHOEBAG_颜色` cat=shoes_and_bags attr=颜色 raw_claims=1 seeded=1
- `p3494583142998751397__HOME_价格` cat=smart_home attr=<价格> raw_claims=1 seeded=1
- `p3703129246668030235__HOME_价格` cat=smart_home attr=<价格> raw_claims=1 seeded=1
- `p3703129246668030235__HOME_商品等级` cat=smart_home attr=商品等级 raw_claims=1 seeded=1
