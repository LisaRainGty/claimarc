# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next500_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_full_lownoise_next80_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_full_lownoise_next80_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `500`
- `selected_rows`: `80`
- `claim_found_pairs`: `429`
- `no_claim_pairs`: `71`
- `excluded_pair_ids`: `120`
- `excluded_rows`: `120`
- `seeded_claim_count_bucket`: `{'1': 66, '2-4': 14}`
- `category`: `{'apparel_and_underwear': 11, 'baby_kids_and_pets': 14, 'beauty_and_personal_care': 5, 'digital_and_electronics': 3, 'food_and_beverages': 5, 'general': 23, 'shoes_and_bags': 1, 'smart_home': 11, 'sports_and_outdoor': 7}`

## Examples

- `p3720163847332561117__APPAREL_颜色分类` cat=apparel_and_underwear attr=颜色分类 raw_claims=1 seeded=1
- `p3720538529466548264__APPAREL_颜色分类` cat=apparel_and_underwear attr=颜色分类 raw_claims=1 seeded=1
- `p3730164951386554776__APPAREL_适用季节` cat=apparel_and_underwear attr=适用季节 raw_claims=1 seeded=1
- `p3730164951386554776__APPAREL_面料成分含量` cat=apparel_and_underwear attr=面料成分含量 raw_claims=1 seeded=1
- `p3731321152212172814__APPAREL_品牌` cat=apparel_and_underwear attr=品牌 raw_claims=1 seeded=1
- `p3768630107141439951__APPAREL_价格` cat=apparel_and_underwear attr=<价格> raw_claims=1 seeded=1
- `p3482085810386379885__BABY_产地` cat=baby_kids_and_pets attr=产地 raw_claims=1 seeded=1
- `p3601244194531867599__BABY_价格` cat=baby_kids_and_pets attr=<价格> raw_claims=1 seeded=1
- `p3629782961358001865__BABY_保暖性` cat=baby_kids_and_pets attr=保暖性 raw_claims=1 seeded=1
- `p3663621559769277407__BABY_颜色分类` cat=baby_kids_and_pets attr=颜色分类 raw_claims=1 seeded=1
- `p3693461401977880924__BABY_客观属性名词短语` cat=baby_kids_and_pets attr=客观属性名词短语 raw_claims=1 seeded=1
- `p3790945142245032045__BABY_产品名称` cat=baby_kids_and_pets attr=产品名称 raw_claims=1 seeded=1
- `p3798169408317293005__BABY_口感` cat=baby_kids_and_pets attr=<口感> raw_claims=1 seeded=1
- `p3808361039184134380__BABY_颜色分类` cat=baby_kids_and_pets attr=颜色分类 raw_claims=1 seeded=1
- `p3709009939570753771__BEAUTY_保质期` cat=beauty_and_personal_care attr=保质期 raw_claims=1 seeded=1
- `p3709009939570753771__BEAUTY_形状` cat=beauty_and_personal_care attr=形状 raw_claims=1 seeded=1
- `p3716328694839640329__BEAUTY_条形码` cat=beauty_and_personal_care attr=条形码 raw_claims=1 seeded=1
- `p3751869753099157973__DIGITAL_价格` cat=digital_and_electronics attr=价格 raw_claims=1 seeded=1
- `p3794823514993852677__DIGITAL_差价补偿` cat=digital_and_electronics attr=差价补偿 raw_claims=1 seeded=1
- `p3683596134795837638__FOOD_品牌` cat=food_and_beverages attr=品牌 raw_claims=1 seeded=1
