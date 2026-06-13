# Full Pair Claim-Reextract Joint Review Queue v1

This queue sends exact SRT claim-reextract candidates back through the full-pair reviewer.
It is a recall-to-review bridge, not a promoted training dataset.

## Inputs

- queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next1000_after500_v1_20260614.jsonl`
- claim reextract: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_v1_20260614.jsonl`

## Outputs

- joint review queue: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_fewerclaims_joint_review_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next1000_after500_fewerclaims_joint_review_queue_v1_20260614.report.json`

## Summary

- `claim_reextract_rows`: `1000`
- `selected_rows`: `820`
- `claim_found_pairs`: `820`
- `no_claim_pairs`: `180`
- `excluded_pair_ids`: `0`
- `excluded_rows`: `0`
- `seeded_claim_count_bucket`: `{'1': 308, '2-4': 304, '5-10': 208}`
- `category`: `{'apparel_and_underwear': 83, 'baby_kids_and_pets': 141, 'beauty_and_personal_care': 29, 'digital_and_electronics': 24, 'food_and_beverages': 227, 'general': 139, 'jewelry_and_collectibles': 7, 'shoes_and_bags': 18, 'smart_home': 87, 'sports_and_outdoor': 65}`

## Examples

- `p3637605842993421416__APPAREL_颜色分类` cat=apparel_and_underwear attr=颜色分类 raw_claims=1 seeded=1
- `p3702755032744198404__APPAREL_头晕` cat=apparel_and_underwear attr=头晕 raw_claims=1 seeded=1
- `p3718539716002447832__APPAREL_厚度` cat=apparel_and_underwear attr=厚度 raw_claims=1 seeded=1
- `p3720538529466548264__APPAREL_颜色` cat=apparel_and_underwear attr=颜色 raw_claims=1 seeded=1
- `p3724646493936812309__APPAREL_面料材质` cat=apparel_and_underwear attr=面料材质 raw_claims=1 seeded=1
- `p3729134507027202188__APPAREL_是否加绒` cat=apparel_and_underwear attr=是否加绒 raw_claims=1 seeded=1
- `p3730164951386554776__APPAREL_拍照技术` cat=apparel_and_underwear attr=拍照技术 raw_claims=1 seeded=1
- `p3731321152212172814__APPAREL_价格` cat=apparel_and_underwear attr=<价格> raw_claims=1 seeded=1
- `p3751877580878381132__APPAREL_异味` cat=apparel_and_underwear attr=<异味> raw_claims=1 seeded=1
- `p3756360813653393631__APPAREL_货品一致性` cat=apparel_and_underwear attr=<货品一致性> raw_claims=1 seeded=1
- `p3768630107141439951__APPAREL_包装方式` cat=apparel_and_underwear attr=包装方式 raw_claims=1 seeded=1
- `p3769360928148160743__APPAREL_颜色` cat=apparel_and_underwear attr=颜色 raw_claims=1 seeded=1
- `p3771938831893397784__APPAREL_面料成分含量` cat=apparel_and_underwear attr=面料成分含量 raw_claims=1 seeded=1
- `p3771938831893397784__APPAREL_颜色` cat=apparel_and_underwear attr=颜色 raw_claims=1 seeded=1
- `p3774679334686687719__APPAREL_是否加绒` cat=apparel_and_underwear attr=是否加绒 raw_claims=1 seeded=1
- `p3786058455190733186__APPAREL_客观属性名词短语` cat=apparel_and_underwear attr=客观属性名词短语 raw_claims=1 seeded=1
- `p3786058455190733186__APPAREL_尺码` cat=apparel_and_underwear attr=尺码 raw_claims=1 seeded=1
- `p3787702886532776010__APPAREL_是否加绒` cat=apparel_and_underwear attr=是否加绒 raw_claims=1 seeded=1
- `p3482085810386379885__BABY_大灯` cat=baby_kids_and_pets attr=大灯 raw_claims=1 seeded=1
- `p3482085810386379885__BABY_智能程度` cat=baby_kids_and_pets attr=智能程度 raw_claims=1 seeded=1
