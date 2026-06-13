# Full Pair Claim Repair Queue v1

This queue routes claim-missing full-pair reviews to exact SRT re-extraction.
It preserves hard/source-missing rows and does not change labels.

## Inputs

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- reviews: ``

## Outputs

- claim repair queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next200_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next200_v1_20260614.report.json`

## Summary

- `review_rows`: `0`
- `selected_rows`: `200`
- `limit`: `200`
- `excluded_pair_ids`: `120`
- `excluded_rows`: `120`
- `missing_queue_rows`: `0`
- `prefilter_state`: `{'strong_srt_candidate': 200}`
- `source_review_action`: `{'seed_claim_repair': 200}`
- `trigger_comment_bucket`: `{'5+': 200}`
- `category`: `{'food_and_beverages': 69, 'general': 42, 'smart_home': 13, 'sports_and_outdoor': 8, 'baby_kids_and_pets': 28, 'beauty_and_personal_care': 20, 'shoes_and_bags': 8, 'apparel_and_underwear': 11, 'digital_and_electronics': 1}`
- `old_label_state`: `{'label_negative_no_aligned_review': 166, 'label_positive_claim_aligned_neg': 33, 'label_negative_claim_aligned_nonneg': 1}`

## Examples

- `p3753593497471550320__FOOD_杯子属性` cat=food_and_beverages attr=杯子属性 pref=strong_srt_candidate triggers=12
- `p3779880885608907125__FOOD_风味` cat=food_and_beverages attr=风味 pref=strong_srt_candidate triggers=12
- `p3784789614443760279__FOOD_商品条形码` cat=food_and_beverages attr=商品条形码 pref=strong_srt_candidate triggers=12
- `p3657116391363977564__GEN_净含量` cat=general attr=净含量 pref=strong_srt_candidate triggers=12
- `p3657116391363977564__GEN_包装方式` cat=general attr=包装方式 pref=strong_srt_candidate triggers=12
- `p3716987214157185084__GEN_品牌` cat=general attr=品牌 pref=strong_srt_candidate triggers=12
- `p3720772695445602597__GEN_鞋底材质` cat=general attr=鞋底材质 pref=strong_srt_candidate triggers=12
- `p3663574409718972261__HOME_清洁效果` cat=smart_home attr=<清洁效果> pref=strong_srt_candidate triggers=12
- `p3538902193669330044__SPORT_类型` cat=sports_and_outdoor attr=类型 pref=strong_srt_candidate triggers=12
- `p3482085810386379885__BABY_厚度` cat=baby_kids_and_pets attr=厚度 pref=strong_srt_candidate triggers=12
- `p3629782961358001865__BABY_厚度` cat=baby_kids_and_pets attr=厚度 pref=strong_srt_candidate triggers=12
- `p3660066402737519648__BABY_头皮屑` cat=baby_kids_and_pets attr=<头皮屑> pref=strong_srt_candidate triggers=12
- `p3724404517601673447__BABY_面料材质` cat=baby_kids_and_pets attr=面料材质 pref=strong_srt_candidate triggers=12
- `p3705157179901345903__BEAUTY_价格` cat=beauty_and_personal_care attr=<价格> pref=strong_srt_candidate triggers=12
- `p3709009939570753771__BEAUTY_净含量` cat=beauty_and_personal_care attr=净含量 pref=strong_srt_candidate triggers=12
- `p3709009939570753771__BEAUTY_适合肤质` cat=beauty_and_personal_care attr=适合肤质 pref=strong_srt_candidate triggers=12
- `p3716328694839640329__BEAUTY_适合肤质` cat=beauty_and_personal_care attr=适合肤质 pref=strong_srt_candidate triggers=12
- `p3717436738369618131__BEAUTY_适合肤质` cat=beauty_and_personal_care attr=适合肤质 pref=strong_srt_candidate triggers=12
- `p3483616667710416469__FOOD_商品条形码` cat=food_and_beverages attr=商品条形码 pref=strong_srt_candidate triggers=12
- `p3585819184510160371__FOOD_风味` cat=food_and_beverages attr=风味 pref=strong_srt_candidate triggers=12
