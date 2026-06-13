# Full Pair Claim Repair Queue v1

This queue routes claim-missing full-pair reviews to exact SRT re-extraction.
It preserves hard/source-missing rows and does not change labels.

## Inputs

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- reviews: ``

## Outputs

- claim repair queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next1000_after500_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next1000_after500_v1_20260614.report.json`

## Summary

- `review_rows`: `0`
- `selected_rows`: `1000`
- `limit`: `1000`
- `excluded_pair_ids`: `858`
- `excluded_rows`: `858`
- `missing_queue_rows`: `0`
- `prefilter_state`: `{'strong_srt_candidate': 956, 'weak_srt_candidate': 44}`
- `source_review_action`: `{'seed_claim_repair': 1000}`
- `trigger_comment_bucket`: `{'2-4': 193, '1': 763, '5+': 44}`
- `category`: `{'baby_kids_and_pets': 176, 'beauty_and_personal_care': 50, 'digital_and_electronics': 29, 'food_and_beverages': 244, 'general': 188, 'jewelry_and_collectibles': 7, 'shoes_and_bags': 18, 'smart_home': 120, 'sports_and_outdoor': 71, 'apparel_and_underwear': 97}`
- `old_label_state`: `{'label_positive_claim_aligned_neg': 86, 'label_negative_no_aligned_review': 907, 'label_negative_claim_aligned_nonneg': 7}`

## Examples

- `p3482085810386379885__BABY_组合件数` cat=baby_kids_and_pets attr=组合件数 pref=strong_srt_candidate triggers=2
- `p3482085810386379885__BABY_适用腰围最大值` cat=baby_kids_and_pets attr=适用腰围最大值 pref=strong_srt_candidate triggers=2
- `p3486029801419094983__BABY_数量` cat=baby_kids_and_pets attr=数量 pref=strong_srt_candidate triggers=2
- `p3660066402737519648__BABY_沐浴露` cat=baby_kids_and_pets attr=沐浴露 pref=strong_srt_candidate triggers=2
- `p3660066402737519648__BABY_泡沫特性` cat=baby_kids_and_pets attr=泡沫特性 pref=strong_srt_candidate triggers=2
- `p3660066402737519648__BABY_添加剂` cat=baby_kids_and_pets attr=<添加剂> pref=strong_srt_candidate triggers=2
- `p3660066402737519648__BABY_清洗残留感` cat=baby_kids_and_pets attr=清洗残留感 pref=strong_srt_candidate triggers=2
- `p3660066402737519648__BABY_蓬松` cat=baby_kids_and_pets attr=<蓬松> pref=strong_srt_candidate triggers=2
- `p3675112863778865361__BABY_价格优惠` cat=baby_kids_and_pets attr=价格优惠 pref=strong_srt_candidate triggers=2
- `p3675112863778865361__BABY_活动价格` cat=baby_kids_and_pets attr=活动价格 pref=strong_srt_candidate triggers=2
- `p3743252672376078348__BABY_是否带帽子` cat=baby_kids_and_pets attr=是否带帽子 pref=strong_srt_candidate triggers=2
- `p3757291204451107454__BABY_适用季节` cat=baby_kids_and_pets attr=适用季节 pref=strong_srt_candidate triggers=2
- `p3809578084848500939__BABY_直播间展示` cat=baby_kids_and_pets attr=直播间展示 pref=strong_srt_candidate triggers=2
- `p3649304698210641818__BEAUTY_是否防水` cat=beauty_and_personal_care attr=是否防水 pref=strong_srt_candidate triggers=2
- `p3649304698210641818__BEAUTY_适用季节` cat=beauty_and_personal_care attr=适用季节 pref=strong_srt_candidate triggers=2
- `p3709009939570753771__BEAUTY_假货` cat=beauty_and_personal_care attr=<假货> pref=strong_srt_candidate triggers=2
- `p3716328694839640329__BEAUTY_是否临期` cat=beauty_and_personal_care attr=是否临期 pref=strong_srt_candidate triggers=2
- `p3722540703549620336__BEAUTY_适合肤质` cat=beauty_and_personal_care attr=适合肤质 pref=strong_srt_candidate triggers=2
- `p3772518315281482179__BEAUTY_电源方式` cat=beauty_and_personal_care attr=电源方式 pref=strong_srt_candidate triggers=2
- `p3711592039608090920__DIGITAL_材质` cat=digital_and_electronics attr=材质 pref=strong_srt_candidate triggers=2
