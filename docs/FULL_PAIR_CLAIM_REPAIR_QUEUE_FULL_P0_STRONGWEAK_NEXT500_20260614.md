# Full Pair Claim Repair Queue v1

This queue routes claim-missing full-pair reviews to exact SRT re-extraction.
It preserves hard/source-missing rows and does not change labels.

## Inputs

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- reviews: ``

## Outputs

- claim repair queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next500_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next500_v1_20260614.report.json`

## Summary

- `review_rows`: `0`
- `selected_rows`: `500`
- `limit`: `500`
- `excluded_pair_ids`: `320`
- `excluded_rows`: `320`
- `missing_queue_rows`: `0`
- `prefilter_state`: `{'strong_srt_candidate': 500}`
- `source_review_action`: `{'seed_claim_repair': 500}`
- `trigger_comment_bucket`: `{'5+': 180, '2-4': 320}`
- `category`: `{'general': 79, 'shoes_and_bags': 27, 'baby_kids_and_pets': 110, 'beauty_and_personal_care': 35, 'food_and_beverages': 108, 'sports_and_outdoor': 33, 'smart_home': 46, 'apparel_and_underwear': 50, 'digital_and_electronics': 12}`
- `old_label_state`: `{'label_negative_no_aligned_review': 416, 'label_positive_claim_aligned_neg': 83, 'label_negative_claim_aligned_nonneg': 1}`

## Examples

- `p3439082779739038362__GEN_是否有香味` cat=general attr=是否有香味 pref=strong_srt_candidate triggers=8
- `p3709750295312597707__SHOEBAG_内里材质` cat=shoes_and_bags attr=内里材质 pref=strong_srt_candidate triggers=8
- `p3482085810386379885__BABY_侧漏` cat=baby_kids_and_pets attr=侧漏 pref=strong_srt_candidate triggers=8
- `p3630823105900996885__BABY_裤长` cat=baby_kids_and_pets attr=裤长 pref=strong_srt_candidate triggers=8
- `p3809773793598111808__BABY_尺码` cat=baby_kids_and_pets attr=尺码 pref=strong_srt_candidate triggers=8
- `p3705480440681988181__BEAUTY_质地` cat=beauty_and_personal_care attr=质地 pref=strong_srt_candidate triggers=8
- `p3753593497471550320__FOOD_耐用性` cat=food_and_beverages attr=耐用性 pref=strong_srt_candidate triggers=8
- `p3753593497471550320__FOOD_触控响应性` cat=food_and_beverages attr=触控响应性 pref=strong_srt_candidate triggers=8
- `p3709421951429771344__GEN_是否加绒` cat=general attr=是否加绒 pref=strong_srt_candidate triggers=8
- `p3691242379324555611__SPORT_尺码` cat=sports_and_outdoor attr=尺码 pref=strong_srt_candidate triggers=8
- `p3768024980667891731__SPORT_水印表现` cat=sports_and_outdoor attr=水印表现 pref=strong_srt_candidate triggers=8
- `p3657116391363977564__GEN_生产日期` cat=general attr=生产日期 pref=strong_srt_candidate triggers=7
- `p3649304698210641818__BEAUTY_主成分` cat=beauty_and_personal_care attr=主成分 pref=strong_srt_candidate triggers=7
- `p3660066402737519648__BABY_宣传` cat=baby_kids_and_pets attr=宣传 pref=strong_srt_candidate triggers=7
- `p3683596134795837638__FOOD_宣传内容` cat=food_and_beverages attr=<宣传内容> pref=strong_srt_candidate triggers=7
- `p3743612721522933924__FOOD_产品名称` cat=food_and_beverages attr=产品名称 pref=strong_srt_candidate triggers=7
- `p3717408677242732721__BEAUTY_品牌` cat=beauty_and_personal_care attr=品牌 pref=strong_srt_candidate triggers=7
- `p3483616667710416469__FOOD_生产日期` cat=food_and_beverages attr=生产日期 pref=strong_srt_candidate triggers=7
- `p3784789614443760279__FOOD_产品名称` cat=food_and_beverages attr=产品名称 pref=strong_srt_candidate triggers=7
- `p3663574409718972261__HOME_使用方法` cat=smart_home attr=使用方法 pref=strong_srt_candidate triggers=7
