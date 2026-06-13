# Full Pair Claim Repair Queue v1

This queue routes claim-missing full-pair reviews to exact SRT re-extraction.
It preserves hard/source-missing rows and does not change labels.

## Inputs

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- reviews: ``

## Outputs

- claim repair queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak120_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak120_v1_20260614.report.json`

## Summary

- `review_rows`: `0`
- `selected_rows`: `120`
- `limit`: `120`
- `missing_queue_rows`: `0`
- `prefilter_state`: `{'strong_srt_candidate': 120}`
- `source_review_action`: `{'seed_claim_repair': 120}`
- `trigger_comment_bucket`: `{'5+': 120}`
- `category`: `{'food_and_beverages': 36, 'apparel_and_underwear': 3, 'baby_kids_and_pets': 25, 'beauty_and_personal_care': 17, 'general': 17, 'digital_and_electronics': 4, 'smart_home': 10, 'shoes_and_bags': 5, 'sports_and_outdoor': 3}`
- `old_label_state`: `{'label_positive_claim_aligned_neg': 23, 'label_negative_no_aligned_review': 96, 'label_negative_claim_aligned_nonneg': 1}`

## Examples

- `p3683596134795837638__FOOD_风味` cat=food_and_beverages attr=风味 pref=strong_srt_candidate triggers=12
- `p3731321152212172814__APPAREL_电源容量` cat=apparel_and_underwear attr=电源容量 pref=strong_srt_candidate triggers=12
- `p3482085810386379885__BABY_品牌` cat=baby_kids_and_pets attr=品牌 pref=strong_srt_candidate triggers=12
- `p3717436738369618131__BEAUTY_功效` cat=beauty_and_personal_care attr=功效 pref=strong_srt_candidate triggers=12
- `p3708680373845229842__GEN_面料材质` cat=general attr=面料材质 pref=strong_srt_candidate triggers=12
- `p3660066402737519648__BABY_功效` cat=baby_kids_and_pets attr=功效 pref=strong_srt_candidate triggers=12
- `p3709009939570753771__BEAUTY_功效` cat=beauty_and_personal_care attr=功效 pref=strong_srt_candidate triggers=12
- `p3580184154987456894__DIGITAL_贴膜特点` cat=digital_and_electronics attr=贴膜特点 pref=strong_srt_candidate triggers=12
- `p3649304698210641818__BEAUTY_是否电动` cat=beauty_and_personal_care attr=是否电动 pref=strong_srt_candidate triggers=12
- `p3702755032744198404__APPAREL_主要功能` cat=apparel_and_underwear attr=主要功能 pref=strong_srt_candidate triggers=12
- `p3705157179901345903__BEAUTY_产品名称` cat=beauty_and_personal_care attr=产品名称 pref=strong_srt_candidate triggers=12
- `p3660066402737519648__BABY_品牌` cat=baby_kids_and_pets attr=品牌 pref=strong_srt_candidate triggers=12
- `p3660066402737519648__BABY_安全等级` cat=baby_kids_and_pets attr=安全等级 pref=strong_srt_candidate triggers=12
- `p3683596134795837638__FOOD_广告宣传` cat=food_and_beverages attr=<广告宣传> pref=strong_srt_candidate triggers=12
- `p3752312351777488914__DIGITAL_品牌` cat=digital_and_electronics attr=品牌 pref=strong_srt_candidate triggers=12
- `p3753593497471550320__FOOD_搅拌性能` cat=food_and_beverages attr=搅拌性能 pref=strong_srt_candidate triggers=12
- `p3660066402737519648__BABY_产品名称` cat=baby_kids_and_pets attr=产品名称 pref=strong_srt_candidate triggers=12
- `p3693461401977880924__BABY_品牌` cat=baby_kids_and_pets attr=品牌 pref=strong_srt_candidate triggers=12
- `p3663574409718972261__HOME_描述` cat=smart_home attr=描述 pref=strong_srt_candidate triggers=12
- `p3571193789647317734__BEAUTY_功效` cat=beauty_and_personal_care attr=功效 pref=strong_srt_candidate triggers=12
