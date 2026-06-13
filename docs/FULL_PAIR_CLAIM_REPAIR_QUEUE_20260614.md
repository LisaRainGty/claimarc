# Full Pair Claim Repair Queue v1

This queue routes claim-missing full-pair reviews to exact SRT re-extraction.
It preserves hard/source-missing rows and does not change labels.

## Inputs

- queue: `data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl`
- reviews: `data/final/repaired_v1/full_pair_reconstruction_llm_pilot72_noimg_v1_20260614.jsonl`

## Outputs

- claim repair queue: `data/final/repaired_v1/full_pair_claim_repair_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_repair_queue_v1_20260614.report.json`

## Summary

- `review_rows`: `72`
- `selected_rows`: `44`
- `limit`: `0`
- `missing_queue_rows`: `0`
- `prefilter_state`: `{'strong_srt_candidate': 6, 'weak_srt_candidate': 15, 'very_weak_srt_candidate': 15, 'no_srt_candidate': 8}`
- `source_review_action`: `{'rerun_claim': 42, 'rerun_joint': 1, 'drop_no_reconstructable_claim': 1}`
- `trigger_comment_bucket`: `{'5+': 31, '2-4': 10, '1': 3}`
- `category`: `{'apparel_and_underwear': 6, 'beauty_and_personal_care': 6, 'digital_and_electronics': 7, 'smart_home': 4, 'shoes_and_bags': 6, 'general': 4, 'baby_kids_and_pets': 5, 'sports_and_outdoor': 1, 'jewelry_and_collectibles': 3, 'food_and_beverages': 2}`
- `old_label_state`: `{'label_negative_no_aligned_review': 43, 'label_positive_claim_aligned_neg': 1}`

## Examples

- `p3731321152212172814__APPAREL_电源容量` cat=apparel_and_underwear attr=电源容量 pref=strong_srt_candidate triggers=12
- `p3717436738369618131__BEAUTY_功效` cat=beauty_and_personal_care attr=功效 pref=strong_srt_candidate triggers=12
- `p3580184154987456894__DIGITAL_贴膜特点` cat=digital_and_electronics attr=贴膜特点 pref=strong_srt_candidate triggers=12
- `p3649304698210641818__BEAUTY_是否电动` cat=beauty_and_personal_care attr=是否电动 pref=strong_srt_candidate triggers=12
- `p3663574409718972261__HOME_描述` cat=smart_home attr=描述 pref=strong_srt_candidate triggers=12
- `p3703683175160086673__SHOEBAG_是否商场同款` cat=shoes_and_bags attr=是否商场同款 pref=strong_srt_candidate triggers=12
- `p3649304698210641818__BEAUTY_使用方法` cat=beauty_and_personal_care attr=使用方法 pref=weak_srt_candidate triggers=12
- `p3716622350351990803__BEAUTY_功效` cat=beauty_and_personal_care attr=功效 pref=weak_srt_candidate triggers=12
- `p3772877260806292222__GEN_胀包问题` cat=general attr=胀包问题 pref=weak_srt_candidate triggers=12
- `p3772877260806292222__GEN_品牌` cat=general attr=品牌 pref=weak_srt_candidate triggers=12
- `p3675112863778865361__BABY_是否进口` cat=baby_kids_and_pets attr=是否进口 pref=weak_srt_candidate triggers=12
- `p3703683175160086673__SHOEBAG_材质` cat=shoes_and_bags attr=材质 pref=weak_srt_candidate triggers=12
- `p3663621559769277407__BABY_包装方式` cat=baby_kids_and_pets attr=包装方式 pref=weak_srt_candidate triggers=12
- `p3753596480485720540__BABY_是否进口` cat=baby_kids_and_pets attr=是否进口 pref=weak_srt_candidate triggers=12
- `p3703683175160086673__SHOEBAG_货号` cat=shoes_and_bags attr=货号 pref=weak_srt_candidate triggers=12
- `p3494583142998751397__HOME_刀具锋利度` cat=smart_home attr=<刀具锋利度> pref=weak_srt_candidate triggers=12
- `p3768024980667891731__SPORT_风格` cat=sports_and_outdoor attr=风格 pref=weak_srt_candidate triggers=12
- `p3494583142998751397__HOME_是否开刃` cat=smart_home attr=是否开刃 pref=weak_srt_candidate triggers=12
- `p3580184154987456894__DIGITAL_贴膜神器` cat=digital_and_electronics attr=贴膜神器 pref=weak_srt_candidate triggers=12
- `p3772134166922133792__JEWEL_是否镶嵌` cat=jewelry_and_collectibles attr=是否镶嵌 pref=weak_srt_candidate triggers=3
