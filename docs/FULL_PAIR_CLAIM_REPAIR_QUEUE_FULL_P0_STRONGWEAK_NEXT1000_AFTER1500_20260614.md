# Full Pair Claim Repair Queue v1

This queue routes claim-missing full-pair reviews to exact SRT re-extraction.
It preserves hard/source-missing rows and does not change labels.

## Inputs

- queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- reviews: ``

## Outputs

- claim repair queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next1000_after1500_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next1000_after1500_v1_20260614.report.json`

## Summary

- `review_rows`: `0`
- `selected_rows`: `1000`
- `limit`: `1000`
- `excluded_pair_ids`: `1820`
- `excluded_rows`: `1820`
- `missing_queue_rows`: `0`
- `prefilter_state`: `{'weak_srt_candidate': 1000}`
- `source_review_action`: `{'seed_claim_repair': 1000}`
- `trigger_comment_bucket`: `{'5+': 483, '2-4': 517}`
- `category`: `{'beauty_and_personal_care': 169, 'general': 201, 'baby_kids_and_pets': 179, 'shoes_and_bags': 79, 'smart_home': 51, 'sports_and_outdoor': 39, 'digital_and_electronics': 30, 'food_and_beverages': 130, 'apparel_and_underwear': 117, 'jewelry_and_collectibles': 5}`
- `old_label_state`: `{'label_negative_no_aligned_review': 915, 'label_positive_claim_aligned_neg': 81, 'label_negative_claim_aligned_nonneg': 4}`

## Examples

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
- `p3679197849955992032__DIGITAL_安装方式` cat=digital_and_electronics attr=安装方式 pref=weak_srt_candidate triggers=12
- `p3768053602246066626__FOOD_风味` cat=food_and_beverages attr=风味 pref=weak_srt_candidate triggers=12
- `p3494583142998751397__HOME_是否开刃` cat=smart_home attr=是否开刃 pref=weak_srt_candidate triggers=12
- `p3768024980667891731__SPORT_质量` cat=sports_and_outdoor attr=质量 pref=weak_srt_candidate triggers=12
- `p3731321152212172814__APPAREL_电池类型` cat=apparel_and_underwear attr=电池类型 pref=weak_srt_candidate triggers=12
- `p3768630107141439951__APPAREL_面料材质` cat=apparel_and_underwear attr=面料材质 pref=weak_srt_candidate triggers=12
- `p3675112863778865361__BABY_包装方式` cat=baby_kids_and_pets attr=包装方式 pref=weak_srt_candidate triggers=12
- `p3580184154987456894__DIGITAL_贴膜神器` cat=digital_and_electronics attr=贴膜神器 pref=weak_srt_candidate triggers=12
- `p3483616667710416469__FOOD_套餐详情` cat=food_and_beverages attr=套餐详情 pref=weak_srt_candidate triggers=12
