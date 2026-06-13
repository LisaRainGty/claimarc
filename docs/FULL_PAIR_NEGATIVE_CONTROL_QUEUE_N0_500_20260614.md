# Full Pair Negative-Control Queue v1

This queue samples claim-present rows likely to become natural negatives.
It does not assign labels; labels are rebuilt by the full-pair reviewer.

## Summary

- `input_rows`: `13769`
- `selected_rows`: `500`
- `excluded_pair_ids`: `329`
- `excluded_rows`: `329`
- `old_label_state`: `{'label_negative_claim_aligned_nonneg': 377, 'label_negative_no_aligned_review': 123}`
- `claim_state`: `{'claim_present_specific': 300, 'claim_present_review_needed': 200}`
- `evidence_state`: `{'evidence_multi_source': 264, 'evidence_single_source': 236}`
- `category`: `{'food_and_beverages': 21, 'beauty_and_personal_care': 39, 'baby_kids_and_pets': 97, 'shoes_and_bags': 64, 'apparel_and_underwear': 158, 'digital_and_electronics': 54, 'general': 42, 'jewelry_and_collectibles': 12, 'sports_and_outdoor': 7, 'smart_home': 6}`

## Examples

- `p3483616667710416469__FOOD_风味` cat=food_and_beverages attr=风味 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3772518315281482179__BEAUTY_功效` cat=beauty_and_personal_care attr=功效 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3772518315281482179__BEAUTY_使用方法` cat=beauty_and_personal_care attr=使用方法 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3649304698210641818__BEAUTY_功能` cat=beauty_and_personal_care attr=功能 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3649304698210641818__BEAUTY_产品名称` cat=beauty_and_personal_care attr=产品名称 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3649304698210641818__BEAUTY_品牌` cat=beauty_and_personal_care attr=品牌 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3772518315281482179__BEAUTY_功能` cat=beauty_and_personal_care attr=功能 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3772518315281482179__BEAUTY_产品名称` cat=beauty_and_personal_care attr=产品名称 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3482085810386379885__BABY_适用年龄` cat=baby_kids_and_pets attr=适用年龄 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3676842541531136387__BEAUTY_适用对象` cat=beauty_and_personal_care attr=适用对象 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3717412697340510500__BEAUTY_适用对象` cat=beauty_and_personal_care attr=适用对象 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3693461401977880924__BABY_产品名称` cat=baby_kids_and_pets attr=产品名称 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3587490886713489261__BEAUTY_适用对象` cat=beauty_and_personal_care attr=适用对象 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3721888693162737924__SHOEBAG_鞋跟高度` cat=shoes_and_bags attr=鞋跟高度 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3727479136529285164__BEAUTY_净含量` cat=beauty_and_personal_care attr=净含量 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3483616667710416469__FOOD_成分含量` cat=food_and_beverages attr=成分含量 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3645238708304752490__BABY_适用阶段` cat=baby_kids_and_pets attr=适用阶段 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3487313915229168499__APPAREL_款式` cat=apparel_and_underwear attr=款式 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3777908224368444145__DIGITAL_材质` cat=digital_and_electronics attr=材质 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3749284292551901241__DIGITAL_屏幕尺寸` cat=digital_and_electronics attr=屏幕尺寸 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
