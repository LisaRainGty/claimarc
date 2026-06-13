# Full Pair Negative-Control Queue v1

This queue samples claim-present rows likely to become natural negatives.
It does not assign labels; labels are rebuilt by the full-pair reviewer.

## Summary

- `input_rows`: `13769`
- `selected_rows`: `30`
- `excluded_pair_ids`: `269`
- `excluded_rows`: `269`
- `old_label_state`: `{'label_negative_claim_aligned_nonneg': 30}`
- `claim_state`: `{'claim_present_specific': 19, 'claim_present_review_needed': 11}`
- `evidence_state`: `{'evidence_multi_source': 30}`
- `category`: `{'digital_and_electronics': 7, 'jewelry_and_collectibles': 1, 'shoes_and_bags': 5, 'apparel_and_underwear': 7, 'baby_kids_and_pets': 2, 'food_and_beverages': 1, 'general': 2, 'beauty_and_personal_care': 5}`

## Examples

- `p3719463917936837098__DIGITAL_材质` cat=digital_and_electronics attr=材质 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3772134272358547681__JEWEL_尺寸规格` cat=jewelry_and_collectibles attr=尺寸/规格 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3737274784296337568__SHOEBAG_材质` cat=shoes_and_bags attr=材质 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3791067731332301269__SHOEBAG_材质` cat=shoes_and_bags attr=材质 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3791978732667928854__SHOEBAG_容量` cat=shoes_and_bags attr=容量 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3791987586717843497__SHOEBAG_工艺` cat=shoes_and_bags attr=工艺 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3794824930303017029__DIGITAL_保修期` cat=digital_and_electronics attr=保修期 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3718539716002447832__APPAREL_弹力` cat=apparel_and_underwear attr=弹力 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3784732424798340018__APPAREL_适用季节` cat=apparel_and_underwear attr=适用季节 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3790344607175409821__APPAREL_尺码` cat=apparel_and_underwear attr=尺码 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3641721099965847230__BABY_净含量` cat=baby_kids_and_pets attr=净含量 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3662085582886158777__DIGITAL_风格` cat=digital_and_electronics attr=风格 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3794824930303017029__DIGITAL_制冷量` cat=digital_and_electronics attr=制冷量 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3794824930303017029__DIGITAL_匹数` cat=digital_and_electronics attr=匹数 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3794824930303017029__DIGITAL_型号` cat=digital_and_electronics attr=型号 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3668409871310651466__FOOD_溶解性` cat=food_and_beverages attr=溶解性 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3751877224438038787__GEN_适用季节` cat=general attr=适用季节 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3754321247501156800__GEN_规格` cat=general attr=规格 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3759361621437317216__SHOEBAG_工艺` cat=shoes_and_bags attr=工艺 old=label_negative_claim_aligned_nonneg claim=claim_present_specific evidence=evidence_multi_source
- `p3649304698210641818__BEAUTY_适用对象` cat=beauty_and_personal_care attr=适用对象 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
