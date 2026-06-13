# Full Pair Negative-Control Queue v1

This queue samples claim-present rows likely to become natural negatives.
It does not assign labels; labels are rebuilt by the full-pair reviewer.

## Summary

- `input_rows`: `13769`
- `selected_rows`: `120`
- `excluded_pair_ids`: `544`
- `excluded_rows`: `544`
- `old_label_state`: `{'label_negative_claim_aligned_nonneg': 120}`
- `claim_state`: `{'claim_present_review_needed': 60, 'claim_present_specific': 60}`
- `evidence_state`: `{'evidence_multi_source': 60, 'evidence_single_source': 60}`
- `category`: `{'beauty_and_personal_care': 7, 'baby_kids_and_pets': 35, 'apparel_and_underwear': 41, 'digital_and_electronics': 9, 'shoes_and_bags': 16, 'general': 7, 'food_and_beverages': 3, 'sports_and_outdoor': 2}`

## Examples

- `p3701432060234301842__BEAUTY_类型` cat=beauty_and_personal_care attr=类型 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3808361039184134380__BABY_面料材质` cat=baby_kids_and_pets attr=面料材质 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3702418478913945969__APPAREL_面料材质` cat=apparel_and_underwear attr=面料材质 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3796986711830036824__APPAREL_净含量` cat=apparel_and_underwear attr=净含量 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3752312351777488914__DIGITAL_能效等级` cat=digital_and_electronics attr=能效等级 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3789796805546475969__APPAREL_面料材质` cat=apparel_and_underwear attr=面料材质 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3784732424798340018__APPAREL_款式` cat=apparel_and_underwear attr=款式 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3768974357519533029__BABY_面料材质` cat=baby_kids_and_pets attr=面料材质 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3791987097259343906__SHOEBAG_材质` cat=shoes_and_bags attr=材质 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3784732424798340018__APPAREL_鞋跟高度` cat=apparel_and_underwear attr=鞋跟高度 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3810008932236263827__BABY_款式` cat=baby_kids_and_pets attr=款式 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3708097016463753496__APPAREL_款式` cat=apparel_and_underwear attr=款式 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3720047646866538676__APPAREL_款式` cat=apparel_and_underwear attr=款式 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3791978732667928854__SHOEBAG_皮质特征` cat=shoes_and_bags attr=皮质特征 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3693461401977880924__BABY_是否为特殊用途化妆品` cat=baby_kids_and_pets attr=是否为特殊用途化妆品 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3649304698210641818__BEAUTY_结构及组成` cat=beauty_and_personal_care attr=结构及组成 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3714741768856666516__APPAREL_厚度` cat=apparel_and_underwear attr=厚度 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3715531422216355864__APPAREL_款式` cat=apparel_and_underwear attr=款式 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3774697498044662004__APPAREL_款式` cat=apparel_and_underwear attr=款式 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
- `p3715349622693167329__BABY_数量` cat=baby_kids_and_pets attr=数量 old=label_negative_claim_aligned_nonneg claim=claim_present_review_needed evidence=evidence_multi_source
