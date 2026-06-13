# Full Pair LLM Pilot Queue v1

This queue is a stratified pilot for reconstruction protocol diagnosis, not a training dataset.

## Outputs

- queue: `data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl`
- report: `data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.report.json`

## Summary

- rows: `72`
- priority: `{'P0': 72}`
- prefilter state: `{'no_srt_candidate': 8, 'strong_srt_candidate': 24, 'very_weak_srt_candidate': 16, 'weak_srt_candidate': 24}`
- claim state: `{'claim_missing': 58, 'claim_present_review_needed': 14}`
- queue type: `{'full_claim_evidence_label_rebuild': 41, 'claim_reextract_label_rebuild': 31}`
- category: `{'apparel_and_underwear': 11, 'digital_and_electronics': 11, 'shoes_and_bags': 7, 'baby_kids_and_pets': 8, 'beauty_and_personal_care': 8, 'food_and_beverages': 6, 'general': 6, 'jewelry_and_collectibles': 5, 'smart_home': 5, 'sports_and_outdoor': 5}`

## Why This Sampling Matters

- Strong SRT candidates test whether the new prefilter recovers exact claim spans.
- Weak and very weak candidates test whether lexical overlap is too noisy.
- No-candidate rows test the boundary between true missing claims and schema/comment noise.
- Old labels are retained only for audit and are not exposed as target labels.

## Examples

- `p3750430700810141764__APPAREL_产地` state=no_srt_candidate claim=claim_missing cat=apparel_and_underwear attr=产地 top=0
- `p3750430700810141764__APPAREL_是否为有机食品` state=no_srt_candidate claim=claim_missing cat=apparel_and_underwear attr=是否为有机食品 top=0
- `p3750430700810141764__APPAREL_是否进口` state=no_srt_candidate claim=claim_missing cat=apparel_and_underwear attr=是否进口 top=0
- `p3775112844819956137__DIGITAL_功能` state=no_srt_candidate claim=claim_missing cat=digital_and_electronics attr=功能 top=0
- `p3775112844819956137__DIGITAL_描述相符` state=no_srt_candidate claim=claim_missing cat=digital_and_electronics attr=描述相符 top=0
- `p3775112844819956137__DIGITAL_适用人群` state=no_srt_candidate claim=claim_missing cat=digital_and_electronics attr=适用人群 top=0
- `p3689001967574712538__SHOEBAG_是否商场同款` state=no_srt_candidate claim=claim_missing cat=shoes_and_bags attr=是否商场同款 top=0
- `p3689001967574712538__SHOEBAG_材质` state=no_srt_candidate claim=claim_missing cat=shoes_and_bags attr=材质 top=0
- `p3702755032744198404__APPAREL_主要功能` state=strong_srt_candidate claim=claim_present_review_needed cat=apparel_and_underwear attr=主要功能 top=32
- `p3702755032744198404__APPAREL_镜片分类` state=strong_srt_candidate claim=claim_missing cat=apparel_and_underwear attr=镜片分类 top=23
- `p3731321152212172814__APPAREL_电源容量` state=strong_srt_candidate claim=claim_missing cat=apparel_and_underwear attr=电源容量 top=22
- `p3482085810386379885__BABY_品牌` state=strong_srt_candidate claim=claim_missing cat=baby_kids_and_pets attr=品牌 top=28
- `p3660066402737519648__BABY_功效` state=strong_srt_candidate claim=claim_missing cat=baby_kids_and_pets attr=功效 top=53
- `p3660066402737519648__BABY_品牌` state=strong_srt_candidate claim=claim_missing cat=baby_kids_and_pets attr=品牌 top=60
- `p3649304698210641818__BEAUTY_是否电动` state=strong_srt_candidate claim=claim_missing cat=beauty_and_personal_care attr=是否电动 top=28
- `p3709009939570753771__BEAUTY_功效` state=strong_srt_candidate claim=claim_missing cat=beauty_and_personal_care attr=功效 top=22
- `p3717436738369618131__BEAUTY_功效` state=strong_srt_candidate claim=claim_missing cat=beauty_and_personal_care attr=功效 top=20
- `p3580184154987456894__DIGITAL_贴膜特点` state=strong_srt_candidate claim=claim_missing cat=digital_and_electronics attr=贴膜特点 top=25
- `p3752312351777488914__DIGITAL_品牌` state=strong_srt_candidate claim=claim_missing cat=digital_and_electronics attr=品牌 top=29
- `p3794823514993852677__DIGITAL_品牌` state=strong_srt_candidate claim=claim_missing cat=digital_and_electronics attr=品牌 top=22
