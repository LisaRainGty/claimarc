# Proposal Triplet Alignment Audit v2

## Principle
This gate checks whether an already complete record is a natural
`(attribute-grounded claim, product evidence, proposal label)` triplet.
It does not remove hard samples by score and it does not relabel rows.
Rows that fail the gate are routed back to repair queues.

## Outputs
- audit: `data/final/repaired_v1/proposal_triplet_alignment_audit_v2_20260613.jsonl`
- aligned training pool: `data/final/repaired_v1/dataset_attrpol_proposal_triplet_aligned_v2_20260613.jsonl`
- aligned label-supported core: `data/final/repaired_v1/dataset_attrpol_proposal_triplet_aligned_label_supported_v2_20260613.jsonl`
- repair queue: `data/final/repaired_v1/proposal_triplet_alignment_repair_queue_v2_20260613.jsonl`

## Summary
- input rows: `910`
- status: `{'needs_repair_before_training': 451, 'triplet_aligned': 259, 'triplet_aligned_low_confidence_label': 200}`
- labels by status: `{'needs_repair_before_training': {'0': 313, '1': 138}, 'triplet_aligned': {'1': 151, '0': 108}, 'triplet_aligned_low_confidence_label': {'0': 200}}`
- issue counts: `{'product_evidence_alignment_review': 253, 'proposal_low_confidence_negative_label': 444, 'claim_attribute_alignment_review': 320}`
- attribute families: `{'numeric': 156, 'material': 89, 'attribute_value': 238, 'season': 23, 'visual_or_boolean': 128, 'function_effect': 69, 'style': 24, 'identity': 61, 'audience': 69, 'price': 53}`

## Interpretation
The aligned training pool is suitable for the next controlled model run
because every retained row has at least one attribute-specific SRT claim
and one attribute-specific product evidence item.  Low-confidence
proposal negatives are kept with their original sample weights rather
than discarded.  The label-supported core is only a robustness/audit view.

## Top Repair Examples
- `p3709750295312597707__SHOEBAG_件数` 件数 y=0 issues=['product_evidence_alignment_review', 'proposal_low_confidence_negative_label'] claim=沙色3839，只有45双线线，是的，拍到就能发，拍不到的话又要等了 | 234都是线线款，喜欢了234的话直接去买就可以了 evidence=ocr::2
- `p3768024980667891731__SPORT_品牌` 品牌 y=1 issues=['claim_attribute_alignment_review'] claim=王俊凯白露都在穿 而且宝宝现象门店这个 牌牌指导价格 同步赞卖是不是2868 evidence=params:品牌:骆驼火山
- `p3768024980667891731__SPORT_衣长` 衣长 y=1 issues=['claim_attribute_alignment_review'] claim=比5号链接相对来说板要短一堆裂 evidence=params:衣长:短款
- `p3768024980667891731__SPORT_适用人群` 适用人群 y=1 issues=['claim_attribute_alignment_review'] claim=165125斤买给M163135斤穿给A A肉大马迟到年为了其实5号领结我随便卖我想卖都可以但是我觉得一年到头大家给家人给自己你们值得更好的款式 | 160100斤买给A4175150斤买个插肉175125斤穿两个加因为我自己都在留1号领结 evidence=params:适用对象:情侣 | params:适用人群:成人
- `p3649304698210641818__BEAUTY_价格` <价格> y=0 issues=['product_evidence_alignment_review'] claim=今天下个，找1699宝贝。只要1699 | 今年是1699 evidence=ocr::100
- `p3649304698210641818__BEAUTY_优惠` 优惠 y=0 issues=['claim_attribute_alignment_review', 'product_evidence_alignment_review', 'proposal_low_confidence_negative_label'] claim=今天不仅仅是1699。而且会让陶哥有机会，是我们的一个800多带回家 evidence=ocr::直播间下单抽奢礼 | ocr::今日下单享
直播间下单立享
- `p3649304698210641818__BEAUTY_优惠力度` 优惠力度 y=0 issues=['claim_attribute_alignment_review', 'product_evidence_alignment_review', 'proposal_low_confidence_negative_label'] claim=拍到买刀真的就是赚刀了 | 有机会是800多带回家 evidence=ocr::半价
- `p3649304698210641818__BEAUTY_品牌` 品牌 y=0 issues=['claim_attribute_alignment_review'] claim=U-Lite已经做了11年 evidence=params:品牌:Ulike | ocr::Ulike
- `p3649304698210641818__BEAUTY_型号` 型号 y=0 issues=['claim_attribute_alignment_review', 'proposal_low_confidence_negative_label'] claim=所以说想要无首一款，能量比较大的。脱毛扫比较好，脱毛仪。直接选择我们新款的U-Lite AirSams | 直接选择我们新款的ULA K23 evidence=params:型号:UI06PR | ocr::Air3
- `p3649304698210641818__BEAUTY_套餐详情` 套餐详情 y=0 issues=['claim_attribute_alignment_review', 'product_evidence_alignment_review'] claim=没有任何的拆机费,试用费和折扣费 | 现在是买一送一 evidence=ocr::LED化妆镜+体精华油
沐浴啫喱*2
- `p3775015853804879952__BEAUTY_产品名称` 产品名称 y=0 issues=['claim_attribute_alignment_review', 'proposal_low_confidence_negative_label'] claim=直接选择我们新款U like L3 | 直接选择我们ULAK23 evidence=params:型号:9233 | ocr::电动挽脸器绞脸部脱器男唇部拔胡汗毛神器
- `p3775015853804879952__BEAUTY_价格` <价格> y=0 issues=['product_evidence_alignment_review', 'proposal_low_confidence_negative_label'] claim=今年是1699 | 平时日常价格1999 evidence=ocr::76
