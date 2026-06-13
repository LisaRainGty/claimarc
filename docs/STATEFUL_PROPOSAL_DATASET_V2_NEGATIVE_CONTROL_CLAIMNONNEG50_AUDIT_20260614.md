# Proposal Label/Claim/Evidence Consistency Audit v2

- dataset: `data/final/repaired_v1/stateful_proposal_dataset_v2_negative_control_claimnonneg50_all_20260614.jsonl`
- rows: `50`
- y_perception: `{'None': 24, '1': 9, '0': 17}`
- issue_counts: `{'positive_without_refute': 0, 'positive_without_claim': 0, 'negative_or_unobserved_with_refute': 0, 'objective_contradiction_without_refute': 16, 'silver_refute_not_positive': 0, 'claim_family_changed_label_risk': 0}`

## Issue Examples

### positive_without_refute
- none

### positive_without_claim
- none

### negative_or_unobserved_with_refute
- none

### objective_contradiction_without_refute
- `p3649304698210641818__BEAUTY_品牌` attr=品牌 state=repair_identity_claim_value y=None rel={}
- `p3676842541531136387__BEAUTY_适用对象` attr=适用对象 state=main_negative_support y=0 rel={'support': 1}
- `p3727479136529285164__BEAUTY_净含量` attr=净含量 state=main_negative_support y=0 rel={'support': 1}
- `p3777908224368444145__DIGITAL_材质` attr=材质 state=lowinfo_no_aligned_comment y=None rel={}
- `p3749284292551901241__DIGITAL_屏幕尺寸` attr=屏幕尺寸 state=lowinfo_no_aligned_comment y=None rel={}
- `p3716503433512091689__APPAREL_颜色分类` attr=颜色分类 state=lowinfo_no_aligned_comment y=None rel={}
- `p3635765391499890976__GEN_尺寸` attr=尺寸 state=lowinfo_no_aligned_comment y=None rel={}
- `p3679907962878558482__GEN_适用季节` attr=适用季节 state=main_negative_support y=0 rel={'support': 5}
- `p3661518415283271279__APPAREL_弹力` attr=弹力 state=main_negative_support y=0 rel={'support': 3}
- `p3794824930303017029__DIGITAL_功能` attr=功能 state=lowinfo_no_aligned_comment y=None rel={}

### silver_refute_not_positive
- none

### claim_family_changed_label_risk
- none
