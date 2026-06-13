# Proposal Label/Claim/Evidence Consistency Audit v2

- dataset: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_next500p151l60_all_20260614.jsonl`
- rows: `404`
- y_perception: `{'None': 183, '0': 51, '1': 170}`
- issue_counts: `{'positive_without_refute': 0, 'positive_without_claim': 0, 'negative_or_unobserved_with_refute': 11, 'objective_contradiction_without_refute': 25, 'silver_refute_not_positive': 0, 'claim_family_changed_label_risk': 0}`

## Issue Examples

### positive_without_refute
- none

### positive_without_claim
- none

### negative_or_unobserved_with_refute
- `p3549451361445893022__GEN_包数` attr=包数 state=repair_missing_claim y=None rel={'refute': 2}
- `p3703129246668030235__HOME_安装方式` attr=安装方式 state=repair_missing_claim y=None rel={'refute': 1}
- `p3657116391363977564__GEN_品牌` attr=品牌 state=repair_missing_claim y=None rel={'refute': 1}
- `p3580184154987456894__DIGITAL_贴膜特点` attr=贴膜特点 state=repair_missing_claim y=None rel={'refute': 5}
- `p3768024980667891731__SPORT_是否瑕疵` attr=是否瑕疵 state=repair_missing_claim y=None rel={'refute': 3}
- `p3724404517601673447__BABY_面料材质` attr=面料材质 state=repair_missing_claim y=None rel={'refute': 1}
- `p3784789614443760279__FOOD_是否临期` attr=是否临期 state=repair_missing_claim y=None rel={'refute': 1}
- `p3703129246668030235__HOME_控制方式` attr=控制方式 state=repair_missing_claim y=None rel={'refute': 3}
- `p3784732424798340018__APPAREL_适用季节` attr=适用季节 state=repair_missing_claim y=None rel={'refute': 1}
- `p3705157179901345903__BEAUTY_是否临期` attr=是否临期 state=repair_missing_claim y=None rel={'refute': 1}

### objective_contradiction_without_refute
- `p3549451361445893022__GEN_品牌` attr=品牌 state=repair_identity_claim_value y=None rel={}
- `p3683596134795837638__FOOD_商品质量` attr=<商品质量> state=silver_subjective_eval_attribute y=None rel={}
- `p3768053602246066626__FOOD_价格` attr=价格 state=lowinfo_no_aligned_comment y=None rel={}
- `p3720772695445602597__GEN_尺寸` attr=尺寸 state=lowinfo_no_aligned_comment y=None rel={}
- `p3649304698210641818__BEAUTY_品牌` attr=品牌 state=repair_identity_claim_value y=None rel={}
- `p3676842541531136387__BEAUTY_适用对象` attr=适用对象 state=main_negative_support y=0 rel={'support': 1}
- `p3727479136529285164__BEAUTY_净含量` attr=净含量 state=main_negative_support y=0 rel={'support': 1}
- `p3777908224368444145__DIGITAL_材质` attr=材质 state=lowinfo_no_aligned_comment y=None rel={}
- `p3749284292551901241__DIGITAL_屏幕尺寸` attr=屏幕尺寸 state=lowinfo_no_aligned_comment y=None rel={}
- `p3716503433512091689__APPAREL_颜色分类` attr=颜色分类 state=lowinfo_no_aligned_comment y=None rel={}

### silver_refute_not_positive
- none

### claim_family_changed_label_risk
- none
