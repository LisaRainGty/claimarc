# Proposal Label/Claim/Evidence Consistency Audit v2

- dataset: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_all_20260614.jsonl`
- rows: `99`
- y_perception: `{'None': 42, '0': 8, '1': 49}`
- issue_counts: `{'positive_without_refute': 0, 'positive_without_claim': 0, 'negative_or_unobserved_with_refute': 5, 'objective_contradiction_without_refute': 2, 'silver_refute_not_positive': 0, 'claim_family_changed_label_risk': 0}`

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

### objective_contradiction_without_refute
- `p3549451361445893022__GEN_品牌` attr=品牌 state=repair_identity_claim_value y=None rel={}
- `p3683596134795837638__FOOD_商品质量` attr=<商品质量> state=silver_subjective_eval_attribute y=None rel={}

### silver_refute_not_positive
- none

### claim_family_changed_label_risk
- none
