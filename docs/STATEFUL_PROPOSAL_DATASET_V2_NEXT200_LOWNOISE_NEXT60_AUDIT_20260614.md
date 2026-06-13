# Proposal Label/Claim/Evidence Consistency Audit v2

- dataset: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_lownoise_next60_all_20260614.jsonl`
- rows: `60`
- y_perception: `{'None': 25, '1': 30, '0': 5}`
- issue_counts: `{'positive_without_refute': 0, 'positive_without_claim': 0, 'negative_or_unobserved_with_refute': 2, 'objective_contradiction_without_refute': 5, 'silver_refute_not_positive': 0, 'claim_family_changed_label_risk': 0}`

## Issue Examples

### positive_without_refute
- none

### positive_without_claim
- none

### negative_or_unobserved_with_refute
- `p3784789614443760279__FOOD_是否临期` attr=是否临期 state=repair_missing_claim y=None rel={'refute': 1}
- `p3703129246668030235__HOME_控制方式` attr=控制方式 state=repair_missing_claim y=None rel={'refute': 3}

### objective_contradiction_without_refute
- `p3743612721522933924__FOOD_是否临期` attr=是否临期 state=lowinfo_no_aligned_comment y=None rel={}
- `p3784789614443760279__FOOD_净含量` attr=净含量 state=lowinfo_no_aligned_comment y=None rel={}
- `p3759321650240290980__GEN_大纸张` attr=大纸张 state=lowinfo_no_aligned_comment y=None rel={}
- `p3671415216744300934__FOOD_产品名称` attr=产品名称 state=main_negative_support y=0 rel={'support': 7}
- `p3716987214157185084__GEN_品牌` attr=品牌 state=repair_identity_claim_value y=None rel={}

### silver_refute_not_positive
- none

### claim_family_changed_label_risk
- none
