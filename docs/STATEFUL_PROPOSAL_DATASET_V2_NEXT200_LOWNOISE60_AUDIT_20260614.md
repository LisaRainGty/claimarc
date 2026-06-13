# Proposal Label/Claim/Evidence Consistency Audit v2

- dataset: `data/final/repaired_v1/stateful_proposal_dataset_v2_next200_lownoise60_all_20260614.jsonl`
- rows: `60`
- y_perception: `{'1': 31, 'None': 28, '0': 1}`
- issue_counts: `{'positive_without_refute': 0, 'positive_without_claim': 0, 'negative_or_unobserved_with_refute': 1, 'objective_contradiction_without_refute': 2, 'silver_refute_not_positive': 0, 'claim_family_changed_label_risk': 0}`

## Issue Examples

### positive_without_refute
- none

### positive_without_claim
- none

### negative_or_unobserved_with_refute
- `p3724404517601673447__BABY_面料材质` attr=面料材质 state=repair_missing_claim y=None rel={'refute': 1}

### objective_contradiction_without_refute
- `p3768053602246066626__FOOD_价格` attr=价格 state=lowinfo_no_aligned_comment y=None rel={}
- `p3720772695445602597__GEN_尺寸` attr=尺寸 state=lowinfo_no_aligned_comment y=None rel={}

### silver_refute_not_positive
- none

### claim_family_changed_label_risk
- none
