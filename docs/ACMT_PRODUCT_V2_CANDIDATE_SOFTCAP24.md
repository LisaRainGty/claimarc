# A_cmt Product v2

## Summary
- input mentions: `151717`
- clean mentions: `108033`
- auxiliary mentions: `8153`
- dropped mentions: `35531`
- original cardinality: `{'products': 550, 'mean': 27.05, 'p50': 19, 'p90': 59, 'max': 199, 'pairs': 14879}`
- product-v2 cardinality: `{'products': 544, 'mean': 12.96, 'p50': 12, 'p90': 24, 'max': 38, 'pairs': 7052}`
- auxiliary pairs: `{'aux_price_dynamic': 585}`

## Interpretation
This view contracts A_cmt(p) to product-fact attributes before rebuilding
Stage B/C. Price, service, live-process, and perception-only mentions are
kept out of the clean schema and can be used as auxiliary/process analyses.

## Top Drop Reasons
- `drop_perception`: `471`
- `drop_uncertain_not_verifiable`: `4858`
- `drop_process`: `566`
- `drop_service_stagea_type`: `818`
- `drop_service`: `52`
- `drop_over_product_attr_cap`: `477`
