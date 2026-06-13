# A_cmt Product v2

## Summary
- input mentions: `151717`
- clean mentions: `103571`
- auxiliary mentions: `9608`
- dropped mentions: `38538`
- original cardinality: `{'products': 550, 'mean': 27.05, 'p50': 19, 'p90': 59, 'max': 199, 'pairs': 14879}`
- product-v2 cardinality: `{'products': 544, 'mean': 10.99, 'p50': 11, 'p90': 18, 'max': 18, 'pairs': 5979}`
- auxiliary pairs: `{'aux_price_dynamic': 590}`

## Interpretation
This view contracts A_cmt(p) to product-fact attributes before rebuilding
Stage B/C. Price, service, live-process, and perception-only mentions are
kept out of the clean schema and can be used as auxiliary/process analyses.

## Top Drop Reasons
- `drop_perception`: `473`
- `drop_uncertain_not_verifiable`: `5308`
- `drop_process`: `619`
- `drop_service_stagea_type`: `824`
- `drop_service`: `67`
- `drop_over_product_attr_cap`: `1019`
