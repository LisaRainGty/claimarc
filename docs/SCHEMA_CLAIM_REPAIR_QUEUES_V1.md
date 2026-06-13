# Schema And Claim Repair Queues v1

## Summary
- schema queue rows: `10652`
- claim re-extract rows: `3404`
- A_cmt cardinality: `{'products': 550, 'mean': 30.33, 'p50': 21, 'p90': 68, 'max': 218, 'high_cardinality_threshold': 50, 'products_ge_threshold': 102}`
- schema priorities: `{'P0': 2038, 'P1': 2308, 'P2': 6306}`
- claim priorities: `{'P0': 1559, 'P1': 1845}`

## Interpretation
Use the schema queue to contract overly broad or evaluative attributes
before rebuilding Stage B/C. Use the claim re-extraction queue only for
pairs where review text explicitly suggests a missed livestream claim.
