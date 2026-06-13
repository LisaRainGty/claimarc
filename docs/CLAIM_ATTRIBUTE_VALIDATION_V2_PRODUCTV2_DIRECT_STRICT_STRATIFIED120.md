# Claim Attribute Validation v2

## Summary
- products: `544`
- product-v2 pairs: `5979`
- pairs with direct claim: `306`
- existing claims read: `1502`
- validation status: `{'too_vague': 769, 'direct': 701, 'promo_or_order': 31, 'wrong_attribute': 1}`

## Interpretation
This is a deterministic bridge over the old B1 claim extraction.  It is
not a substitute for rerunning B1 with product-v2 A_cmt, but it shows how
many current claims survive a product-only schema and a basic relation gate.
