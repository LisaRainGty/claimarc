# Claim Attribute Validation v2

## Summary
- products: `544`
- product-v2 pairs: `5979`
- pairs with direct claim: `1440`
- existing claims read: `11692`
- validation status: `{'direct': 4678, 'too_vague': 1585, 'promo_or_order': 131, 'wrong_attribute': 9}`

## Interpretation
This is a deterministic bridge over the old B1 claim extraction.  It is
not a substitute for rerunning B1 with product-v2 A_cmt, but it shows how
many current claims survive a product-only schema and a basic relation gate.
