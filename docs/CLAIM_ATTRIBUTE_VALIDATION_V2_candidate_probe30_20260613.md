# Claim Attribute Validation v2

## Summary
- products: `544`
- product-v2 pairs: `6914`
- pairs with direct claim: `71`
- existing claims read: `589`
- validation status: `{'too_vague': 318, 'direct': 237, 'wrong_attribute': 24, 'promo_or_order': 2}`

## Interpretation
This is a deterministic bridge over the old B1 claim extraction.  It is
not a substitute for rerunning B1 with product-v2 A_cmt, but it shows how
many current claims survive a product-only schema and a basic relation gate.
