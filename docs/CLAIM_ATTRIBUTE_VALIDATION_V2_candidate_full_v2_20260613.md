# Claim Attribute Validation v2

## Summary
- products: `544`
- product-v2 pairs: `6914`
- pairs with direct claim: `861`
- existing claims read: `6169`
- validation status: `{'direct': 2434, 'too_vague': 3373, 'wrong_attribute': 303, 'promo_or_order': 59}`

## Interpretation
This is a deterministic bridge over the old B1 claim extraction.  It is
not a substitute for rerunning B1 with product-v2 A_cmt, but it shows how
many current claims survive a product-only schema and a basic relation gate.
