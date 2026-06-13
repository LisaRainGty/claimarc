# Product-v2 Comment-triggered Claim Re-extraction Queue

## Summary
- rows: `526`
- products: `193`
- pairs: `526`
- priority: `{'P0': 268, 'P1': 169, 'P2': 89}`
- source_family: `{'numeric': 113, 'material': 120, 'identity_or_spec': 113, 'visual_or_boolean': 61, 'direct_text_match': 79, 'objective_name_only': 40}`
- skipped: `{'has_direct_claim': 1440, 'no_comment_trigger': 3155, 'low_priority_score': 100, 'no_text_trigger': 758}`

## Use
Use this queue before expanding the clean benchmark. Each row requires an
SRT-grounded claim re-extraction pass, then product-evidence verification,
then atomic consumer-signal alignment. Rows should not enter clean training
until all three states are available.
