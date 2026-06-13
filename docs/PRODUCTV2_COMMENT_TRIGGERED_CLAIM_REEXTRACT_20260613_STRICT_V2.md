# Product-v2 Comment-triggered Claim Re-extraction Queue

## Summary
- rows: `692`
- products: `244`
- pairs: `692`
- priority: `{'P0': 348, 'P1': 222, 'P2': 122}`
- source_family: `{'numeric': 162, 'material': 161, 'visual_or_boolean': 98, 'identity_or_spec': 100, 'direct_text_match': 123, 'objective_name_only': 48}`
- skipped: `{'no_comment_trigger': 4035, 'low_priority_score': 115, 'identity_without_product_raw_hit': 196, 'no_text_trigger': 905, 'unspoken_or_promo_attribute': 36}`

## Use
Use this queue before expanding the clean benchmark. Each row requires an
SRT-grounded claim re-extraction pass, then product-evidence verification,
then atomic consumer-signal alignment. Rows should not enter clean training
until all three states are available.
