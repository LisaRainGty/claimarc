# Product-v2 Comment-triggered Claim Re-extraction Queue

## Summary
- rows: `1199`
- products: `300`
- pairs: `1199`
- priority: `{'P0': 367, 'P1': 515, 'P2': 317}`
- source_family: `{'numeric': 268, 'material': 233, 'identity_or_spec': 326, 'visual_or_boolean': 114, 'direct_text_match': 181, 'objective_name_only': 77}`
- skipped: `{'has_direct_claim': 1440, 'no_comment_trigger': 3155, 'low_priority_score': 185}`

## Use
Use this queue before expanding the clean benchmark. Each row requires an
SRT-grounded claim re-extraction pass, then product-evidence verification,
then atomic consumer-signal alignment. Rows should not enter clean training
until all three states are available.
