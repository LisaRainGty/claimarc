# Product-v2 Comment-triggered Claim Re-extraction Queue

## Summary
- rows: `567`
- products: `197`
- pairs: `567`
- priority: `{'P0': 275, 'P1': 190, 'P2': 102}`
- source_family: `{'numeric': 121, 'material': 128, 'identity_or_spec': 117, 'visual_or_boolean': 64, 'direct_text_match': 89, 'objective_name_only': 48}`
- skipped: `{'has_direct_claim': 1440, 'no_comment_trigger': 4046, 'low_priority_score': 124, 'no_text_trigger': 875}`

## Use
Use this queue before expanding the clean benchmark. Each row requires an
SRT-grounded claim re-extraction pass, then product-evidence verification,
then atomic consumer-signal alignment. Rows should not enter clean training
until all three states are available.
