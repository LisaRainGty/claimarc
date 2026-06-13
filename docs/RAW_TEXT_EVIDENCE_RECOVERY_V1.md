# Raw Text Evidence Recovery v1

- recovered evidence: `data/final/repaired_v1/raw_text_evidence_recovery_hq_product_v1.jsonl`
- diagnostic dataset: `data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_v1.jsonl`

## Summary
- `targets`: `727`
- `targets_source0`: `727`
- `schema_noisy`: `19`
- `recovered_any`: `134`
- `recovered_params`: `116`
- `recovered_ocr`: `21`
- `source0_after`: `593`
- `confidence_after`: `{'low': 1184, 'medium': 558, 'absent': 593, 'high': 29}`
- `label_source0_after`: `{'0:False': 1098, '1:False': 673, '0:True': 300, '1:True': 293}`
- `category_recovered`: `{'shoes_and_bags': 18, 'sports_and_outdoor': 11, 'baby_kids_and_pets': 22, 'general': 17, 'jewelry_and_collectibles': 2, 'apparel_and_underwear': 27, 'smart_home': 21, 'beauty_and_personal_care': 6, 'food_and_beverages': 5, 'digital_and_electronics': 5}`

## Notes
This is a high-recall candidate recovery step. It should be followed by
LLM/VLM adjudication before recovered evidence is promoted to a clean
benchmark.
