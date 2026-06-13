# Raw-Text LLM Curated Dataset v1

- dataset: `data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_v1.jsonl`
- input: `data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_v1.jsonl`
- adjudication: `data/final/repaired_v1/llm_recovered_evidence_adjudication_v1.jsonl`

## Summary
- input rows: `2364`
- adjudicated rows: `134`
- output rows: `2328`
- dropped rows: `36`
- input labels: `{0: 1398, 1: 966}`
- output labels: `{0: 1368, 1: 960}`
- source0 output: `593`
- changes: `{'keep_adjudicated': 98, 'drop': 36, 'flip_0_to_1': 25, 'flip_1_to_0': 10}`
- transitions: `{'1->1': 48, '1->drop': 21, '0->drop': 15, '0->1': 25, '0->0': 15, '1->0': 10}`
- rules: `{'evidence_and_consumer_refute': 25, 'consumer_refutes_claim': 8, 'mixed_or_insufficient': 29, 'evidence_refutes_claim': 40, 'evidence_and_consumer_support': 24, 'bad_claim_or_attribute': 7, 'evidence_supports_no_consumer_refute': 1}`

## Curation Rule
The LLM final label recommendation is not used directly. The script keeps
only clear product-attribute rows and derives labels from atomic evidence
and consumer-response states: contradicted/refuting cases become risk
positives; supported and non-refuting cases become clean negatives;
mixed, insufficient, service, subjective, or malformed rows are removed
from this strict benchmark and remain candidates for regeneration.
