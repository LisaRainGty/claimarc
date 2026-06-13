# CLAIMARC Data Regeneration Protocol v2

Date: 2026-06-12

## Motivation

The raw-to-training audit found that the original pipeline mixed several
different uncertainty sources:

- attribute-level consumer polarity was sometimes replaced by global review
  polarity;
- malformed `FREE_...` attribute ids were not resolved by Stage A2/A3;
- product evidence absence (`source0`) was treated like a normal low-confidence
  source rather than a separate coverage failure;
- some schema items were not product facts, but livestream display, purchase
  process, service, or broad satisfaction dimensions.

The repair goal is to keep the retrieval-augmented contrastive learning
mechanism, while making supervision easier to learn and easier to defend.

## Deterministic Repairs

Source pipeline patches:

- `stage_a/a1_extract_aspects.py`
  - normalizes `FREE_foo`, `FREE:foo`, `FREE：foo` into `FREE::foo`;
  - preserves short `review_text` context for later claim-comment alignment;
  - keeps attribute-level `polarity` separate from `review_polarity`.
- `stage_a/a2_aggregate_free.py` and `stage_a/a3_resolve_labels.py`
  - resolve all common FREE prefix variants, reducing avoidable unresolved
    attributes.
- `stage_b/b4_b5_align.py`
  - aggregates comment polarity from Stage A attribute-level `polarity`, not
    global review sentiment.

Versioned data outputs:

- `data/final/repaired_v1/dataset_attrpol_hq_product_v1.jsonl`
  - clean main benchmark; 2364 product-attribute pairs; no room leakage.
- `data/final/repaired_v1/dataset_attrpol_product_train_v2.jsonl`
  - 3103 claim-bearing product-attribute pairs; quality-weighted training pool.
- `data/final/repaired_v1/regeneration_manifest_v1.jsonl`
  - targeted queue for LLM/VLM/claim-extraction repair.

## v2 Training Pool

`build_product_training_v2.py` keeps product-scope claimful pairs and adds:

- `_quality_bucket`: deterministic clean/silver/weak tier;
- `_source_count`: number of PARAM/OCR/VLM sources;
- `_attribute_noise_flags`: schema items needing review, such as livestream
  display, purchase process, broad product quality, authenticity, and effect
  claims;
- `_regeneration_actions`: repair actions needed before promotion to a clean
  benchmark;
- quality-adjusted `c`: weak/source0/noisy rows are retained for scale but
  down-weighted.

This pool is for training augmentation. The HQ product set remains the clean
evaluation benchmark until manifest items are re-adjudicated.

## LLM Regeneration Adjudication

`llm_regeneration_adjudicate_v1.py` consumes the manifest and asks an LLM to
judge each flagged pair without seeing `y`, `c`, or `split`.

Input dimensions:

- grounded livestream claim;
- product evidence from PARAM/OCR/VLM;
- attribute-level consumer comment snippets from repaired Stage B records;
- manifest repair actions and schema noise flags.

Output dimensions:

- `claim_quality`: clear, mixed, garbled, or no claim;
- `attribute_quality`: product attribute, service/process, subjective/noisy, or
  wrong attribute;
- `product_evidence_state`: supported, contradicted, insufficient, or not
  verifiable;
- `consumer_signal`: refutes, supports, mixed, irrelevant, or insufficient;
- `label_recommendation`: positive risk, negative clean, or drop/regenerate;
- `recommended_actions`: keep, rerun claim extraction, rerun product evidence,
  schema review, or human review.

The adjudicator is a curation instrument. Any resulting label changes must be
reported as LLM-assisted data repair, not as hidden test-time supervision.

## Next Experimental Use

1. Run CLAIMARC-only CV on the 3103-row v2 training pool to test whether
   quality weighting and source-aware contrastive training improve stability.
2. Compare OOF diagnostics against the clean HQ product result, especially
   `source0`, `medium` confidence, and `PO` evidence groups.
3. If source0 remains weak, run LLM regeneration on priority-1 manifest rows:
   claimful source0 positives, missing-claim risk pairs, and schema-noisy
   product attributes.
4. Promote only adjudicated clear/silver rows into a future
   `dataset_attrpol_product_train_v3` or a new clean benchmark.
