# Data Reconstruction Progress: Product-v2 Gates

## Why This Round

The main failure is upstream: `A_cmt(p)` was broader than the methodology assumed.  The expected width was about 5-15 attributes per product, but the repaired Stage-A view still produced 14,879 product-attribute pairs, mean 27.05, p90 59, max 199.  Service/process/perception attributes then contaminated LangExtract, product evidence extraction, and weak labels.

## New Artifacts

- `src/data_quality/build_acmt_product_v2.py`
  - Writes a parallel `data/processed/stageB_product_v2/` directory.
  - Contracts `A_cmt(p)` to product-fact attributes.
  - Moves dynamic price/service/process/perception attributes into auxiliary/drop pools.
- `src/data_quality/validate_claim_attribute_v2.py`
  - Reuses old grounded `claim_list` only when the claim belongs to the product-v2 schema and is plausibly a product-fact claim.
  - Builds `pair_skeleton_product_v2.jsonl`.
- `src/data_quality/build_product_v2_bridge_dataset.py`
  - Diagnostic bridge only: product-v2 pairs + old labels/facts.
- `src/data_quality/build_claim_reextract_dataset_v4.py`
  - Promotes strict comment-triggered re-extraction verifications into clean rows by fixed rules.
- `src/data_quality/build_expansion_candidate_pools_v4.py`
  - Builds high-value LLM/VLM queues without mutating training data.

## Product-v2 A_cmt Gate

Input: `data/processed/stageA_repaired_v1/resolved_aspects_schema_clean_v1.jsonl`

Output:

- `data/processed/stageB_product_v2/acmt_product_v2.json`
- `data/processed/stageB_product_v2/resolved_aspects_product_v2.jsonl`
- `data/processed/stageB_product_v2/resolved_aspects_aux_v2.jsonl`
- `data/processed/stageB_product_v2/acmt_product_v2_drop_audit.jsonl`
- `data/final/repaired_v1/acmt_product_v2_report.json`

Key numbers:

- Original repaired Stage-A pairs: 14,879
- Product-v2 pairs: 5,979
- Mean attributes per product: 27.05 -> 10.99
- p90 attributes per product: 59 -> 18
- Max attributes per product: 199 -> 18
- Clean mentions retained: 103,571
- Dynamic-price auxiliary pairs: 590

Interpretation: this restores the candidate schema to the methodological range and removes most service/process/perception leakage before Stage B/C.

## Claim-Attribute Validation

Output:

- `data/processed/stageB_product_v2/claim_attribute_validation_v2.jsonl`
- `data/processed/stageB_product_v2/pair_skeleton_product_v2.jsonl`
- `data/final/repaired_v1/claim_attribute_validation_v2_report.json`

Key numbers:

- Product-v2 pairs: 5,979
- Existing grounded claims read: 11,692
- Claims within product-v2 schema: 6,403
- Direct product-fact claims retained: 4,678
- Product-v2 pairs with direct SRT claim: 1,440
- Filtered as too vague/promo/wrong attribute: 1,725

Interpretation: old B1 contains reusable grounded claims, but full B1 should still be rerun with product-v2 `A_cmt(p)` because 4,539 product-v2 pairs lack direct claims under the old extraction.

## Bridge Learnability

Diagnostic bridge datasets used product-v2 schema with old labels/facts.  Learnability was poor:

- sourceful: AP 0.0941, AUROC 0.6768, Macro-F1 0.5324
- claimful_sourceful: AP 0.2185, AUROC 0.5496, Macro-F1 0.5150
- claimful_sourceful_weighted: AP 0.4290, AUROC 0.5689, Macro-F1 0.5433

Interpretation: source schema contraction alone is not enough.  Old B4 labels and old fact extraction cannot be blindly reused; the final v2 dataset must rerun claim extraction, evidence extraction, review alignment, and fixed-rule adjudication from atomic states.

## Comment-Triggered Re-Extraction Pilot

Expansion pool:

- `claim_reextract`: 3,404 candidates
- strict high-value verification queue: 629 candidates

Pilot 20 with strict LLM/VLM triad verification:

- risk_candidate: 2
- rerun_more_evidence: 2
- drop: 16

Promoted pilot rows:

- `data/final/repaired_v1/dataset_claim_reextract_v4_pilot_merged.jsonl`
- n: 2,095 = 2,093 base + 2 strict risk rows

Interpretation: many explicit complaint rows refute title/detail-page expectations or product experience rather than a recoverable SRT claim.  They should not enter the main CLAIMARC benchmark unless a live claim is found.  They can become auxiliary/page-promotion-risk data.

## Top-120 Re-Extraction Check

Strict LLM/VLM triad verification on the first 120 priority candidates produced:

- drop: 81
- rerun_more_evidence: 16
- risk_candidate: 22
- clean_candidate: 1

Directly merging all promotable rows yielded:

- `dataset_claim_reextract_v4_top120_merged.jsonl`
- 2,116 rows = 2,093 base + 23 promoted
- promoted labels: 22 risk, 1 clean

After adding a product-v2 schema gate to `build_claim_reextract_dataset_v4.py`:

- `dataset_claim_reextract_v4_top120_productv2_merged.jsonl`
- 2,113 rows = 2,093 base + 20 promoted
- promoted labels: 19 risk, 1 clean
- 22 verification rows were kept auxiliary because the pair was outside `acmt_product_v2.json`.

Fast learnability diagnostics:

- base `source_recovered_v3_dropunresolved`: AP 0.8467, AUROC 0.9389, Macro-F1 0.9071
- top120 merged: AP 0.8381, AUROC 0.9282, Macro-F1 0.8897
- top120 product-v2 gated: AP 0.8269, AUROC 0.9229, Macro-F1 0.8823

Interpretation: the verified top120 rows are valuable, but not as unconditional benchmark expansion.  They are hard, boundary-shifting cases and should be used as auxiliary contrastive/retrieval stress data, or fed into a fuller product-v2 rerun.  The current strongest main benchmark remains `source_recovered_v3_dropunresolved`.

## Product-v2 B1 Rerun Pilot

A 20-product pilot was run from raw SRT with `acmt_product_v2.json`.

Old B1 on the same 20 products, using the stricter deterministic validator:

- claims read: 556
- claims inside product-v2 schema: 385
- deterministic direct product-fact claims: 193
- direct `(product, attribute)` pairs: 53
- average direct quote length: 14.8 chars

Product-v2 B1 prompt v1:

- claims read: 177
- claims inside product-v2 schema: 177
- deterministic direct product-fact claims: 103
- direct pairs: 40
- issue found: old `FREE::` few-shot examples caused LangExtract prompt-alignment warnings and some near-neighbor attribute errors, such as elastic/stretch claims assigned to thickness.

Product-v2 B1 prompt v2 plus direct-JSON fallback for one long SRT product:

- claims read: 248
- claims inside product-v2 schema: 248
- deterministic direct product-fact claims: 117
- direct pairs: 48
- B2/B3 raw skeleton has 72 claimful pairs before validation.

Direct JSON exact-grounding backend:

- non-strict direct prompt: 187 claims, 114 direct claims, 50 direct pairs, average direct quote length 12.3 chars.
- strict direct prompt: 289 claims, 175 direct claims, 65 direct pairs, average direct quote length 6.7 chars.
- strict direct gives higher validated pair coverage than old B1 while producing much shorter exact source quotes and no out-of-schema attributes.

Direct strict expansion pilots:

- First 100 products by schema order were all apparel.  Results:
  - claims read: 2,108
  - direct claims: 1,156
  - direct pairs: 312
  - raw B2/B3 claimful pairs before validation: 508
- Stratified 120 products, 12 per top-level category.  Results:
  - claims read: 1,502
  - direct claims: 701
  - direct pairs: 306
  - direct pairs per product: 2.55
  - category range: shoes/bags 4.83 pairs/product, jewelry 1.17, smart-home 1.00

Interpretation: pair-level expansion from product-v2 B1 is high-quality but will probably not reach 3,000 clean supervised pairs by itself.  However, direct strict extraction yields many more atomic claim instances.  A cleaner next formulation is to train on atomic `(claim, product evidence, consumer signal)` instances and aggregate to `(product, attribute)` for pair-level reporting.  This can preserve the retrieval-augmented contrastive learning mechanism while increasing sample size in a methodologically defensible way.

Code changes:

- `stage_b/b1_claim_extract.py` now uses product-v2 style few-shot examples, explicit "do not choose nearest broad attribute" rules, `source_family/value_type` in the attribute block, `--product_id`, `--max_char_buffer`, and cue-aware `--chunk_chars`.
- `stage_b/b1_claim_extract_direct.py` was added as a fallback that asks for exact source substrings and performs deterministic local substring grounding.  It successfully processed a long SRT item that repeatedly hung under the LangExtract OpenAI-compatible provider.
- `data_quality/validate_claim_attribute_v2.py` now requires non-generic evidence cues: `是/有` alone can no longer pass a claim as direct.

Interpretation: product-v2 direct JSON exact extraction is now the preferred B1 expansion path.  It removes schema leakage, avoids LangExtract long-SRT hangs, and improves validated pair coverage on the 20-product pilot.  The final data builder should keep B1 extraction, deterministic claim-attribute validation, and later B4 consumer-signal adjudication as separate atomic states.

## Next Data Steps

1. Rerun B1 with `acmt_product_v2.json` as the schema source.
2. Add claim-attribute validation immediately after B1.
3. Rerun Stage C by source family: numeric/identity via params/title/OCR, material via params/OCR/VLM, visual/boolean via OCR/VLM.
4. Rebuild B4 as atomic `consumer_signal` rather than a direct weak label.
5. Derive clean/silver/drop views only from fixed atomic states:
   - `claim_quality`
   - `attribute_quality`
   - `product_evidence_state`
   - `consumer_signal`

This path is slower than patching old labels but is the only route that matches the paper's methodological claim.

## Atomic Claim Full-Rerun Update

The product-v2 direct extraction path has now been promoted from pilot to full
raw-SRT rerun.

Stratified-120 atomic pipeline:

- atomic skeleton: 701 claims, 306 pairs, 96 products
- strict atomic B4 + verifier + deterministic precision filter:
  - high-precision dataset: 692 atomic claims, 304 pairs, 96 products
  - positives: 166 (24.0%)
  - records with refuting consumer signal: 166
  - records with supporting consumer signal only: 67
  - unaligned records: 459
- deterministic filters removed high-risk generic alignments:
  - numeric without comparable cue: 220 comments
  - color without color cue: 89 comments
  - generic comments: 3 comments
  - bad verifier reasons: 6 comments
  - objective-name low-overlap: 6 comments
  - noisy/question-like claim records dropped: 9

Learnability diagnostics on stratified-120:

- refined atomic dataset: AP 0.3771, AUROC 0.5742, Macro-F1 0.4337
- high-precision atomic dataset: AP 0.4186, AUROC 0.6223, Macro-F1 0.5582
- high-precision + RAFTS-style fact arguments: AP 0.4606, AUROC 0.6670, Macro-F1 0.6027

Interpretation: label-side precision filters and fact-only argument augmentation
both improve learnability, but the remaining gap shows that product evidence
coverage is still a major bottleneck.  Coverage-0 positive rows remain common;
they should be downweighted or used only as consumer-signal auxiliary rows
unless Stage C recovers evidence.

Full product-v2 B1 direct strict rerun:

- products processed: 544
- raw exact-grounded candidate claims: 8,647
- deterministic direct product-fact claims: 4,215
- claimful pairs: 1,473
- products with direct claims: 423
- atomic skeleton:
  - atomic claims: 4,215
  - pairs: 1,473
  - products: 423
  - source_family distribution: visual_or_boolean 1,521; numeric 1,304;
    material 543; direct_text_match 386; identity_or_spec 272;
    objective_name_only 189

The 4,215 full atomic claims satisfy the requested 3,000+ sample target before
B4/refine filtering.  Full atomic B4 alignment is currently the next critical
stage; after it finishes, run the same verifier, deterministic precision
filter, weak-label builder, fact-argument augmentation, and learnability
diagnostics used on stratified-120.

Recommended experimental use:

1. Keep the original pair-level benchmark as the main validation/test target
   until a full atomic benchmark is complete.
2. Use high-precision atomic rows as train-only auxiliary data with strict
   room/product/pair leakage guards.
3. Compare `--aux_train_weight_scale 0.25` and `0.35`.
4. Report atomic-to-pair aggregation only as an additional analysis unless the
   full atomic pipeline has a stable pair-level holdout.

## Full Atomic Data Repair Update: Stage C Evidence Rerun

Full strict atomic B4 has now been completed and repaired through the same
two-stage precision process used in the stratified pilot:

- full strict atomic B4 records: 4,215 atomic claims, 1,473 pairs, 423 products
- second-pass verifier tasks: 7,880 low-overlap alignments
- high-precision filtered output:
  - atomic claims: 4,197
  - pairs: 1,467
  - products: 423
  - positives: 1,031 (24.6%)
  - dropped noisy/question-like claims: 18
  - demoted comments: numeric-without-comparable-cue 1,056; color-without-color
    1,011; generic comments 5; objective-name low-overlap 77; bad verifier
    reasons 31

The first full high-precision dataset still exposed a Stage C bottleneck:

- original full HP coverage over final atomic rows:
  - coverage 0: 981
  - coverage 1: 1,862
  - coverage 2: 1,280
  - coverage 3: 74
- learnability diagnostic: AP 0.3635, AUROC 0.6273, Macro-F1 0.5659

Audit finding:

- final atomic join did not lose keys; every `(product_id, attribute_id)` had a
  fact record.
- the problem was evidence recall: Stage C evidence was produced under the older
  wide product schema and stale remote image paths (`/root/claimarc/...`), then
  reused for product-v2/atomic attributes.
- this was fixed by:
  - remapping stale absolute image paths to the current `CLAIMARC_ROOT`;
  - allowing product-v2/atomic-specific C1/C3/C4 outputs;
  - adding product title as a pseudo-parameter source in C2;
  - adding C2/C3/C4 `rerun-empty` and `rerun-missing-attrs` controls;
  - rerunning C2/C3/C4 on the atomic 1,473 attribute slots rather than the old
    30+ attribute/product schema.

Atomic Stage C rerun results:

| fact source setting | fact coverage 0 | coverage 1 | coverage 2 | coverage 3 | diagnostic AP | AUROC | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original full HP facts | 981 | 1,862 | 1,280 | 74 | 0.3635 | 0.6273 | 0.5659 |
| params/title + old OCR/VLM | 815 | 1,650 | 1,642 | 90 | 0.3638 | 0.6266 | 0.5673 |
| params/title + atomic OCR + old VLM | 955 | 1,819 | 1,348 | 75 | 0.3438 | 0.6334 | 0.5821 |
| params/title + atomic OCR + atomic VLM | 263 | 1,348 | 1,479 | 1,107 | 0.3618 | 0.6311 | 0.5699 |

Interpretation:

- atomic OCR improves macro decision separability, especially for negative
  product-evidence states.
- atomic VLM dramatically improves coverage but also introduces noisier
  visual descriptions; it should be used through sourceful filtering rather
  than blindly in the full dataset.
- the strongest current auxiliary view is:
  `dataset_atomic_productv2_direct_strict_full_strictv2_refined_hp_paramtitle_ocrblock_vlmatomic_sourceful_cov1.jsonl`
  - rows: 3,934
  - positives: 963
  - rooms: 102
  - diagnostic: AP 0.3763, AUROC 0.6653, Macro-F1 0.6098

Current pair-level experiment:

- original main benchmark remains:
  `dataset_attrpol_hq_product_rawtext_llmcurated_source_recovered_v3_dropunresolved.jsonl`
- train-only atomic auxiliary must keep fold-specific room/product/pair guards.
- first remote CV is running with the older sourceful auxiliary
  (`..._refined_hp_sourceful_cov1.jsonl`), weight 0.25, cap 1,500/fold,
  `bs=8, accum=4` after the uncapped run OOMed on A30.
- next remote CV should use the repaired VLM-sourceful auxiliary above, same
  guard/cap settings first; if it improves OOF metrics, then sweep weight 0.15,
  0.25, 0.35 and compare against `coverage>=2` as a lower-noise ablation.

## Product-v2 Claim Validation and Expansion Update: 2026-06-13

New audit finding:

- `claim_attribute_validation_v2.py` was still too permissive for clean atomic
  supervision.  It allowed link/order/SKU utterances such as `1号链接...`, bare
  numeric link references, generic `是/有/个/款` cues, questions, and attribute
  names without concrete values.
- The validator has been tightened so that:
  - hard promo/order terms (`链接`, `拍`, `库存`, `券`, `价格`, shipping/service
    terms) cannot enter direct claims;
  - question-like or “看一下” utterances are blocked;
  - material claims require material values such as cotton, wool, leather, or
    fiber terms;
  - numeric claims require actual numbers/units or direct thickness/layer cues;
  - visual generic cues such as `颜色/色/款式/款` no longer rescue link-shaped
    claims.

Validation outputs:

| validator | direct claims | direct pairs | note |
|---|---:|---:|---|
| old v2 | 4,215 | 1,473 | admitted many link/order utterances |
| v4 | 3,757 | 1,336 | high-recall silver pool |
| v6 | 3,234 | 1,093 | high-precision clean-anchor pool |
| v7 | 3,147 | 1,056 | stricter identity/capacity audit; too strict for cov1 auxiliary |

`v6` residual audit:

- hard promo direct claims: 0
- question-like direct claims: 0
- remaining link-number claims: 25, mostly SKU-specific thickness/layer
  statements; keep as silver or diagnostic rather than clean anchor.
- v7 adds two extra checks: identity/spec claims cannot be rescued by bare
  numbers, and battery/power-bank capacity claims must mention capacity,
  charge amount, or mAh-like evidence.  This is useful for audit, but it
  over-prunes current cov1 auxiliary rows.

New deterministic expansion queues:

- `productv2_comment_triggered_claim_reextract_queue_20260613_broad.jsonl`
  - 1,199 rows, 300 products
  - includes broad `explicit_fact_hit` comment triggers
- `productv2_comment_triggered_claim_reextract_queue_20260613_strict.jsonl`
  - 526 rows, 193 products
  - requires explicit text triggers such as `直播/宣传/虚标/不符/不是/说的`
  - priority: P0 268, P1 169, P2 89
- `productv2_candidate_softcap24_comment_triggered_claim_reextract_queue_20260613_strict.jsonl`
  - 567 rows, 197 products
  - candidate soft-cap schema adds only 41 strict rows, so expansion should
    prioritize claim/evidence recovery rather than further widening A_cmt.

LLM re-extraction pilot:

- script: `src/stage_b/b1_reextract_from_queue.py`
- input: top 5 strict P0 queue rows
- exact SRT substrings found for all 5 pairs, but post-validation showed only
  a subset are clean:
  - good: material value `婴儿棉的`
  - rejected: question-like `什么材质?看一下...`
  - rejected: capacity mismatch `支持三台设备同时快充`
  - rejected/silver: texture or density phrases that do not exactly match the
    target attribute

Full strict-P0 re-extraction:

- input queue rows: 268
- exact SRT claim candidates generated: 434
- v7 validation:
  - direct claims: 106
  - direct pairs: 72
  - direct products: 43
  - too vague: 284
  - wrong attribute: 6
  - promo/order: 38
- top recovered attributes include type, net content, flavor, size, thickness,
  brand, specification, function, and usage method.

Interpretation:

- comment-triggered re-extraction is promising for adding data from raw SRT,
  but it must be followed by the stricter v6 validator and product-evidence
  verification before promotion to the clean benchmark.
- The clean workflow is:
  1. strict queue -> exact SRT re-extraction;
  2. v6 claim-attribute validation;
  3. Stage C product evidence;
  4. atomic consumer-signal alignment;
  5. train-only auxiliary inclusion with room/product/pair guards.

High-purity auxiliary slices:

| slice | rows | pairs | products | positives | diagnostic AP | AUROC | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| repaired VLM sourceful cov1 | 3,934 | 1,329 | 416 | 963 | 0.3763 | 0.6653 | 0.6098 |
| repaired VLM sourceful cov2 | 2,586 | - | - | - | 0.3872 | 0.6393 | 0.5918 |
| v6-clean VLM sourceful cov1 | 2,941 | 970 | 359 | 772 | 0.3925 | 0.6457 | 0.6143 |
| v6-clean VLM sourceful cov2 | 1,888 | 671 | 306 | 516 | 0.3820 | 0.6175 | 0.5497 |
| v7-clean VLM sourceful cov1 | 2,861 | 937 | 356 | 769 | 0.3449 | 0.5850 | 0.5339 |
| v7-clean VLM sourceful cov2 | 1,819 | 643 | 302 | 513 | 0.4195 | 0.6341 | 0.5604 |

Recommended next experiment:

- run pair-level CV with `v6-clean VLM sourceful cov1` as train-only auxiliary,
  `aux_train_weight_scale=0.25`, `aux_train_max_per_fold=1500`, `bs=8`,
  `accum=4`;
- if positive, sweep `aux_train_weight_scale` 0.15/0.25/0.35 and compare with
  repaired VLM sourceful cov1 and cov2;
- report `available/blocked/added` auxiliary counts per fold and grouped
  diagnostics by coverage/source family.

Remote CV status:

- old non-VLM sourceful auxiliary (`...refined_hp_sourceful_cov1.jsonl`,
  3,216 rows) finished:
  - CLAIMARC_pcls: AP 0.8193, AUROC 0.9345, Macro-F1 0.8682, wF1 0.7563
  - BGE-LR: AP 0.8550, AUROC 0.9495, Macro-F1 0.8862, wF1 0.7542
  - interpretation: atomic auxiliary helps substantially over the previous
    CLAIMARC result, but still does not beat the strongest BGE-LR baseline.
- `v6-clean VLM sourceful cov1` CV has been launched with the same
  `weight_scale=0.25`, `cap=1500/fold`, `bs=8`, `accum=4` settings.

## 2026-06-13 Fact-aware P0 Re-extraction Update

Motivation:

- The first strict-P0 re-extraction still contained attribute-shift noise:
  `商品条形码` was triggered by price/link comments, `快充` was extracted as
  capacity, `300大卡` as flavor, and link/order fragments leaked into identity
  attributes.
- The fix is to make claim recovery evidence-aware: comments remain search
  hints, but product detail evidence from params/OCR/VLM anchors the target
  attribute dimension.  The model may extract claims that support or contradict
  the detail evidence, but the claim must be about the same attribute.

Code changes:

- `build_productv2_claim_reextract_queue.py`
  - excludes unspoken/promo attributes such as barcode, price, links, shipping,
    inventory, and identity/spec rows without product raw evidence.
  - strict v2 queue: 692 rows, 244 products; P0 348, P1 222, P2 122.
- `b1_reextract_from_queue.py`
  - adds `--fact_records` and `--min_fact_coverage`;
  - injects params/OCR/VLM snippets into the extraction prompt as attribute
    anchors.
- `validate_claim_attribute_v2.py`
  - v8-v11 add identity rescue blocking, price/sales slang (`39.9米`), brand
    generic-word rejection, concrete color-value requirements, screen-size vs
    film-thickness rejection, and type/flavor/net-content guards.
- `filter_atomic_records_v1.py`
  - hp2 keeps consumer-perception signals for color mismatch (`色差/不符`) and
    concrete function feedback, while still demoting generic exact-number
    mismatches.
- `build_atomic_training_dataset.py`
  - new deterministic joiner for atomic records + atomic labels + fact records.

Fact-aware P0 generation:

| stage | result |
|---|---:|
| strict-v2 P0 rows with product evidence | 306 |
| exact SRT candidates | 653 |
| v11 direct claims | 154 |
| direct pairs | 72 |
| direct products | 47 |
| B4 atomic records | 154 |
| hp2 labels | 111 positive / 43 negative |
| hp2 refined labels | 110 positive / 44 negative |

Key artifacts:

- `data/processed/stageB_product_v2/claim_reextract_productv2_strict_v2_factaware_p0_full_20260613_validated_v11.jsonl`
- `data/processed/stageB_product_v2/atomic_records_productv2_factaware_p0_v11_20260613_hp2.jsonl`
- `data/final/repaired_v1/dataset_atomic_productv2_factaware_p0_v11_20260613_hp2.jsonl`
- `data/final/repaired_v1/dataset_atomic_productv2_v6clean_plus_factaware_p0_v11_hp2_20260613.jsonl`

Quality diagnostics:

| auxiliary slice | rows | positives | diagnostic AP | AUROC | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| fact-aware P0 v11 hp2 only | 154 | 111 | 0.7953 | 0.5774 | 0.4432 |
| fact-aware P0 v11 hp2 refined only | 154 | 110 | 0.7356 | 0.5430 | 0.4694 |
| v6-clean cov1 + fact-aware P0 hp2 | 3,095 | 883 | 0.4406 | 0.6765 | 0.6291 |
| v6-clean cov1 + fact-aware P0 hp2 refined | 3,095 | 882 | 0.4155 | 0.6812 | 0.5974 |
| fact-aware P1 v11 hp2 only | 114 | 62 | 0.5387 | 0.4704 | 0.5006 |
| v6-clean cov1 + fact-aware P0+P1 hp2 | 3,209 | 945 | 0.4422 | 0.6694 | 0.5552 |

Interpretation:

- The standalone P0 patch is intentionally small and positive-heavy; it should
  not be used as a standalone benchmark.
- As a train-only auxiliary patch, hp2 improves the lightweight diagnostic over
  v6-clean cov1 alone (Macro-F1 0.6291 vs 0.6143; AP 0.4406 vs 0.3925).
- The refined variant is too strict for the consumer-perception objective and
  will be kept as an audit slice, not the next main auxiliary.
- P1 adds balanced labels but much noisier attribute semantics; adding it to
  P0 hurts Macro-F1 in the diagnostic.  Keep P1 as a candidate pool for future
  hand/LLM verification, not as part of the next main auxiliary.

Next experiment:

- after the running v6-clean cov1 CV completes, run the same CV settings with
  `dataset_atomic_productv2_v6clean_plus_factaware_p0_v11_hp2_20260613.jsonl`
  as `--aux_train_dataset`;
- keep `weight_scale=0.25`, `cap=1500/fold` for direct comparability, then
  sweep 0.15/0.35 only if the combined hp2 auxiliary improves the 5-fold
  CLAIMARC/RKC result.

## 2026-06-13 Hard-clean and Expectation-gap Audit

Motivation:

- A read-only subagent audit and manual samples found that the v6-clean
  auxiliary still contains a small but harmful residue of transaction/promotion
  claims: price fragments, orders, links, installation fees, logistics/service
  wording, and title-only evidence counted as full product facts.
- The hp2 filter also demoted some valid consumer-perception gaps, such as
  "说是大包其实小包", "没直播间看的厚实", and "没有宣传的不沾手汗".  These
  are not strict numeric refutations, but they are valid expectation-gap
  signals for the paper's perceived misleading-advertising target.

Code changes:

- `filter_atomic_records_v1.py`
  - hp3 adds explicit expectation-gap cues (`宣传/直播间/视频/描述/实物不符`)
    and comparative size/amount/thickness/color cues;
  - keeps demoting generic sentiment and support comments without concrete
    color/value overlap.
- `clean_atomic_aux_dataset_v1.py`
  - creates reproducible auxiliary views without modifying original JSONL;
  - removes transaction-only claims, strips price terms when a product claim
    remains, excludes promo/service attributes, moves `商品标题` to
    `evidence_title_hints`, ignores weak OCR labels, recomputes effective
    evidence coverage, and downweights rows whose coverage drops.

hp3 fact-aware slices:

| slice | rows | positives | notes |
|---|---:|---:|---|
| P0 v11 hp3 | 154 | 116 | restores expectation-gap positives; P0 remains positive-heavy |
| P1 v11 hp3 | 114 | 71 | recovers more size/color gaps, but P1 still contains weaker semantics |

Hard-clean candidate views:

| auxiliary view | rows | positives | diagnostic AP | AUROC | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| v6-clean cov1 original | 2,941 | 772 | 0.3925 | 0.6457 | 0.6143 |
| v6-clean hardclean v1 | 2,888 | 757 | 0.3740 | 0.6689 | 0.6318 |
| hardclean + P0 hp2 | 3,042 | 868 | 0.4029 | 0.6051 | 0.5878 |
| hardclean + P0 hp2 refined | 3,042 | 867 | 0.4141 | 0.6123 | 0.5492 |
| hardclean + P0 hp3 | 3,042 | 873 | 0.3833 | 0.6225 | 0.5882 |

Interpretation:

- Hard-cleaning is beneficial as a noise-control step: it raises lightweight
  AUROC and Macro-F1 by removing transaction and title-only shortcuts while
  changing only a small number of rows.  AP falls after the final installation
  fee scrub, suggesting fewer lexical shortcuts but a cleaner decision boundary.
- P0 hp3 is conceptually closer to consumer-perception risk, but it is still a
  positive-heavy repair slice.  Its lower lightweight AUROC suggests it should
  be evaluated as a train-only auxiliary patch, not accepted solely by the
  diagnostic.
- The next most defensible remote CV candidate is `v6-clean hardclean v1`.
  P0 variants should be treated as consumer-perception sensitivity runs rather
  than the immediate mainline, because the repair slice is positive-heavy and
  weakens the lightweight diagnostic when merged.
  P1 remains an audit/verification pool rather than a main auxiliary.

## 2026-06-13 Upstream Candidate Rebuild: Claim Extraction and Schema Scope

Motivation:

- A second read-only audit located the largest remaining upstream risk in
  Stage B rather than final joining: `claim_attribute_validation_v2` was still
  able to rescue bare link/order numbers as numeric facts, while the product
  schema could keep transaction attributes such as `赠品信息` when their aliases
  contained objective words like `数量` or `颜色`.
- The fix is kept as a candidate rebuild line so that current GPU experiments
  remain comparable.

Code changes:

- `validate_claim_attribute_v2.py`
  - separates bare numbers from real measurements/ranges;
  - prevents `N号链接` from validating numeric or identity/spec attributes;
  - restores valid product values such as `1.2米`, `35到40码`,
    `羊毛混纺`, `双层`, and `3号链接是黑色`;
  - removes the old false price pattern that treated decimal meter values as
    prices.
- `b1_claim_extract_direct.py`
  - strengthens the LLM prompt with negative rules for link/order/price claims;
  - adds a deterministic post-generation veto before writing exact-grounded
    claims, including attribute-family checks for color, material, and numeric
    claims.
- `build_acmt_product_v2.py`
  - raises the candidate cap to 24 attributes per product for recall;
  - prioritizes hard process/transaction terms (`链接`, `下单`, `赠品`,
    `发货`, `售后`, etc.) before objective-word rescue, removing leaked
    attributes such as `赠品信息`;
  - records over-cap audit counts for reproducibility.

Candidate artifacts:

| artifact | value |
|---|---:|
| candidate A_cmt pairs | 6,914 |
| products | 544 |
| p50 / p90 / max attrs | 12 / 24 / 37 |
| old-claim validation direct claims | 2,661 |
| old-claim validation pairs with direct claim | 912 |
| old-claim validation promo/order rejections | 433 |

30-product B1 probe:

| source | raw claims | promo text | price text | validation direct | validation promo/order |
|---|---:|---:|---:|---:|---:|
| old direct-strict claims on same products | 710 | 3 | 4 | not rerun | not rerun |
| candidate B1 prompt+veto | 589 | 0 | 0 | 237 | 2 |

Interpretation:

- The candidate extractor is stricter but cleaner: it removes price/promo text
  at the raw claim level and preserves enough direct facts for downstream
  reconstruction.
- Full candidate B1 regeneration has been launched to
  `data/processed/stageB_product_v2_candidate_20260613/claim_list_direct_full_v2_20260613`.
  After completion, run the same deterministic validation and rebuild the
  atomic evidence/label path as a separate candidate dataset.

Full B1 candidate regeneration result:

| stage | value |
|---|---:|
| products regenerated | 544 |
| raw exact-grounded claims | 6,169 |
| raw price-text claims | 0 |
| validation direct claims | 2,434 |
| validation pairs with direct claim | 861 |
| validation products with direct claim | 334 |
| validation promo/order rejections | 59 |
| candidate atomic skeleton claims | 2,434 |

Atomic skeleton source-family mix:

| family | claims |
|---|---:|
| visual_or_boolean | 1,446 |
| numeric | 440 |
| direct_text_match | 255 |
| material | 177 |
| objective_name_only | 102 |
| identity_or_spec | 14 |

Next step:

- run B4 atomic consumer-comment alignment on
  `atomic_claim_skeleton_full_v2_20260613.jsonl`;
- filter exact-number/color over-alignments with hp3 filters;
- build atomic labels and join with existing Stage-C fact evidence as a new
  candidate auxiliary dataset.

## 2026-06-13 Remote CV Result: v6-clean cov1 Auxiliary

Setting:

- main data:
  `dataset_attrpol_hq_product_rawtext_llmcurated_source_recovered_v3_dropunresolved.jsonl`
- auxiliary data:
  `dataset_atomic_productv2_direct_strict_full_strictv2_refined_hp_paramtitle_ocrblock_vlmatomic_sourceful_cov1_v6clean.jsonl`
- `aux_train_weight_scale=0.25`, `aux_train_max_per_fold=1500`,
  `source0_ce_scale=0.5`, `source0_cl_scale=0.25`,
  `threshold_policy=prior_stable`.

Pooled 5-fold result:

| model | AP | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| CLAIMARC_pcls | 0.8285 | 0.9394 | 0.8785 | 0.7564 |
| CLAIMARC_selectiveRKC | 0.8287 | 0.9396 | 0.8797 | 0.7587 |
| CLAIMARC_v2 | 0.8196 | 0.9253 | 0.8682 | 0.7488 |
| bge_lr | 0.8550 | 0.9498 | 0.8886 | 0.7747 |

Interpretation:

- The repaired atomic auxiliary substantially improves over the earlier
  CLAIMARC mainline, but it still does not beat the strong BGE-LR baseline on
  AP/AUROC.
- This is a useful but not publishable stopping point.  The next active GPU
  experiment is `v6-clean hardclean v1`, which removes transaction/title-only
  shortcuts from the auxiliary view.
- In parallel, the upstream B1 candidate rebuild is continuing because the
  remaining AP gap is likely caused by data noise and claim/evidence alignment
  more than by model capacity alone.

## 2026-06-13 Remote CV: hardclean v1 First Fold

The `v6-clean hardclean v1` run has started automatically after the v6-clean
run.  Fold 0 is complete:

| run | head | AP | AUROC | Macro-F1 | wF1 |
|---|---|---:|---:|---:|---:|
| v6-clean cov1 | PCLS | 0.8773 | 0.9444 | 0.9029 | 0.8028 |
| v6-clean cov1 | selectiveRKC | 0.8919 | 0.9511 | 0.9083 | 0.8288 |
| hardclean v1 | PCLS | 0.8868 | 0.9523 | 0.9124 | 0.8370 |
| hardclean v1 | selectiveRKC | 0.8895 | 0.9532 | 0.9148 | 0.8377 |

Interpretation:

- The hard-clean auxiliary improves fold-0 PCLS AP/AUROC/Macro-F1 and improves
  RKC Macro-F1/wF1, while RKC AP is essentially tied with the v6-clean fold.
- This supports the hypothesis that removing transaction/title-only shortcuts
  helps calibration and class-boundary quality.  Wait for all five folds before
  deciding on source-aware hard-negative/proto auxiliary sweeps.

## 2026-06-13 Candidate Full-Rerun Labels and Merge Gates

The upstream candidate line has now completed the full atomic path:

| stage | value |
|---|---:|
| candidate atomic records before hp3 filter | 2,434 |
| hp3-filtered atomic records | 2,432 |
| hp3 record positives | 706 |
| joined auxiliary rows | 2,232 |
| joined positives | 653 |
| joined pairs / products | 730 / 312 |
| coverage 1 / 2 / 3 | 856 / 858 / 518 |

Direct replacement diagnostics:

| candidate view | rows | positives | diagnostic AP | AUROC | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| full hp3 candidate | 2,232 | 653 | 0.3849 | 0.6104 | 0.5711 |
| coverage >= 2 only | 1,376 | 430 | 0.3162 | 0.4030 | 0.4699 |
| above-floor positives + cov2 negatives | 1,493 | 474 | 0.4333 | 0.6060 | 0.5493 |
| above-floor only | 665 | 474 | 0.7360 | 0.4974 | 0.4787 |
| non-objective-name only | 2,169 | 644 | 0.4016 | 0.6134 | 0.5240 |

Interpretation:

- The full-rerun candidate has useful recall and a modest AP gain over the
  hard-clean auxiliary diagnostic, but AUROC/Macro-F1 are weaker.  This means
  the candidate pool should not replace hard-clean as the main auxiliary view.
- `above-floor only` is positive-heavy and has an inflated AP; it is not a
  balanced training distribution.
- Coverage-only filtering is too blunt: it removes many consumer-signal
  positives and leaves a weak negative boundary.

Reproducible merge script:

- `src/data_quality/build_atomic_aux_merge_v2.py`
  - anchors on `v6clean_hardclean_v1`;
  - skips exact `(product, attribute, claim text)` duplicates;
  - optionally keeps only unseen product-attribute pairs;
  - gates positives by refuting alignment and minimum confidence;
  - gates negatives by product-evidence coverage and optional supporting
    alignment;
  - caps candidate rows per product-attribute pair.

Merged auxiliary diagnostics:

| merged auxiliary view | rows | candidate rows added | positives | pairs | diagnostic AP | AUROC | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hardclean only | 2,888 | 0 | 757 | 941 | 0.3740 | 0.6689 | 0.6318 |
| hardclean + candidate hpmerge | 3,274 | 386 | 878 | 1,033 | 0.4409 | 0.7131 | 0.5985 |
| hardclean + candidate new-pair hpmerge | 3,002 | 114 | 788 | 1,033 | 0.3894 | 0.6617 | 0.6064 |
| hardclean + candidate support-neg hpmerge | 3,058 | 170 | 878 | 979 | 0.3996 | 0.6679 | 0.6025 |
| hardclean + candidate pos0.15 hpmerge | 3,260 | 372 | 861 | 1,032 | 0.4158 | 0.6329 | 0.5674 |

Interpretation:

- `hpmerge` gives the strongest AP/AUROC and adds 386 non-duplicate rows, but
  its Macro-F1 drop suggests threshold instability.  It is a good low-weight
  train-only auxiliary candidate, not a clean benchmark replacement.
- `new-pair hpmerge` adds only 114 rows but expands coverage to the same 1,033
  pairs while preserving a stronger Macro-F1.  This is the safest candidate
  for a conservative coverage-expansion CV.
- `support-neg hpmerge` is useful for a consumer-support ablation, but it
  becomes positive-heavy and should not be the immediate mainline.

Next GPU candidates after hardclean CV completes:

1. If hardclean closes most of the gap to BGE-LR, run the conservative
   `new-pair hpmerge` auxiliary at the same `weight_scale=0.25`, cap 1,500
   setting to test whether coverage expansion improves without label drift.
2. If hardclean still loses mainly on AP/AUROC, first keep the same data and
   open source-aware RACL hard negatives:
   `--cl_neg_filter same_evtype_conf` or a small
   `--cl_neg_bonus 0.03 --cl_neg_bonus_filter same_evtype_conf`.
3. Add `--proto_aux_weight` only as a later ablation, because it changes the
   representation objective more than the hard-negative policy.

## 2026-06-13 Remote CV Result: hardclean v1 Auxiliary

Setting:

- main data:
  `dataset_attrpol_hq_product_rawtext_llmcurated_source_recovered_v3_dropunresolved.jsonl`
- auxiliary data:
  `dataset_atomic_productv2_v6clean_hardclean_v1_20260613.jsonl`
- `aux_train_weight_scale=0.25`, `aux_train_max_per_fold=1500`,
  `source0_ce_scale=0.5`, `source0_cl_scale=0.25`,
  `threshold_policy=prior_stable`.

Pooled 5-fold result:

| model | AP | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| CLAIMARC_pcls | 0.8389 | 0.9440 | 0.8840 | 0.7764 |
| CLAIMARC_selectiveRKC | 0.8389 | 0.9440 | 0.8840 | 0.7764 |
| CLAIMARC_v2 | 0.8301 | 0.9342 | 0.8737 | 0.7734 |
| bge_lr | 0.8547 | 0.9492 | 0.8882 | 0.7722 |

Compared with `v6-clean cov1`, hardclean improves CLAIMARC:

| run | CLAIMARC AP | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| v6-clean cov1 | 0.8285 | 0.9394 | 0.8785 | 0.7564 |
| hardclean v1 | 0.8389 | 0.9440 | 0.8840 | 0.7764 |

Interpretation:

- Hardclean is the best data-only auxiliary so far.  It improves CLAIMARC over
  v6-clean and exceeds BGE-LR on confidence-weighted F1.
- It still does not unambiguously beat BGE-LR on AP/AUROC/Macro-F1.  The
  strongest remaining weakness is ranking/calibration, not basic classification
  boundary quality.
- Fold diagnostics show RKC sometimes improves Macro-F1 but often hurts AP,
  suggesting that retrieval neighbors are still too coarse when evidence source
  and confidence differ.

Follow-up model ablations:

| ablation | change | AP | AUROC | Macro-F1 | wF1 | decision |
|---|---|---:|---:|---:|---:|---|
| hardclean baseline | none | 0.8389 | 0.9440 | 0.8840 | 0.7764 | keep as current anchor |
| hard negative filter | `cl_neg_filter=same_evtype_conf` | fold0 only: 0.8493 | 0.9450 | 0.8552 | 0.7414 | aborted after fold0; too much recall loss |
| soft hard-negative bonus | `cl_neg_bonus=0.03`, `same_evtype_conf` | 0.8310 | 0.9426 | 0.8781 | 0.7736 | reject; softer but still below anchor |

Interpretation:

- Source/confidence-aware hard negatives are directionally plausible for
  validation AP in some folds, but filtering the negative pool changes the
  optimization geometry too aggressively and hurts test Macro-F1/wF1.
- Soft bonus preserves recall but still does not improve pooled AP/AUROC over
  hardclean.  The next model-side line should avoid changing negative sampling
  and instead regularize the retrieval space more gently.
- Active next run: source/confidence-aware prototype auxiliary,
  `proto_aux_weight=0.03`, `proto_aux_group=source_conf`,
  `proto_aux_mode=margin`, on the same hardclean auxiliary setting.

## 2026-06-13 Value-Gated Claim Validation Audit

Change:

- Added a narrow attribute-name-aware value gate in
  `src/data_quality/validate_claim_attribute_v2.py`.
- Purpose: recover short but concrete ASR claims such as `尖头` for `鞋头款式`,
  `香辣味` for `香味`, `中长款` for `衣长/款式`, and OCR/ASR material variants
  such as `三醋`/`桑残丝`.
- The gate deliberately does not rescue identity/spec attributes such as
  `品牌`, `型号`, or `产品名称`.

Validation funnel:

| validation view | direct claims | pairs with direct claim | atomic claims | joined rows | positives |
|---|---:|---:|---:|---:|---:|
| v2 candidate full | 2,434 | 861 | 2,434 | 2,232 | 653 |
| v3 valuegate full | 2,723 | 995 | 2,723 | 2,475 | 699 |

`v3` recovers 289 direct claims and 134 additional product-attribute pairs.
However, the recovered rows are dominated by `适用季节`, `适用对象/人群`, generic
`功能/功效`, and style short values.  Many are legitimate claim snippets, but
their consumer-comment labels are weak and mostly negative.

Learnability diagnostics:

| dataset view | rows | positives | AP | AUROC | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| v2 candidate hp3 | 2,232 | 653 | 0.3849 | 0.6104 | 0.5711 |
| v3 valuegate hp3 | 2,475 | 699 | 0.3360 | 0.5691 | 0.5217 |
| v3 + style/material only | 2,294 | 657 | 0.3317 | 0.4920 | 0.5583 |
| v3 + style/material cov2 | 2,278 | 656 | 0.3255 | 0.5211 | 0.5311 |
| v3 + direct-text positive cov2 | 2,272 | 693 | 0.3896 | 0.5732 | 0.5289 |
| v3 + non-objective cov2 | 2,375 | 696 | 0.3488 | 0.5134 | 0.5164 |
| v3 + no audience/season/function | 2,350 | 684 | 0.3398 | 0.5587 | 0.5229 |
| v3 + positive-only recovered rows | 2,280 | 701 | 0.3748 | 0.6059 | 0.5247 |

Decision:

- Keep the value gate as a documented diagnostic and optional recovery tool.
- Do not use `v3` valuegate rows as the main auxiliary training pool yet.
  Recall improved, but label specificity did not.
- The data bottleneck is now less about finding any `(product, attribute)`
  evidence and more about aligning recovered short claims to consumer comments
  with enough specificity.  Future expansion should use LLM adjudication only
  for high-priority recovered rows, especially positive/refuting candidates,
  rather than admitting all recovered short-value claims.

Additional valuegate P0 LLM review:

- Review queue: `data/final/repaired_v1/valuegate_recovered_review_queue_20260613.jsonl`
  contains 243 v3-only rows: P0=43, P1=125, P2=75.
- Qwen-Flash review on the 43 P0 rows: 10 `keep`, 33 `drop`;
  `risk_label_valid=1` for 32 rows, but no row satisfies both
  `recommended_action=keep` and `risk_label_valid=1`.
- This means the recovered P0 rows often contain either a concrete claim whose
  current risk label is not valid, or a risk-like comment alignment attached to
  an insufficiently concrete/unsupported claim.  They are valuable for manual
  audit and prompt repair, but not for automatic positive expansion.

## 2026-06-13 Teacher-Consistent Expansion and Distillation Queue

Prototype/source-confidence auxiliary result:

- The source/confidence prototype auxiliary run was stopped after early folds.
- Fold 0 underperformed the hardclean anchor on Macro-F1/wF1:
  - prototype PCLS: AP 0.8746, AUROC 0.9524, Macro-F1 0.8982, wF1 0.8009
  - prototype selectiveRKC: AP 0.9066, AUROC 0.9555, Macro-F1 0.8982, wF1 0.8009
  - hardclean fold-0 anchor: PCLS AP 0.8868, AUROC 0.9523, Macro-F1 0.9124,
    wF1 0.8370; selectiveRKC AP 0.8895, AUROC 0.9532, Macro-F1 0.9148,
    wF1 0.8377
- Decision: reject source/confidence prototype loss as the next main line.  It
  can raise AP in a fold, but it suppresses the consumer-risk recall geometry
  needed for Macro-F1/wF1.

Teacher-consistent data selector:

- Added `src/data_quality/build_teacher_consistent_aux_v1.py`.
- The selector keeps `v6-clean hardclean v1` intact, then admits candidate
  rows only if an out-of-fold char-TFIDF teacher agrees with the label and the
  row passes source-coverage gates.  New rows are capped per
  `(attribute_id, label)` block to avoid color/size dominance.
- Main run on `hardclean + candidate hpmerge`:
  - base rows: 2,888
  - candidate-new rows: 343
  - accepted after teacher/source/cap: 163
  - final rows: 3,051
  - added labels: 152 negative, 11 positive
  - added confidence: high 38, medium 118, low 7

Learnability diagnostics:

| auxiliary view | rows | added policy | AP | AUROC | Macro-F1 | decision |
|---|---:|---|---:|---:|---:|---|
| hardclean v1 | 2,888 | trusted anchor | 0.3740 | 0.6689 | 0.6318 | keep |
| teacher-consistent v1 | 3,051 | strict teacher agreement | 0.4077 | 0.7076 | 0.6046 | AP/AUROC up, positives over-pruned |
| tc pos45/neg44 | n/a | looser positive teacher | 0.3773 | 0.6635 | 0.6163 | not enough AP gain |
| tc pos50/neg44 | n/a | medium positive teacher | 0.4047 | 0.6718 | 0.5739 | reject |
| tc pos00c12/neg44 | n/a | c-gated positives | 0.3668 | 0.6353 | 0.5653 | reject |
| tc pos45/neg40 | n/a | stricter negatives | 0.4081 | 0.6775 | 0.5973 | AP up, Macro too low |
| full hpmerge | 3,274 | recall-oriented candidate merge | 0.4409 | 0.7131 | 0.5985 | useful recall pool, not clean anchor |

Interpretation:

- Expansion improves lexical separability (AP/AUROC) but easily weakens the
  consumer-perception boundary reflected in Macro-F1.
- Teacher agreement is useful for rejecting noisy negatives, but if applied
  symmetrically it deletes positive boundary cases; this matches the theoretical
  claim that perceived-misleading positives are often less lexically separable.
- Next data line should not be "more rows" alone.  Use a two-pool strategy:
  trusted hardclean rows for the main auxiliary objective, and recall-oriented
  hpmerge rows only for ablation/teacher or low-weight auxiliary experiments.

Active GPU run:

- `cv_attrpol_aux_atomic_hardclean_distill005_disagree_c025_w025_cap1500_bs8_s0_20260613`
- Change: keep hardclean auxiliary data and RACL unchanged; add fold-local
  BGE+LR teacher distillation only on high-confidence teacher/student
  disagreements (`weight=0.05`, `temp=1.5`, `conf_min=0.25`).
- Purpose: recover AP/AUROC against BGE-LR without letting the teacher overwrite
  the RACL boundary representation.

Follow-up status:

- The distillation run was stopped in fold 0.  The run repeatedly emitted
  tokenizer fork warnings after the fold-local BGE teacher was built and then
  stalled after epoch 4.  Its validation curve was also unstable:
  warmup ep0 was strong (val Macro-F1 0.8936, AP 0.8950), but CL epochs dropped
  to val AP 0.7639/0.7782 at ep3/ep4.
- Decision: do not use online fold-local BGE distillation as the next main
  model line.  If teacher guidance is used later, precompute teacher scores as
  an offline data-QA artifact rather than building the teacher inside each
  training fold.
- New active run:
  `cv_attrpol_aux_atomic_hpmerge_loww015_cap1500_bs8_s0_20260613`.
  This uses the full hpmerge recall pool as low-weight train-only auxiliary
  data (`weight_scale=0.15`) while keeping the CLAIMARC/RACL objective unchanged.

Low-weight hpmerge status:

- Stopped after fold 0.  It underperformed the hardclean fold-0 anchor on all
  primary metrics:
  - hpmerge low-weight: AP 0.8580, AUROC 0.9458, Macro-F1 0.8995, wF1 0.8062
  - hardclean fold-0 anchor: AP 0.8868, AUROC 0.9523, Macro-F1 0.9124, wF1 0.8370
- Decision: full hpmerge remains a recall/QA pool, not a train-only auxiliary
  improvement.  The extra rows improve simple learnability diagnostics but do
  not transfer to grouped-CV CLAIMARC test performance.
- New active run:
  `cv_attrpol_aux_atomic_hardclean_viewcons002_w025_cap1500_bs8_s0_20260613`.
  This keeps hardclean data fixed and adds light evidence-view consistency
  (`view_logit_weight=0.02`, `view_embed_weight=0.02`) across
  `source_first,no_args,params_args,ocr_args,vlm_args`.

View-consistency status:

- First attempt with `bs=8, accum=4` ran out of GPU memory after warmup
  (A30 24GB; double evidence-view forward pushed allocation above 22GB).
- Retry should use `bs=4, accum=8` and the same effective batch size.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not supported by the
  remote PyTorch build, so the retry removes that allocator option.
- Active retry:
  `cv_attrpol_aux_atomic_hardclean_viewcons002_w025_cap1500_bs4_retry_s0_20260613`.

View-consistency retry status:

- Stopped during fold 0.  `bs=4` avoided immediate OOM, but the double-view
  forward made the CL phase too slow, and warmup validation did not improve over
  hardclean (ep1 val Macro-F1 0.8897, AP 0.8540 vs hardclean ep1 Macro-F1
  0.9035, AP 0.8465).
- Decision: keep evidence-view consistency as a future implementation task only
  if we redesign it offline or apply it to a smaller source-specific subset.
- New active run:
  `cv_attrpol_aux_atomic_hardclean_sourceaux002_w025_cap1500_bs8_s0_20260613`.
  It keeps hardclean data and RACL unchanged, adding only lightweight source
  auxiliary heads on the retrieval embedding (`combo=0.02`, `confidence=0.02`,
  `count=0.01`).

Source-auxiliary status:

- Stopped after fold 0.  It did not beat the hardclean anchor:
  - source-aux PCLS: AP 0.8701, AUROC 0.9454, Macro-F1 0.8888, wF1 0.7702
  - source-aux selectiveRKC: AP 0.8768, AUROC 0.9470, Macro-F1 0.8989, wF1 0.7984
  - hardclean fold-0 anchor: PCLS AP 0.8868, AUROC 0.9523, Macro-F1 0.9124,
    wF1 0.8370; selectiveRKC AP 0.8895, AUROC 0.9532, Macro-F1 0.9148,
    wF1 0.8377
- Decision: source metadata prediction is useful as a diagnostic/mechanism
  target, but not as a training regularizer in the current architecture.
- New active run:
  `cv_attrpol_aux_atomic_hardclean_lcl025_w025_cap1500_bs8_s0_20260613`.
  This keeps the data, retrieval mechanism, and architecture fixed, changing
  only the contrastive weight from `lambda_cl=0.5` to `0.25`.

Contrastive-weight sweep status:

- `lambda_cl=0.25` stopped after fold 0:
  - PCLS: AP 0.8707, AUROC 0.9496, Macro-F1 0.8977, wF1 0.8226
  - selectiveRKC: AP 0.8817, AUROC 0.9531, Macro-F1 0.8937, wF1 0.8286
  - both are below the hardclean fold-0 anchor.
- Decision: weaker RACL does not improve the boundary.  The original
  `lambda_cl=0.5` remains the anchor unless stronger RACL helps.
- New active run:
  `cv_attrpol_aux_atomic_hardclean_lcl075_w025_cap1500_bs8_s0_20260613`,
  changing only `lambda_cl` from `0.5` to `0.75`.

Stronger RACL status:

- `lambda_cl=0.75` stopped after fold 0:
  - PCLS: AP 0.8689, AUROC 0.9527, Macro-F1 0.9024, wF1 0.8097
  - selectiveRKC: AP 0.8743, AUROC 0.9543, Macro-F1 0.8897, wF1 0.8148
  - hardclean fold-0 anchor remains higher on AP/Macro-F1/wF1:
    PCLS AP 0.8868, Macro-F1 0.9124, wF1 0.8370; selectiveRKC AP 0.8895,
    Macro-F1 0.9148, wF1 0.8377
- Decision: the contrastive-weight sweep is closed for now.  The problem is no
  longer RACL strength; the next bottleneck is exact claim/evidence/value
  alignment in parameter-heavy rows.

## 2026-06-13 OOF Mechanism Diagnosis

Files generated:

- `data/final/cleancl/claimarc_hardclean_mechanisms_vs_bge_20260613.json`
- `data/final/cleancl/claimarc_vs_bge_hardclean_grouped_bootstrap_20260613.json`
- script: `src/models/diagnose_oof_mechanisms.py`

Overall hardclean OOF comparison:

| method | AP | AUROC | Macro-F1 | wF1 | ECE10 |
|---|---:|---:|---:|---:|---:|
| CLAIMARC_pcls | 0.8389 | 0.9440 | 0.8840 | 0.7764 | 0.1440 |
| BGE-LR | 0.8547 | 0.9492 | 0.8882 | 0.7722 | 0.1742 |

Paired/bootstrap interpretation:

- BGE-LR remains stronger on AP/AUROC/Macro-F1 for the current hardclean OOF.
- CLAIMARC has better calibration and slightly higher confidence-weighted F1,
  but the `dWF1` confidence interval still crosses zero under room-grouped
  bootstrap.
- This means the current result is not yet publishable as "significantly better
  than all baselines"; the next improvement should be data-quality targeted,
  not another broad regularizer.

Mechanism findings:

- CLAIMARC corrects 67 examples that BGE-LR misses; BGE-LR corrects 74 examples
  that CLAIMARC misses.
- CLAIMARC improves over BGE-LR when exactly two evidence sources are present
  (`dAP=+0.0475`, `dAUROC=+0.0154`, `d_wF1=+0.0255`) and on lower-confidence
  quantile slices (`q1 dAP=+0.0482`, `q3 dMacro-F1=+0.0199`).
- CLAIMARC is weaker on digital/electronics, jewelry, shoes/bags, and some
  OCR-only or exact-parameter rows.  The representative false positives show
  a repeated pattern: the model fires on attribute mentions even when the
  product evidence is compatible with the claim, title-derived, or too generic.

Data implication:

- The next reconstruction pass should prioritize exact claim/evidence alignment:
  numeric values, material terms, size/weight, price/coupon math, and visual
  property assertions.
- LLM calls should be applied to a small repair queue rather than the whole
  dataset: verify the live claim span, recover the exact product evidence span
  from raw params/OCR/VLM/detail text, and adjudicate whether consumer evidence
  indicates a perceived mismatch.
- Candidate/hpmerge rows remain useful as a recall pool, but they should only
  enter training after passing value alignment, source sufficiency, and
  positive-boundary preservation checks.

Targeted repair queue:

- Built with `src/data_quality/build_mechanism_repair_queue_v1.py`.
- LLM/VLM review entrypoint prepared:
  `src/data_quality/llm_review_mechanism_repair_queue_v1.py`.
- Review-application entrypoint prepared:
  `src/data_quality/apply_mechanism_repair_reviews_v1.py`.
- Review-audit gate prepared:
  `src/data_quality/audit_mechanism_repair_reviews_v1.py`.
- Output:
  `data/final/repaired_v1/mechanism_repair_queue_v1_20260613.jsonl`
  and
  `data/final/repaired_v1/mechanism_repair_queue_v1_20260613.stats.json`.
- n=400 prioritized rows:
  - exact value/material hints: 345
  - high-confidence CLAIMARC false positives: 114
  - BGE-correct / CLAIMARC-wrong rows: 65
  - high-risk categories: shoes/bags 119, digital/electronics 80, jewelry 42,
    sports/outdoor 39
  - evidence combos: P 141, O 132, PO 109
- Each queue row now includes raw detail-image paths under
  `data/raw/product_images/<product_id>/`, so the next LLM/VLM pass can recover
  evidence from the original product detail images instead of only the current
  compressed OCR/parameter summaries.
- Directly taking the first 80 rows is biased: top-80 rows are all current
  negatives and all high-confidence CLAIMARC false positives.  A balanced pilot
  is therefore built before API review.
- Balanced pilot builder prepared:
  `src/data_quality/build_mechanism_repair_pilot_v1.py`.
- Pilot output:
  `data/final/repaired_v1/mechanism_repair_pilot80_v1_20260613.jsonl`
  and
  `data/final/repaired_v1/mechanism_repair_pilot80_v1_20260613.report.json`.
  n=80, covering all 10 categories, evidence combos O/P/PO plus sparse mixed
  cases, and 14 current positives.
- The review prompt deliberately withholds current labels and model predictions.
  It asks only for claim quality, product-evidence relation, exact value
  alignment, likely extraction issue, and repair action.  This keeps LLM usage
  as an auditable data-quality instrument rather than a label oracle.
- The application script converts review JSONL into candidate datasets with
  three modes:
  - `audit`: attach review/decision metadata only.
  - `conservative`: drop bad claim spans, repair only low-confidence label
    contradictions/supports, and downweight weak uncertain rows.
  - `candidate`: more aggressive relabeling for fold-0 screening, not for the
    final benchmark unless validated.
- Empty-review sanity check passed:
  `dataset_attrpol_hq_mechanism_repair_emptycheck_20260613.jsonl` keeps 2,093
  rows, 686 positives, 1,407 negatives, and touches 0 rows.  This verifies that
  the repair application script is inert unless review evidence exists.
- Review-audit sanity checks passed:
  - empty review fails normal thresholds (`too_few_reviews`,
    `coverage_below_min`);
  - feeding the raw pilot queue as if it were review output fails schema checks
    (`invalid_claim_quality`, `invalid_relation_to_claim`,
    `invalid_value_alignment`, etc.).
  This audit step must pass before any review file is applied to the dataset.

Planned command sequence once `MATPOOL_API_KEY` is safely available in the
environment:

```bash
PYTHONPATH=src python -m data_quality.llm_review_mechanism_repair_queue_v1 \
  --queue data/final/repaired_v1/mechanism_repair_pilot80_v1_20260613.jsonl \
  --out data/final/repaired_v1/mechanism_repair_pilot80_v1_llm_review_20260613.jsonl \
  --report data/final/repaired_v1/mechanism_repair_pilot80_v1_llm_review_20260613.report.json \
  --model Qwen3-VL-Plus --limit 0 --concurrency 2 --max_images 4

PYTHONPATH=src python -m data_quality.audit_mechanism_repair_reviews_v1 \
  --queue data/final/repaired_v1/mechanism_repair_pilot80_v1_20260613.jsonl \
  --review data/final/repaired_v1/mechanism_repair_pilot80_v1_llm_review_20260613.jsonl \
  --out data/final/repaired_v1/mechanism_repair_pilot80_v1_llm_review_audit_20260613.json \
  --min_reviews 72 --min_coverage 0.90 --max_issue_rate 0.10 \
  --require_all_review_ids_in_queue

PYTHONPATH=src python -m data_quality.apply_mechanism_repair_reviews_v1 \
  --dataset data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_source_recovered_v3_dropunresolved.jsonl \
  --review data/final/repaired_v1/mechanism_repair_pilot80_v1_llm_review_20260613.jsonl \
  --queue data/final/repaired_v1/mechanism_repair_pilot80_v1_20260613.jsonl \
  --mode conservative \
  --out data/final/repaired_v1/dataset_attrpol_hq_mechanism_repaired_conservative_v1_20260613.jsonl \
  --report data/final/repaired_v1/mechanism_repaired_conservative_v1_20260613.report.json
```

Evaluation gate:

- First run learnability and a fold-0 CLAIMARC/BGE screen on the conservative
  repaired candidate.
- Promote it only if AP/AUROC do not regress and Macro-F1/wF1 improve on the
  hardclean fold-0 anchor.  Otherwise keep the reviews as diagnostics and
  refine the repair rules.

## 2026-06-13 Addendum: Pair-Aligned Mechanism Repair Queue

The first mechanism queue exposed a pipeline bug rather than a label-quality
fact: the dataset and OOF prediction file contain the same 2,093 `pair_id`s,
and all labels/confidences match when joined by `pair_id`, but 1,889 rows are
in a different order.  The original queue builder used array position, so its
`y_current`, `c_current`, and model-failure fields could be attached to the
wrong product-attribute pair.

Guardrail changes:

- `src/data_quality/build_mechanism_repair_queue_v1.py` now audits both row
  order and pair-key alignment.  It reports row-order mismatches but reindexes
  OOF arrays by `pair_id` before scoring rows.
- `src/data_quality/apply_mechanism_repair_reviews_v1.py` now refuses to apply
  reviews unless the review/queue IDs, current dataset row index, `y_current`,
  `c_current`, `attribute_id`, and category metadata are aligned.  A stale queue
  can only be forced with `--allow_stale_queue`.
- `audit` mode in the apply script is now strictly non-mutating for `y`, `c`,
  and textual `confidence`; it only attaches review/decision metadata.

New safe artifacts:

- `data/final/repaired_v1/mechanism_repair_queue_v2_pairaligned_20260613.jsonl`
  and `.json`
  - n=400
  - `by_pair_y_mismatch_count=0`
  - `by_pair_c_mismatch_count=0`
  - `by_pair_attribute_id_mismatch_count=0`
  - `row_order_pair_id_mismatch_count=1889`, recorded as a reindexing warning
- `data/final/repaired_v1/mechanism_repair_pilot80_v2_pairaligned_20260613.jsonl`
  and `.json`
  - n=80
  - current labels: 67 clean / 13 risk
  - evidence combos: P 29, O 26, PO 20, OV 2, POV 2, V 1
  - categories: apparel 14, digital 12, baby 12, beauty 11, shoes/bags 7,
    food 7, general 7, smart-home 4, sports 3, jewelry 3

Validation checks:

- Rebuilding the old guard test correctly detects the row-order issue.
- Applying a synthetic fixture against the old pilot is rejected because the
  stale queue has mismatched `y_current`/`c_current` for reviewed rows.
- Applying the same style of fixture against the v2 pair-aligned pilot passes
  the queue consistency gate.
- The v2 audit-mode apply test keeps 2,093 rows and changes 0 labels,
  0 confidence weights, and 0 textual confidence fields.

Updated command sequence for the real LLM/VLM review:

```bash
PYTHONPATH=src python -m data_quality.llm_review_mechanism_repair_queue_v1 \
  --queue data/final/repaired_v1/mechanism_repair_pilot80_v2_pairaligned_20260613.jsonl \
  --out data/final/repaired_v1/mechanism_repair_pilot80_v2_llm_review_20260613.jsonl \
  --report data/final/repaired_v1/mechanism_repair_pilot80_v2_llm_review_20260613.report.json \
  --model Qwen3-VL-Plus --limit 0 --concurrency 2 --max_images 4

PYTHONPATH=src python -m data_quality.audit_mechanism_repair_reviews_v1 \
  --queue data/final/repaired_v1/mechanism_repair_pilot80_v2_pairaligned_20260613.jsonl \
  --review data/final/repaired_v1/mechanism_repair_pilot80_v2_llm_review_20260613.jsonl \
  --out data/final/repaired_v1/mechanism_repair_pilot80_v2_llm_review_audit_20260613.json \
  --min_reviews 72 --min_coverage 0.90 --max_issue_rate 0.10 \
  --require_all_review_ids_in_queue

PYTHONPATH=src python -m data_quality.apply_mechanism_repair_reviews_v1 \
  --dataset data/final/repaired_v1/dataset_attrpol_hq_product_rawtext_llmcurated_source_recovered_v3_dropunresolved.jsonl \
  --review data/final/repaired_v1/mechanism_repair_pilot80_v2_llm_review_20260613.jsonl \
  --queue data/final/repaired_v1/mechanism_repair_pilot80_v2_pairaligned_20260613.jsonl \
  --mode conservative \
  --out data/final/repaired_v1/dataset_attrpol_hq_mechanism_repaired_conservative_v2_20260613.jsonl \
  --report data/final/repaired_v1/mechanism_repaired_conservative_v2_20260613.report.json
```

Research implication: before spending API calls or GPU time, every repair or
expansion view must be keyed by stable IDs, not row order.  This also applies
to future grouped-CV OOF diagnostics and auxiliary-data admission gates.

## 2026-06-13 Addendum: LLM/VLM Pilot80 Review and Conservative Repair

The v2 pair-aligned pilot was reviewed with Qwen3-VL-Plus.  The prompt withheld
current labels and model scores and asked only for claim quality,
product-evidence relation, exact value alignment, likely extraction issue, and
repair action.

Review artifacts:

- `data/final/repaired_v1/mechanism_repair_pilot80_v2_llm_review_20260613.jsonl`
- `data/final/repaired_v1/mechanism_repair_pilot80_v2_llm_review_20260613.report.json`
- audit:
  `data/final/repaired_v1/mechanism_repair_pilot80_v2_llm_review_audit_20260613.json`

Review summary:

- n=80 / errors=0
- claim quality: clear 46, mixed 24, garbled 9, no_claim 1
- relation: insufficient 43, not_verifiable 14, contradicts 12, supports 11
- value alignment: ambiguous 29, not_applicable 23, contradiction 12,
  compatible 10, exact_match 6
- repair action: recover_more_evidence 46, drop_bad_claim 24, keep_relation 10
- likely issues: generic_evidence 39, value_mismatch 12, missing_evidence 10,
  none 8, plus a small tail of bad/garbled claim span and OCR/ASR noise cases

Audit status:

- pass
- coverage=1.0
- issue_rate=0.025
- two minor issues were `relation_requires_source`: the model provided concrete
  `key_evidence` but filled `evidence_source=none`.  The cleaner now maps this
  state to `mixed` for future runs when `evidence_found=true` and evidence text
  is non-empty.

Conservative repair candidate:

- dataset:
  `data/final/repaired_v1/dataset_attrpol_hq_mechanism_repaired_conservative_v2_20260613.jsonl`
- report:
  `data/final/repaired_v1/mechanism_repaired_conservative_v2_20260613.report.json`
- rows: 2,093 -> 2,048
- labels: 686 positives -> 674 positives
- touched rows: 80
- dropped rows: 45
- main actions:
  - drop_bad_claim: 24
  - drop_uncertain_weak_row: 21
  - downweight_uncertain_row: 19
  - strengthen_clean_supported: 7
  - relabel_risk_to_clean_supported: 4

Lightweight grouped learnability diagnostic:

| dataset | AP | AUROC | Macro-F1 |
|---|---:|---:|---:|
| hardclean main benchmark | 0.8467 | 0.9389 | 0.9071 |
| mechanism-repaired conservative v2 | 0.8778 | 0.9462 | 0.9159 |

Interpretation:

- This is the strongest data-quality signal in the current round.  The LLM/VLM
  review confirms that many CLAIMARC high-confidence false positives are caused
  by generic evidence, missing evidence, or bad claim spans rather than model
  architecture alone.
- The conservative candidate should proceed to a CLAIMARC fold-0 screen before
  replacing the main benchmark.  Because 45/80 reviewed rows are dropped, a full
  benchmark replacement must also report distribution shifts by category,
  evidence source, confidence, and claim type.
- The pilot review can also become a `RACL-U` utility-label pilot: supports and
  contradictions define utility-positive evidence views, while insufficient and
  not-verifiable rows define masked/ignored contrastive evidence.

Remote GPU status:

- The current A30 host is reachable and idle, but its driver is old
  (470 / CUDA 11.4).  A fresh Python 3.10 environment was created, and the
  scientific stack installed, but PyTorch CUDA installation is not yet stable:
  cu121 is incompatible with the driver, and cu113 wheel download was too slow.
- `src/models/cv_eval.py` now supports `--max_folds` so the next GPU attempt can
  run a reproducible fold-0 screen instead of relying on manual interruption.

## 2026-06-13 Addendum: Fold-0 CLAIMARC Screen on Mechanism Repair v2

The conservative mechanism-repaired candidate passed the lightweight grouped
learnability gate, but its first end-to-end CLAIMARC screen shows a more
specific pattern: data repair improves risk ranking, while fold-local threshold
selection remains unstable.

Remote artifacts:

- `data/final/cleancl/remote_results_20260613/cv_attrpol_mechrepair_v2_fold0_w025_cap1500_bs8_s0_bge_20260613.json`
- `data/final/cleancl/remote_results_20260613/oof_attrpol_mechrepair_v2_fold0_w025_cap1500_bs8_s0_bge_20260613.npz`
- `data/final/cleancl/remote_results_20260613/cv_attrpol_mechrepair_v2_fold0_valmacro_w025_cap1500_bs8_s0_bge_20260613.json`
- `data/final/cleancl/remote_results_20260613/oof_attrpol_mechrepair_v2_fold0_valmacro_w025_cap1500_bs8_s0_bge_20260613.npz`

Fold-0 screen, same hardclean auxiliary setting
(`weight_scale=0.25`, cap 1,500, source0 CE/CL 0.5/0.25, `cm_seeds=[0]`):

| method / threshold policy | AP | AUROC | Macro-F1 | wF1 |
|---|---:|---:|---:|---:|
| CLAIMARC PCLS / prior-stable | 0.9043 | 0.9485 | 0.8581 | 0.7598 |
| CLAIMARC v2 / prior-stable | 0.9182 | 0.9507 | 0.8688 | 0.7828 |
| BGE-LR / prior-stable | 0.8820 | 0.9486 | 0.8892 | 0.7926 |
| CLAIMARC PCLS / val-macro | 0.9043 | 0.9485 | 0.8559 | 0.7591 |
| CLAIMARC v2 / val-macro | 0.9182 | 0.9507 | 0.8562 | 0.7711 |
| BGE-LR / val-macro | 0.8820 | 0.9486 | 0.8892 | 0.7926 |

OOF threshold diagnosis was formalized in
`src/models/diagnose_oof_thresholds.py`.  On the prior-stable fold-0 OOF:

| method | saved Macro-F1 | oracle Macro-F1 | saved pred rate | oracle threshold |
|---|---:|---:|---:|---:|
| CLAIMARC v2 | 0.8688 | 0.8921 | 0.4013 | 0.76 |
| BGE-LR | 0.8892 | 0.8945 | 0.3525 | 0.55 |

Interpretation:

- Mechanism repair v2 is not yet a clean replacement for the main benchmark:
  it gives CLAIMARC the strongest AP/AUROC seen in this fold, but the deployed
  decision threshold overpredicts positives.
- The gap is not a simple `val_macro` switch.  CLAIMARC needs
  evidence-conditioned calibration or utility-aware retrieval masks, especially
  for generic OCR/title evidence and exact-parameter compatibility cases.
- The 80-row review should be expanded before changing labels globally.  The
  next data step is a larger pair-aligned LLM/VLM audit of the same queue,
  followed by deterministic actions: drop bad claim spans, recover missing
  product evidence, or convert support/contradiction states into retrieval
  utility labels.

## 2026-06-13 Addendum: Full400 Mechanism Review and RACL-U Fold-0 Pilot

The pair-aligned mechanism queue was expanded from the 80-row pilot to the full
400-row high-priority queue using the same blinded Qwen3-VL-Plus review prompt.
The review does not see current labels or model predictions.

Review artifacts:

- `data/final/repaired_v1/mechanism_repair_full400_v2_llm_review_20260613.jsonl`
- `data/final/repaired_v1/mechanism_repair_full400_v2_llm_review_batch_20260613.report.json`
- `data/final/repaired_v1/mechanism_repair_full400_v2_llm_review_audit_20260613.json`

Audit summary:

- n_queue=400, n_reviews=400, unique pair_ids=400
- errors=0
- audit status=pass
- coverage=1.0
- issue_rate=0.01
- valid reviews=396/400

Full400 review distribution:

| field | dominant values |
|---|---|
| claim_quality | clear 222, mixed 125, garbled 42, no_claim 7 |
| relation_to_claim | insufficient 197, not_verifiable 85, contradicts 69, supports 45 |
| value_alignment | ambiguous 132, not_applicable 124, contradiction 76, compatible 40, exact_match 24 |
| repair_action | recover_more_evidence 219, drop_bad_claim 134, keep_relation 42 |
| likely_issue | generic_evidence 167, value_mismatch 69, missing_evidence 66, bad_claim_span 33 |

Two deterministic repair candidates were generated:

| candidate | rows | y=1 | main action |
|---|---:|---:|---|
| conservative full400 v3 | 1,828 | 631 | drop bad claims and weak unverifiable rows |
| softdropbad full400 v3 | 1,956 | 650 | drop bad claims; downweight insufficient/not-verifiable rows |

Lightweight grouped learnability:

| dataset | AP | AUROC | Macro-F1 |
|---|---:|---:|---:|
| hardclean main benchmark | 0.8467 | 0.9389 | 0.9071 |
| mechanism-repaired conservative v2 | 0.8778 | 0.9462 | 0.9159 |
| conservative full400 v3 | 0.8877 | 0.9482 | 0.9251 |
| softdropbad full400 v3 | 0.8856 | 0.9529 | 0.9278 |

Fold-0 end-to-end screen on `softdropbad full400 v3`:

| setting | AP | AUROC | Macro-F1 | wF1 | note |
|---|---:|---:|---:|---:|---|
| default RACL | 0.8759 | 0.9612 | 0.9155 | 0.8358 | ordinary CL includes low-confidence rows |
| RACL-U mask (`cl_c_min=0.2`, `cl_neg_c_min=0.2`) | 0.9136 | 0.9675 | 0.9304 | 0.8607 | low-utility rows masked from contrastive learning |
| BGE-LR | 0.8726 | 0.9666 | 0.9315 | 0.8703 | same fold/test rows |

OOF threshold diagnosis for RACL-U:

- CLAIMARC PCLS saved Macro-F1 0.9304; oracle Macro-F1 0.9330.
- CLAIMARC PCLS saved wF1 0.8607; oracle wF1 0.8695.
- BGE-LR saved Macro-F1 0.9315; oracle Macro-F1 0.9374.
- The RACL-U mask shrinks CLAIMARC's calibration gap to 0.0027 Macro-F1, down
  from 0.0233 on the earlier conservative v2 fold-0 screen.

Interpretation:

- The full400 review confirms that the original raw-to-training pipeline often
  produced learnable-looking but weakly grounded samples: generic evidence,
  missing evidence, bad claim spans, and value mismatches dominate the repair
  queue.
- `softdropbad full400 v3` is the current best data candidate because it removes
  bad claim spans without deleting too many consumer-risk boundary cases.
- RACL-U is now the most promising model line: a simple confidence/utility mask
  over contrastive anchors and negatives gives a large AP/AUROC gain and almost
  closes the Macro-F1 gap to BGE-LR while preserving the retrieval-augmented
  contrastive mechanism.
