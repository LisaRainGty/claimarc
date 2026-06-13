# CLAIMARC Data Pipeline Repair Notes

Date: 2026-06-12

## Why This Repair Was Needed

The raw-to-training audit found three label-critical issues:

1. Stage B4/B5 stored global review polarity in `pair_records.jsonl`, while the
   weak-label rule requires attribute-level polarity from Stage A.
2. HQ/broad/adjudicated merged datasets inherited `split` values from auxiliary
   verified/argument files, causing room-level split leakage in later variants.
3. Adjudicated datasets treated `insufficient/not_verifiable + medium/high risk`
   as positive evidence risk, which can turn evidence missingness into a label
   shortcut.

Stage C also showed that `source0` should be interpreted as source coverage, not
confidence. Some `source0` pairs are real extraction gaps, but many are service,
process, or subjective perception attributes that are not directly verifiable
from product factual sources.

## Code Changes

- `src/data_quality/audit_raw_pipeline_integrity.py`
  - Claim grounding now validates against parsed SRT cue text from
    `common.srt.concat_product_srt`, reducing false misses for cross-cue claims.
- `src/data_quality/build_dataset_hq_v1.py`
  - `merge_records()` now protects `pair_id`, `product_id`, `attribute_id`,
    `room_id`, and `split` from auxiliary dataset overwrite.
- `src/data_quality/rebuild_repaired_datasets_v1.py`
  - New versioned rebuild script. It does not overwrite canonical artifacts.
  - Repairs review polarity with Stage A attribute-level polarity.
  - Recomputes pair labels and sample weights.
  - Reassigns room-level splits with zero room leakage.
  - Adds `_attribute_scope` labels for product attributes vs service/process vs
    subjective/personal evaluation.
  - Builds conservative objective datasets using only `evidence_state=contradicted`
    as objective positives and `supported + low/none risk` as objective negatives.
- `src/data_quality/repair_stagea_schema_v1.py`
  - Writes `data/processed/stageA_repaired_v1/` without touching current Stage A.
  - Strips angle-bracket placeholders, removes service/evaluation aliases, drops
    clearly non-product canonical attributes, and merges duplicate canonical names.
- `src/common/srt.py` and `src/stage_b/b1_claim_extract.py`
  - B1 now stores `char_end`, validates extracted claim text against the source
    character interval, and writes timestamps spanning all overlapped cues rather
    than only the single cue with largest overlap.
- `src/stage_c/c3_ocr.py`
  - C3 now discards OCR evidence whose `raw_text` cannot be found in the OCR
    cache for the reported image or any other image for that product.
- `src/stage_c/c4_vlm.py`
  - C4 now supports `--rerun-empty` so silent empty VLM outputs can be repaired.

## Main Outputs

All outputs are under `data/final/repaired_v1/`.

| dataset | n | y=1 | source0 | room leakage | intended use |
|---|---:|---:|---:|---:|---|
| `dataset_attrpol_claimful_v1.jsonl` | 3514 | 1059 | 1469 | 0 | broad repaired consumer-perception pool |
| `dataset_attrpol_hq_claimful_v1.jsonl` | 2647 | 1059 | 892 | 0 | broad HQ consumer-perception experiment |
| `dataset_attrpol_hq_product_v1.jsonl` | 2364 | 966 | 727 | 0 | recommended main dataset |
| `dataset_objective_contradicted_v1.jsonl` | 1347 | 274 | 0 | 0 | objective-risk ablation |
| `dataset_objective_contradicted_product_v1.jsonl` | 1275 | 269 | 0 | 0 | product-only objective-risk ablation |

The polarity repair changed 24,828 comment polarities, flipped 426 pairs from
0 to 1 and 10 pairs from 1 to 0. The final all-pair label count changed from
643 positives to 1059 positives.

Stage A schema repair changed 4155 CAS+ attributes to 3770, dropping 370
service/evaluation attributes and merging 15 duplicate canonical clusters.
The product-scope resolved aspect file contains 151,717 mentions and 14,879
unique product-attribute pairs.

## Learnability Check

Lightweight grouped learnability diagnostics:

| dataset | AUPRC | AUROC | Macro-F1 |
|---|---:|---:|---:|
| `dataset_attrpol_hq_claimful_v1.jsonl` | 0.7510 | 0.7980 | 0.7463 |
| `dataset_attrpol_hq_product_v1.jsonl` | 0.7326 | 0.7965 | 0.7542 |
| `dataset_objective_contradicted_v1.jsonl` | 0.5261 | 0.7985 | 0.6679 |
| `dataset_objective_contradicted_product_v1.jsonl` | 0.5188 | 0.8001 | 0.6669 |

Interpretation: `dataset_attrpol_hq_product_v1.jsonl` is currently the best
main candidate because it is larger than the old HQ dataset, has no split
leakage, fixes the polarity bug, removes service/personal attributes, and has
the strongest simple-model Macro-F1. Objective contradicted-only data is cleaner
semantically but much smaller and imbalanced; it should be framed as an
objective verification ablation rather than the primary benchmark.

## Remaining Repair Sets

- Stage A schema cleanup: normalize invalid `FREE_...` forms, remove placeholder
  attributes like `<...>`, and tighten subjective/evaluative attribute leakage.
  First deterministic repair is available in `data/processed/stageA_repaired_v1/`;
  the next full regeneration should start from this schema.
- Stage B claim grounding: 2102/11696 claim rows still cannot be found in parsed
  cue text after normalization. These should be reviewed before claiming exact
  quote grounding.
- Stage C evidence repair: rerun OCR substring validation and VLM for non-food
  products with silent empty VLM outputs.
- Missing-claim risk: 1914 no-claim pairs contain strong negative explicit-fact
  reviews. These should not be used as clean negatives; they are a separate
  missing-claim diagnostic pool.
