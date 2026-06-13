# Full Pair Reconstruction Protocol v1

## Why This Reset Exists

The proposal task is not a generic product-quality classifier and not an
objective product-page fact checker.  The unit is a natural live-commerce
consumer-perception event:

`(product, attribute, livestream claim, product-side evidence, consumer comments)`

The old filtered datasets kept only rows where the upstream pipeline had already
found a usable claim/evidence pair.  This created a measurement problem: many
valid product-attribute pairs were excluded because claim extraction or product
evidence extraction failed, and labels for missing-claim rows were often treated
as negatives without actually comparing comments against the relevant streamer
claim.

Therefore the main data line is reset to all product-scope pairs from the
proposal audit.  Old labels and confidence scores are audit signals only.

## Full-Pair Coverage

Source audit:

- `data/final/repaired_v1/proposal_quality_audit_all_v1_20260613.jsonl`

Full reconstruction queue:

- `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.report.json`
- queue audit:
  `docs/FULL_PAIR_RECONSTRUCTION_QUEUE_AUDIT_20260614.md`
  and `data/final/repaired_v1/full_pair_reconstruction_queue_audit_v1_20260614.json`

Summary:

| item | count |
|---|---:|
| product-attribute pairs | 13,769 |
| skipped non-product-scope rows | 2,910 |
| pairs with Stage-A comment mentions | 13,769 |
| pairs with negative comment mentions | 6,900 |
| pairs with explicit fact-hit comment mentions | 3,170 |

The user-mentioned total 13,796 is interpreted as the same full product-attribute
population; the current audit artifact contains 13,769 rows.

## Missingness Diagnosis

| state | count |
|---|---:|
| claim missing | 10,666 |
| claim present but needs review | 1,646 |
| claim present and specific | 1,457 |
| product evidence missing | 6,961 |
| single-source product evidence | 4,550 |
| multi-source product evidence | 2,258 |

Old label states:

| old label state | count |
|---|---:|
| old negative with no aligned review | 12,219 |
| old positive with claim-aligned negative review | 966 |
| old negative with claim-aligned nonnegative review | 584 |

This confirms the core defect: the biggest pool is not necessarily true
negative data; it is mostly "claim/evidence not recovered, so comment was never
properly compared to the same claim".

## Queue Design

Builder:

- `src/data_quality/build_full_pair_reconstruction_queue_v1.py`

Queue distribution:

| queue type | count |
|---|---:|
| full claim/evidence/label rebuild | 10,394 |
| claim re-extract + label rebuild | 1,918 |
| evidence refresh + label rebuild | 1,117 |
| label rebuild on existing triplet | 340 |

Priority:

| priority | count |
|---|---:|
| P0 | 6,336 |
| P1 | 5,202 |
| P2 | 2,016 |
| P3 | 215 |

The priority score is only a scheduling device.  It does not make any row easier
or cleaner for training, and it does not delete hard samples.

## Pre-LLM Queue Audit

Audit builder:

- `src/data_quality/audit_full_pair_reconstruction_queue_v1.py`

Additional diagnostics:

| signal | count |
|---|---:|
| compact comment mentions typed as attribute | 49,554 |
| compact comment mentions typed as service | 778 |
| rows with service comments in compact mentions | 395 |
| commercial-promise attributes | 531 |
| rows whose compact comments mention streamer/live/promotion cues | 927 |

This audit supports the reset: many rows are not clean negatives; they are
measurement states where consumer comments may explicitly refer to the streamer
claim, but upstream claim/evidence extraction failed or drifted.

## SRT Claim Prefilter

SRT prefilter builder:

- `src/data_quality/build_srt_claim_prefilter_v1.py`

Outputs:

- candidate file:
  `data/final/repaired_v1/full_pair_claim_srt_prefilter_v1_20260614.jsonl`
- report:
  `data/final/repaired_v1/full_pair_claim_srt_prefilter_v1_20260614.report.json`
- markdown:
  `docs/FULL_PAIR_CLAIM_SRT_PREFILTER_20260614.md`

Coverage over the 12,312 rows needing claim repair or claim review:

| state | count |
|---|---:|
| strong SRT candidate | 3,322 |
| weak SRT candidate | 6,769 |
| very weak SRT candidate | 2,141 |
| no SRT candidate | 80 |

Among the 10,666 rows previously marked `claim_missing`, only 77 have no
deterministic SRT candidate.  This strongly suggests the old claim extraction
pipeline mostly failed at recall/alignment rather than because the raw
livestream material lacked relevant claim text.

The LLM/VLM runner now loads this prefilter by default through
`--srt_prefilter`, so review prompts expose ranked SRT candidates and hit
reasons instead of relying on a shallow per-call keyword scan.

## Stratified LLM Pilot Queue

Pilot queue builder:

- `src/data_quality/build_full_pair_llm_pilot_queue_v1.py`

Outputs:

- pilot queue:
  `data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl`
- report:
  `data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.report.json`
- markdown:
  `docs/FULL_PAIR_LLM_PILOT_QUEUE_20260614.md`

The pilot contains 72 P0 rows sampled across SRT candidate states:

| state | rows |
|---|---:|
| strong SRT candidate | 24 |
| weak SRT candidate | 24 |
| very weak SRT candidate | 16 |
| no SRT candidate | 8 |

It also covers 10 product categories, 58 `claim_missing` rows, and 14
`claim_present_review_needed` rows.  This avoids evaluating the reconstruction
protocol only on easy strong-candidate cases.

## Rebuild Rules

For every pair, the LLM/VLM verifier must recover or judge:

1. A minimal continuous streamer claim from raw SRT.
2. Product-side evidence from title, parameters, detail-image OCR, or
   detail-image visual observation.
3. Attribute-level consumer comments that support, refute, or do not align to
   that exact repaired claim.

Final label rule:

- `new_y=1` only when a recovered claim exists and at least one attribute-level
  consumer comment is aligned to the same claim and refutes it.
- `new_y=0` when comments support the claim, discuss the attribute without
  contradicting the claim, or no claim can be recovered.
- Product evidence contradicting the claim is not enough by itself to create a
  positive consumer-perception label.  It can be a mechanism variable or an
  evidence relation state, but the target label still requires consumer
  refutation.

Old `y` and `c` are stored as `old_y` and `old_c` for auditing only.

## Service-Like Attribute Boundary

The audit source is product-scope, but historical Stage-A comments sometimes
mark transaction promises such as price protection, delivery, warranty, or
after-sales service as `service`.  These should not be blindly deleted.  The
verifier should treat them as commercial-promise attributes only if the streamer
claim and the consumer comment discuss the same promise.  Otherwise they become
`not_aligned` comments and cannot trigger `new_y=1`.

This preserves hard cases while avoiding a category-boundary shortcut.

## LLM/VLM Runner

Runner:

- `src/data_quality/llm_full_pair_reconstruct_v1.py`

Default pilot command:

```bash
PYTHONPATH=src python3 -m data_quality.llm_full_pair_reconstruct_v1 \
  --queue data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl \
  --priority P0 \
  --limit 72 \
  --concurrency 2 \
  --model Qwen3-VL-Plus
```

One-command guarded runner:

```bash
scripts/run_full_pair_reconstruction_pilot.sh P0 72 2 Qwen3-VL-Plus \
  data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl
```

The guarded runner now executes LLM reconstruction, the review audit gate, and
promotion in sequence.  Promotion is blocked if the audit still has missing
reviews or any `high` flags.  The default shell-script limit remains `20` for
cheap smoke tests; use `72` for the full stratified pilot.

Local non-API check:

```bash
PYTHONPATH=src python3 -m data_quality.llm_full_pair_reconstruct_v1 \
  --queue data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl \
  --dry_run
```

The runner enforces `new_y` after parsing model output: even if the model emits
`new_y=1`, the saved label is reset to `0` unless there is a recovered claim and
at least one aligned `refute` comment judgment.

The LLM/VLM prompt does not expose old `y/c`; those fields remain available only
in downstream audit outputs.  This avoids anchoring the relabeling pass to the
old filtered pipeline.

The output is an audit artifact, not a direct training dataset.  A later
promotion step must verify provenance, source coverage, split hygiene, and
label-confidence calibration before building the final supervised benchmark.

## LLM Review Audit Gate

Review audit script:

- `src/data_quality/audit_full_pair_llm_reviews_v1.py`

Default command after the LLM/VLM pilot:

```bash
PYTHONPATH=src python3 -m data_quality.audit_full_pair_llm_reviews_v1 \
  --queue data/final/repaired_v1/full_pair_llm_pilot_queue_v1_20260614.jsonl \
  --reviews data/final/repaired_v1/full_pair_reconstruction_llm_v1_20260614.jsonl
```

Outputs:

- report:
  `data/final/repaired_v1/full_pair_reconstruction_llm_audit_v1_20260614.report.json`
- flagged rows:
  `data/final/repaired_v1/full_pair_reconstruction_llm_audit_flags_v1_20260614.jsonl`
- markdown:
  `docs/FULL_PAIR_LLM_REVIEW_AUDIT_20260614.md`

This gate checks label-definition consistency rather than benchmark
separability.  `high` flags block main promotion until rerun or manual repair;
`medium` flags require manual sampling or silver routing.  Product evidence
contradiction without an aligned consumer `refute` comment is explicitly marked
as a mechanism state, not a positive label.

Manual audit packet:

- `src/data_quality/build_full_pair_manual_audit_packet_v1.py`

Default command:

```bash
PYTHONPATH=src python3 -m data_quality.build_full_pair_manual_audit_packet_v1
```

Outputs:

- CSV packet:
  `data/final/repaired_v1/full_pair_manual_audit_packet_v1_20260614.csv`
- report:
  `data/final/repaired_v1/full_pair_manual_audit_packet_v1_20260614.report.json`
- markdown:
  `docs/FULL_PAIR_MANUAL_AUDIT_PACKET_20260614.md`

The packet joins queue metadata, SRT prefilter candidates, product evidence
previews, consumer comment snippets, optional LLM reviews, and optional audit
flags.  It is for human inspection and adjudication, not for selecting only
easy rows.

## Remote Pilot20 No-Image Finding

Remote smoke pilot on the 2026-06-14 GPU host used the stratified pilot queue
with `--limit 20`, `--max_images 0`, and `Qwen3-VL-Plus`.  This run was a data
protocol check, not a model benchmark.

Outputs:

- reviews:
  `data/final/repaired_v1/full_pair_reconstruction_llm_pilot20_noimg_v1_20260614.jsonl`
- LLM report:
  `data/final/repaired_v1/full_pair_reconstruction_llm_pilot20_noimg_v1_20260614.report.json`
- audit v2 report:
  `data/final/repaired_v1/full_pair_reconstruction_llm_pilot20_noimg_audit_v2_20260614.report.json`
- manual packet:
  `data/final/repaired_v1/full_pair_manual_audit_packet_pilot20_noimg_v2_20260614.csv`

Pilot summary:

- 20 reviewed rows: 8 recovered claims, 12 missing-claim repairs.
- 8 no-SRT-candidate rows behaved as boundary checks: product evidence can be
  found, but labels remain negative/repair because no streamer claim is
  recoverable.
- 12 strong-SRT rows produced 5 `main_positive_refute`, 2
  `main_negative_support`, 1 `silver_refute_insufficient_product_evidence`,
  and 4 claim/evidence repairs.
- The audit caught one important protocol violation: an LLM review proposed
  promotion although `claim_evidence_relation=insufficient`.  Promotion now
  requires a recovered claim, product evidence, aligned consumer relation, and a
  non-`insufficient` claim-evidence relation.

This finding supports continuing full-pair reconstruction, but the next pass
must keep the audit gate active and should route insufficient product evidence
to Stage C/VLM repair rather than main training.

## Remote Pilot72 No-Image Audit v3

The follow-up full stratified pilot used all 72 rows from
`full_pair_llm_pilot_queue_v1_20260614` with `--max_images 0`.  It tests whether
SRT, title/params, OCR text, and consumer snippets are sufficient before adding
detail-image VLM calls.

Outputs:

- reviews:
  `data/final/repaired_v1/full_pair_reconstruction_llm_pilot72_noimg_v1_20260614.jsonl`
- LLM report:
  `data/final/repaired_v1/full_pair_reconstruction_llm_pilot72_noimg_v1_20260614.report.json`
- audit v3 report:
  `data/final/repaired_v1/full_pair_reconstruction_llm_pilot72_noimg_audit_v3_20260614.report.json`
- manual packet:
  `data/final/repaired_v1/full_pair_manual_audit_packet_pilot72_noimg_v3_20260614.csv`

Audit v3 summary:

- matched reviews: 72/72.
- claim recovered: 28/72.
- main candidates: 12 `main_positive_refute`, 3 `main_negative_support`.
- repair/silver states: 43 `repair_missing_claim`, 5
  `silver_refute_missing_product_evidence`, 3
  `silver_refute_insufficient_product_evidence`, 3
  `repair_insufficient_product_evidence`, 1 `repair_missing_evidence`, 1
  `lowinfo_no_aligned_comment`, and 1 `llm_error`.
- audit flags: 1 high `llm_error`, 9 medium flags.  The medium flags mostly
  identify positive consumer refute comments without sufficient product-side
  evidence, or claim spans not found in the deterministic top SRT prefilter.

Engineering correction from this pilot: anonymous LLM parse failures are now
converted into pair-aware `llm_error` rows, preserving auditability and avoiding
unmatched reviews.

Next expansion rule: do not promote the 20 positive labels directly.  Only the
15 main candidates can seed the supervised candidate view after manual sampling;
the 8 silver positive/insufficient-evidence rows should trigger Stage C/VLM
evidence repair.  The 43 missing-claim rows should be used to refine SRT
retrieval/prompting before scaling to all 13,769 pairs.

## Stage-C/VLM Evidence Repair Queue

Evidence repair queue builder:

- `src/data_quality/build_full_pair_evidence_repair_queue_v1.py`

Default command from the pilot72 no-image reviews:

```bash
PYTHONPATH=src python3 -m data_quality.build_full_pair_evidence_repair_queue_v1
```

Outputs:

- repair queue:
  `data/final/repaired_v1/full_pair_evidence_repair_queue_v1_20260614.jsonl`
- report:
  `data/final/repaired_v1/full_pair_evidence_repair_queue_v1_20260614.report.json`
- markdown:
  `docs/FULL_PAIR_EVIDENCE_REPAIR_QUEUE_20260614.md`

The builder selects rows where the claim has been recovered and consumer
comments already refute the repaired claim, but product-side evidence is missing
or `claim_evidence_relation=insufficient`.  It seeds the next prompt with the
recovered SRT claim, keeps old labels as audit-only fields, and routes the row
back to title/params/OCR/VLM evidence search.  It does not delete hard rows,
convert them to negatives, or promote them into training data.

Pilot72 queue summary:

| item | count |
|---|---:|
| reviewed rows | 72 |
| evidence-repair rows | 8 |
| source evidence missing | 5 |
| source evidence present but insufficient | 3 |
| rows with detail images | 8 |

This codifies the data-quality correction requested by the user: quality
improvement means recovering the proposal-faithful claim/evidence/comment chain,
not making AUROC easier by filtering ambiguous rows.

## Remote Evidence-Repair8 VLM Audit

The first Stage-C/VLM repair run used the 8 pilot72 silver-positive rows with
detail images and `Qwen3-VL-Plus`.

Outputs:

- reviews:
  `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair8_vlm_v1_20260614.jsonl`
- audit report:
  `data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair8_vlm_audit_v1_20260614.report.json`
- audit markdown:
  `docs/FULL_PAIR_EVIDENCE_REPAIR8_VLM_AUDIT_20260614.md`
- manual packet:
  `data/final/repaired_v1/full_pair_manual_audit_packet_evidence_repair8_vlm_v1_20260614.csv`
  and
  `docs/FULL_PAIR_MANUAL_AUDIT_PACKET_EVIDENCE_REPAIR8_VLM_20260614.md`

Audit summary:

| state | count |
|---|---:|
| `main_positive_refute` | 3 |
| `silver_refute_missing_product_evidence` | 3 |
| `silver_refute_insufficient_product_evidence` | 2 |
| high flags | 0 |
| medium flags | 4 |

Promotion consistency check:

- stateful rows:
  `data/final/repaired_v1/dataset_full_pair_evidence_repair8_stateful_v1_20260614.jsonl`
- main rows:
  `data/final/repaired_v1/dataset_full_pair_evidence_repair8_main_v1_20260614.jsonl`
- repair/silver rows:
  `data/final/repaired_v1/full_pair_evidence_repair8_repair_silver_v1_20260614.jsonl`
- promotion report:
  `data/final/repaired_v1/full_pair_evidence_repair8_promotion_v1_20260614.report.json`

The promotion builder independently reproduces the audit state: 3 main positive
rows and 5 silver/repair rows.

Interpretation:

- VLM repair can move some positive consumer-refute rows from silver into the
  conservative main candidate once product-side evidence becomes source-backed.
- The 8 rows cannot all be promoted: 5 still need product evidence or stronger
  claim-evidence relation.
- Subjective attributes such as visual style can remain in the consumer
  perception task when the comment directly answers the same claim, but they
  should be tracked as a separate mechanism slice rather than used to inflate
  objective fact-checking claims.

## Stage-B Claim Repair Queue

Claim repair queue builder:

- `src/data_quality/build_full_pair_claim_repair_queue_v1.py`

Default command from the pilot72 no-image reviews:

```bash
PYTHONPATH=src python3 -m data_quality.build_full_pair_claim_repair_queue_v1
```

Outputs:

- claim repair queue:
  `data/final/repaired_v1/full_pair_claim_repair_queue_v1_20260614.jsonl`
- markdown:
  `docs/FULL_PAIR_CLAIM_REPAIR_QUEUE_20260614.md`

The builder routes rows with missing recovered claim or LLM parse errors to a
pair-targeted claim-only SRT scanner.  It attaches consumer trigger comments and
product evidence hints, but it does not use comments as claim text.  Rows where
comments say "直播/主播说过" but no exact SRT source exists remain
source-missing until an original claim can be recovered.

Pilot72 result:

| item | count |
|---|---:|
| reviewed rows | 72 |
| claim-repair rows | 44 |
| strong SRT candidate | 6 |
| weak SRT candidate | 15 |
| very weak SRT candidate | 15 |
| no SRT candidate | 8 |

The same builder can now seed claim-repair queues directly from the full
13,769-row reconstruction queue by merging
`full_pair_claim_srt_prefilter_v1_20260614.jsonl`.  The first full-scale seed
batch is:

- queue:
  `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak120_v1_20260614.jsonl`
- markdown:
  `docs/FULL_PAIR_CLAIM_REPAIR_QUEUE_FULL_P0_STRONGWEAK120_20260614.md`

It selects 120 P0 rows with strong/weak SRT candidates for pair-targeted claim
re-extraction.  In the generated batch all 120 are strong SRT candidates, 23
come from old positive claim-aligned-negative rows, and 96 are old negatives
with strong consumer trigger comments that were never properly compared to a
recovered claim.

The first 20 rows of this full seed batch were processed through the same
claim-only and joint-review stages:

- claim re-extract output:
  `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak20_v1_20260614.jsonl`
- joint review queue:
  `docs/FULL_PAIR_CLAIM_REEXTRACT_FULL_P0_STRONGWEAK20_JOINT_REVIEW_QUEUE_20260614.md`
- joint review audit:
  `docs/FULL_PAIR_CLAIM_REEXTRACT_FULL_P0_STRONGWEAK17_NOIMG_AUDIT_20260614.md`
- promotion report:
  `docs/FULL_PAIR_CLAIM_REEXTRACT_FULL_P0_STRONGWEAK17_NOIMG_PROMOTION_20260614.md`

Full-seed20 result:

| step | count |
|---|---:|
| claim-only processed rows | 20 |
| exact SRT claim found | 17 |
| no exact SRT claim | 3 |
| joint-review rows | 17 |
| joint reviewer `claim_found` | 12 |
| conservative main rows after identity gate | 6 |
| silver evidence-repair rows | 4 |
| repair missing-claim rows | 5 |
| identity claim-value repairs | 2 |

This batch is the first strong signal that full-scale data expansion is viable:
the high-recall claim repair queue can recover many missing-claim rows, while
the joint reviewer and promotion gate still reject noisy, adjacent-attribute, or
identity-inferred claims.

Claim-only re-extraction with `llm_pair_claim_reextract_v1` on these 44 rows
found exact SRT claim candidates for 23 rows and no claim for 21 rows.  This
confirms that the missing-claim pool mixes true source-missing rows with
recoverable recall failures.

## Claim-Reextract Joint Review

Claim re-extract review queue builder:

- `src/data_quality/build_full_pair_claim_reextract_review_queue_v1.py`

Outputs:

- joint review queue:
  `data/final/repaired_v1/full_pair_claim_reextract_joint_review_queue_v1_20260614.jsonl`
- markdown:
  `docs/FULL_PAIR_CLAIM_REEXTRACT_JOINT_REVIEW_QUEUE_20260614.md`

The bridge injects exact SRT claim candidates back into the full-pair reviewer;
it is a recall-to-review step, not a promotion step.  The subsequent no-image
joint review over 23 claim-found rows produced:

| state | count |
|---|---:|
| `repair_missing_claim` | 17 |
| `repair_insufficient_product_evidence` | 2 |
| `main_positive_refute` before identity gate | 2 |
| `silver_refute_insufficient_product_evidence` | 1 |
| `lowinfo_no_aligned_comment` | 1 |

The audit v2 gate adds a high-severity
`identity_attribute_claim_lacks_value` flag for identity attributes such as
brand/model/SKU/barcode when the SRT claim does not itself contain the identity
value.  The promotion builder now mirrors this rule with
`repair_identity_claim_value`, so a model cannot infer an unstated brand claim
from product evidence or consumer comments.

Promotion after the identity gate:

| state | count |
|---|---:|
| conservative main rows | 1 |
| `repair_identity_claim_value` | 1 |
| remaining repair/silver rows | 21 |

This finding sets the scaling rule: claim-only re-extraction should be used as a
high-recall Stage-B repair layer, but final labels still require full joint
review, identity-value checks, product evidence, and comment-same-claim
alignment.

## Promotion Builder

Promotion builder:

- `src/data_quality/build_full_pair_promoted_dataset_v1.py`

Default command after LLM/VLM reviews exist:

```bash
PYTHONPATH=src python3 -m data_quality.build_full_pair_promoted_dataset_v1
```

Outputs:

- stateful reviewed view:
  `data/final/repaired_v1/dataset_full_pair_reconstruction_stateful_v1_20260614.jsonl`
- conservative main supervised candidate:
  `data/final/repaired_v1/dataset_full_pair_reconstruction_main_v1_20260614.jsonl`
- repair/silver queue:
  `data/final/repaired_v1/full_pair_reconstruction_repair_silver_v1_20260614.jsonl`
- report:
  `data/final/repaired_v1/full_pair_reconstruction_promotion_v1_20260614.report.json`

Promotion states:

- `main_positive_refute`: claim found, product evidence found, and at least one
  aligned consumer comment refutes the same claim; the product evidence must be
  related to the claim rather than `insufficient`.
- `main_negative_support`: claim found, product evidence found, and aligned
  consumer comments support rather than refute the claim; the product evidence
  must be related to the claim rather than `insufficient`.
- `silver_refute_missing_product_evidence`,
  `silver_refute_insufficient_product_evidence`, `repair_missing_claim`,
  `repair_missing_evidence`, `repair_insufficient_product_evidence`,
  `repair_identity_claim_value`, `silver_mixed_comment_relation`, and
  `lowinfo_no_aligned_comment` remain outside the main benchmark but are
  preserved for repair, weighting, and mechanism analysis.

The builder uses conservative reliability weights.  A single high-confidence
LLM/VLM reconstruction with one explicit aligned refuting comment receives a
moderate high weight around `0.70`, not a near-gold weight.

## Immediate Next Steps

1. Run a P0 pilot once the remote API environment is configured, then manually
   inspect at least 30 reconstructed examples across categories and attribute
   families.
2. Run the LLM review audit gate and inspect all `high` flags plus a stratified
   sample of `medium` flags before any row enters the main benchmark.
3. Run the promotion builder after the pilot to inspect state distribution and
   confirm that main rows have complete `(claim, product evidence,
   comment-aligned label)` provenance.
4. Rebuild grouped train/validation/test splits by product or room before any
   model comparison.
5. Re-run baseline and CLAIMARC experiments only after the new full-pair
   promotion artifact is created; the older 910/481 datasets remain diagnostics.
