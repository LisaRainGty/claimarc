# Next1000-after500 Pilot40 Data Repair

Date: 2026-06-14

## Motivation

The GPU CV run on the current stateful v2 dataset did not beat BGE-LR on ranking. The next step is therefore data reconstruction, not aggressive model tuning. This pilot tests whether unprocessed full-pair rows can recover proposal-faithful claim/evidence/comment triplets from raw SRT and product material.

## Queue

- Source queue: `data/final/repaired_v1/full_pair_reconstruction_queue_v1_20260614.jsonl`
- New claim repair queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next1000_after500_v1_20260614.jsonl`
- Selection: P0, `claim_missing|claim_present_review_needed`, `strong_srt_candidate|weak_srt_candidate`
- Exclusion: prior claim-repair/reextract/joint-review batches
- Size: 1,000 rows
- SRT prefilter mix: 956 strong / 44 weak
- Trigger comments: 763 rows with 1 trigger, 193 with 2-4, 44 with 5+

This is a more natural batch than the previous top-trigger batches. It should improve coverage without making the task artificially easy.

## Re-Extraction Pilot

Ran the first 40 rows through pair-targeted SRT claim re-extraction:

- `claim_found`: 35/40
- `no_claim_found`: 5/40
- claim count buckets: 5 rows with 0, 6 with 1, 16 with 2-4, 8 with 5-10, 5 with 11+

This confirms that the earlier missing-claim bottleneck is at least partly an extraction failure, not true absence of streamer claims.

## Joint Review Pilot

The 35 claim-found rows were sent through full claim/evidence/comment reconstruction:

- action: 16 promote, 7 silver, 6 rerun_evidence, 6 drop_no_reconstructable_claim
- label: 15 positive / 20 negative
- claim_found after joint review: 29/35
- product_evidence_found: 32/35

Stateful v2 promotion:

- reviewed rows: 35
- observed supervised rows: 18
- contrastive-eligible rows: 6
- repair/unobserved rows: 17
- y mix: 15 positive / 3 negative / 17 unobserved

Audit:

- flagged rows: 13/35
- high-severity flags: 0
- main flags: mixed comment relation, promote-not-main-ready, a few evidence gaps

## Important Finding

The pilot exposed a schema issue consistent with the proposal audit: some attributes such as `购买意图`, `视频内容`, and very broad `产品` are not stable product/service attributes in the sense required by `Methodology_Data.md`. These rows should not be deleted to inflate metrics; instead, they should be routed to schema review or kept as low-reliability/silver states unless the claim, evidence, and comments form a concrete product-attribute proposition.

## Decision

Expand pair-targeted claim re-extraction on the next1000 queue, but do not directly promote all recovered claims. Every batch must pass through:

1. exact SRT substring grounding,
2. joint claim-evidence-comment review,
3. local full-pair audit,
4. stateful v2 role assignment,
5. schema drift review for broad/evaluative attributes.

This preserves the proposal logic and avoids turning data repair into hard-sample deletion.
