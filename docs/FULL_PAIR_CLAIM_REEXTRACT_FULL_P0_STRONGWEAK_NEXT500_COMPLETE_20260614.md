# Full Pair Claim Re-Extract Complete: next500

This report summarizes the completed pair-targeted SRT claim re-extraction for the `full_p0_strongweak_next500` repair queue. It is a recall layer only; all labels still pass through the full claim-evidence-comment reviewer.

## Inputs

- repair queue: `data/final/repaired_v1/full_pair_claim_repair_queue_full_p0_strongweak_next500_v1_20260614.jsonl`
- claim re-extract output: `data/final/repaired_v1/full_pair_claim_reextract_full_p0_strongweak_next500_v1_20260614.jsonl`
- model: `Qwen-Flash`

## Summary

- processed pair rows: `500`
- unique pair ids: `500`
- claim found: `429`
- no claim found: `71`
- error rows: `0`
- claim count buckets: `{"0": 71, "1": 137, "2-4": 144, "5-10": 93, "11+": 55}`

## Pipeline Note

The completed run confirms that many previously claim-missing pairs were recall failures rather than genuinely claim-absent cases. The next proposal-faithful gate remains the full-pair reviewer, which rebuilds the final consumer-perception label from recovered SRT claim, product evidence, and aligned consumer comments.

