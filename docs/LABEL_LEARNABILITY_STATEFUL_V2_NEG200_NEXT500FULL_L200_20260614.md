# Label Learnability Check: Stateful v2 neg200 + next500full-l200

This is a lightweight data sanity check, not a paper baseline. It uses character TF-IDF plus logistic regression under room-grouped 5-fold OOF evaluation. Inputs are claim/evidence/category/attribute text only; LLM rationales are not used.

## Compared Views

- previous supervised view: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg80_next500full_l200_supervised_20260614.jsonl`
- current supervised view: `data/final/repaired_v1/stateful_proposal_dataset_v2_seed120_next200_neg200_next500full_l200_supervised_20260614.jsonl`

## Main Comparison

- previous neg80 view, seed 14: `AUPRC=0.8407`, `AUROC=0.6373`, `Macro-F1=0.5701`
- current neg200 view, seed 14: `AUPRC=0.8231`, `AUROC=0.6984`, `Macro-F1=0.6368`

## Current View Multi-Seed Check

- seed 1: `AUPRC=0.8125`, `AUROC=0.7088`, `Macro-F1=0.5967`
- seed 7: `AUPRC=0.8096`, `AUROC=0.7039`, `Macro-F1=0.6366`
- seed 21: `AUPRC=0.8273`, `AUROC=0.7028`, `Macro-F1=0.6354`

## Interpretation

Adding natural negative controls improved room-grouped AUROC and Macro-F1 while keeping scores in a credible range. This supports the current data direction: improve proposal-faithful claim/evidence/comment alignment and class balance, not artificial separability.

