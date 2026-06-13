# Literature-to-Method Alignment Notes

This note records the current design implications for CLAIMARC. It is not a
claim of final experimental readiness.

## Relevant Recent Work

- [RAFTS: Retrieval Augmented Fact Verification by Synthesizing Contrastive Arguments](https://aclanthology.org/2024.acl-long.556/) (ACL 2024) retrieves and reranks evidence, then synthesizes supporting/refuting arguments conditioned on evidence. CLAIMARC should keep the retrieval-augmented contrastive core, but instantiate arguments as claim-aligned product evidence and consumer support/refute comments.
- [FIRE: Fact-checking with Iterative Retrieval and Verification](https://aclanthology.org/2025.findings-naacl.158.pdf) (Findings of NAACL 2025) motivates selective/iterative retrieval: use extra retrieval when uncertainty remains, rather than always paying the full retrieval or multimodal cost. CLAIMARC should treat detail-image VLM evidence as a dynamic repair route, not as a mandatory component for every row.
- [FactLens: Benchmarking Fine-Grained Fact Verification](https://aclanthology.org/2025.findings-acl.929/) (Findings of ACL 2025) argues for fine-grained verification units to reduce ambiguity. CLAIMARC's `(product, attribute)` unit and minimal SRT claim extraction should be kept; broad product-quality or subjective comments should not overwrite attribute-level labels.
- EMNLP 2024 included work on multi-stage reranking for fact-verification evidence retrieval and unified active retrieval for RAG, reinforcing that retrieval quality and retrieval timing are central design choices rather than peripheral preprocessing.
- ACL 2025 lists REAL-MM-RAG, a real-world multimodal retrieval benchmark, which is aligned with CLAIMARC's need to combine product parameters, OCR, and detail images, but only after text evidence is insufficient.

## Design Commitments for CLAIMARC

- Data unit: keep `(product_id, attribute_id)` as the atomic supervision unit.
- Label rule: `y_perception=1` must come from an aligned consumer refutation of the repaired livestream claim. Product evidence contradiction alone is an audit signal, not a positive label.
- Retrieval: use claim-attribute aware evidence retrieval over title, params, OCR, and VLM captions. Use VLM only for missing/insufficient product evidence or when OCR/params cannot ground the attribute.
- Contrastive learning: contrast clean support/refute evidence only when claim, product evidence, and consumer relation are complete. Silver rows remain supervised CE examples with lower reliability, but are masked from hard contrastive losses.
- Negative controls: sample natural negatives from claim-present, evidence-present, old non-refuting rows and re-adjudicate them with the same full-pair reviewer, instead of manufacturing easy negatives.

## Immediate Implementation Changes

- Added `build_stateful_proposal_dataset_v2.py` to separate `y_perception`, `promotion_state`, `c_reliability`, and `contrastive_mask`.
- Added `audit_proposal_label_claim_evidence_consistency_v2.py` for recurring label/claim/evidence QA.
- Added `build_full_pair_negative_control_queue_v1.py` to recover natural negative controls under the same proposal logic.
