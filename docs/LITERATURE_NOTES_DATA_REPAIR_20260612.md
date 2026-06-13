# Literature Notes for the Repaired CLAIMARC Direction

Date: 2026-06-12

## Most Relevant Recent Work

1. RAFTS, ACL 2024
   - Source: https://aclanthology.org/2024.acl-long.556/
   - Relevance: retrieval-augmented fact verification with synthesized
     supporting/refuting arguments. This supports keeping CLAIMARC's
     retrieval-augmented contrastive design, but using evidence-conditioned
     argument views as auxiliary inputs rather than treating LLM rationales as
     ground-truth labels.

2. Contrastive Fact-Checking Reranker, FEVER 2024
   - Source: https://aclanthology.org/2024.fever-1.28/
   - Relevance: contrastive training improves evidence retrieval for
     real-world fact checking, especially when relevant evidence is inferential
     rather than lexical. This supports our retained contrastive objective and
     motivates attribute/source-conditioned negatives instead of global random
     negatives.

3. Resolving Conflicting Evidence in Automated Fact-Checking, IJCAI 2025
   - Source: https://www.ijcai.org/proceedings/2025/1073
   - Relevance: RAG fact-checking degrades under conflicting evidence and
     benefits from modeling source credibility. This maps directly to our
     Stage C finding that params/OCR/VLM/source0 should not be collapsed into a
     naive coverage-confidence score. CLAIMARC should model source type and
     source reliability explicitly.

## Design Implications

- Main benchmark should use `dataset_attrpol_hq_product_v1.jsonl`:
  repaired consumer-perception labels, product-scope attributes, no room-level
  split leakage, and enough size for stable grouped CV.
- Objective verification should be an ablation:
  `dataset_objective_contradicted_product_v1.jsonl` is semantically cleaner but
  has only 269 positives.
- Keep the retrieval-enhanced contrastive learning mechanism, but make it
  source-aware:
  - positives/negatives should be sampled within attribute/source-confidence
    neighborhoods;
  - source auxiliary heads should predict evidence type and reliability;
  - inference should use selective retrieval correction only when validation
    supports it.
- Do not use evidence-missing states as positives in the main objective.
  `insufficient/not_verifiable` should become an uncertainty or stress-test
  setting, not a shortcut label.

