# Literature Notes for Full-Pair RACL Reset

This note maps recent fact-checking, RAG, multimodal verification, and consumer
deception work to the current CLAIMARC full-pair reconstruction plan.

## Sources Checked

- DACLR, "Checking Fact with Better Retrieval: Dynamic Contrastive Learning for
  Evidence Retrieval" (arXiv, 2026):
  https://arxiv.org/html/2605.27449v1
- FIRE, "Fact-checking with Iterative Retrieval and Verification" (Findings of
  NAACL 2025):
  https://aclanthology.org/2025.findings-naacl.158.pdf
- CONFACT, "Resolving Conflicting Evidence in Automated Fact-Checking"
  (IJCAI 2025):
  https://www.ijcai.org/proceedings/2025/1073
- MAFT, "Multimodal Automated Fact-Checking via Textualization" (AAAI 2025):
  https://ojs.aaai.org/index.php/AAAI/article/view/35354/37509
- RAFTS, "Retrieval Augmented Fact Verification by Synthesizing Contrastive
  Arguments" (2024):
  https://arxiv.org/html/2406.09815v1
- EMNLP 2025 accepted main-paper list, including "Retrieval-Augmented
  Generation with Estimation of Source Reliability":
  https://2025.emnlp.org/program/main_papers/
- Bothma, "The influence of perceived deceptive advertising on consumer
  behaviour in the online fashion environment" (SAJEMS):
  https://sajems.org/index.php/sajems/article/view/6398/3655

## Method Implications

1. Atomic claim grounding remains mandatory.

FIRE motivates treating atomic claim verification as the difficult central
problem rather than aggregating broad passages.  For CLAIMARC this means each
training row must keep a minimal target-attribute SRT claim, not a loose
livestream passage.  The full-pair reconstruction runner follows this by asking
for a minimal continuous claim span and by refusing positive labels when no
claim is found.

2. Multimodal detail evidence should be textualized, then source-tagged.

MAFT supports converting image/video/audio evidence into text before
verification.  CLAIMARC should keep title, params, detail OCR, and detail-image
VLM as separate textualized evidence channels.  The current queue and promotion
builder already preserve source type rather than concatenating everything into
one evidence string.

3. Source reliability is a model variable, not a post-hoc note.

CONFACT and the EMNLP 2025 source-reliability line both support explicit source
reliability modeling.  In CLAIMARC, source reliability should enter as:

- evidence source embeddings: title, params, OCR, VLM;
- source-count and source-agreement features;
- contrastive masks that avoid treating low-quality OCR snippets as hard gold
  evidence;
- calibration/mechanism tables by source family.

4. Dynamic contrastive retrieval should use reconstruction states.

DACLR supports dynamic adaptive contrastive learning for evidence retrieval.
The CLAIMARC analogue should not add a large extra agent.  Instead, use
promotion states from the full-pair reconstruction pipeline:

- positives: same pair claim-evidence/comment views with `main_positive_refute`
  or `main_negative_support`;
- hard negatives: same attribute family, same source-confidence bucket, opposite
  consumer relation;
- ignored pairs: `repair_missing_claim`, `repair_missing_evidence`, and
  `lowinfo_no_aligned_comment`;
- silver curriculum: `silver_refute_missing_product_evidence` only after product
  evidence is recovered.

5. Supporting/refuting argument generation is auxiliary only.

RAFTS shows that synthesized supporting/refuting arguments can help retrieval
augmented fact verification.  For CLAIMARC, generated arguments should be used
for explanation and error analysis, not as labels.  The label must still come
from consumer comments aligned to the same recovered streamer claim.

6. Consumer perception is a defensible target distinct from objective fact
checking.

Consumer-behavior work on perceived deceptive advertising links perceived
deception to satisfaction, trust, and negative reaction.  This supports the
proposal's target: platform risk should model whether consumers perceive a
streamer claim as misleading, while product evidence provides mechanism and
retrieval context.

## Resulting CLAIMARC Design Constraint

The publishable architecture should remain compact:

`attribute-conditioned claim encoder + source-tagged product-evidence encoder + consumer-relation supervision + retrieval-augmented contrastive loss`

Allowed refinements:

- source-reliability embeddings and source-family calibration;
- dynamic contrastive masks from full-pair promotion states;
- optional textualized VLM evidence channel;
- mechanism analysis over support/refute/evidence-source states.

Avoid:

- inference-time multi-agent systems;
- labels derived from product evidence contradiction alone;
- deleting hard rows to improve AUROC;
- concatenating all evidence sources without reliability/source identity.
