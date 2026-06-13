# Literature Notes: Stage-C-Aware RACL Repair

Date: 2026-06-13

Purpose: translate recent fact-checking / multimodal retrieval / e-commerce
attribute-mining work into concrete design constraints for CLAIMARC data repair
and model tuning.

## Sources Checked

- RAFTS, ACL 2024:
  <https://aclanthology.org/2024.acl-long.556/>
- FIRE, Findings of NAACL 2025:
  <https://aclanthology.org/2025.findings-naacl.158/>
- CONFACT / conflicting evidence for RAG fact-checking, IJCAI 2025:
  <https://www.ijcai.org/proceedings/2025/1073>
- DEFAME, dynamic multimodal fact-checking:
  <https://arxiv.org/abs/2412.10510>
- ACL 2025 accepted paper list, e-commerce and multimodal directions:
  <https://2025.aclweb.org/program/main_papers/>
- EMNLP 2025 Industry accepted paper list, fine-grained advertisement
  violation detection:
  <https://2025.emnlp.org/program/ind_papers/>

## Design Takeaways

1. Retrieval is not verification.
   RAFTS supports using evidence-conditioned supporting/refuting arguments, but
   the current CLAIMARC diagnostics show that generated arguments only help when
   product evidence is already sufficiently aligned.  Arguments should therefore
   be auxiliary views, not replacements for Stage C evidence repair.

2. Evidence sufficiency is source- and conflict-aware.
   CONFACT reports that RAG fact-checking degrades under conflicting evidence and
   benefits from explicit source credibility/background signals.  For CLAIMARC,
   params/OCR/VLM should remain separate sources with coverage/source-type
   features.  The VLM rerun confirms this empirically: coverage improved sharply,
   but ungated full-data learnability did not.

3. Multimodal verification should be dynamic and tool-like.
   DEFAME uses modular multimodal experts and dynamically selects tools/search
   depth.  CLAIMARC should mirror this in a lightweight way: numeric/identity
   attributes use params/title/OCR first; visual/color/material attributes use
   OCR/VLM; low-source or conflicting evidence gets downweighted or routed to a
   sourceful auxiliary view.

4. Fine-grained advertisement violation detection is moving toward localized
   claims and active/reasoning-style violation localization.
   EMNLP 2025 Industry lists RAVEN++ for fine-grained advertisement video
   violations.  This reinforces the move from broad `(product, attribute)` pairs
   to atomic claim-level evidence and consumer disagreement signals.

5. E-commerce attribute research now emphasizes open-world multimodal attribute
   mining and self-correction.
   ACL 2025 lists work on open-world attribute mining for e-commerce products
   with multimodal self-correction.  CLAIMARC's product-v2 schema, exact
   source-grounded claim extraction, and Stage C reruns are aligned with that
   direction; however, the final paper should present these as controlled data
   states rather than opaque LLM relabeling.

## Concrete CLAIMARC Decisions

- Keep the core method as retrieval-augmented contrastive learning.
- Add atomic auxiliary supervision only as train-only data with
  room/product/pair leakage guards.
- Use `sourceful_cov1` over the repaired params/title + OCR + VLM evidence as
  the next auxiliary candidate:
  `dataset_atomic_productv2_direct_strict_full_strictv2_refined_hp_paramtitle_ocrblock_vlmatomic_sourceful_cov1.jsonl`.
- Treat the full VLM-augmented dataset as an evidence-recall view; do not use
  ungated VLM evidence as the default full dataset because diagnostics show
  coverage gain without full-data learnability gain.
- Report coverage, source type, and conflict/consumer-signal states separately
  in the paper's data section.

## 2026-06-13 Addendum: Keep the Agent Offline and the Verifier Simple

Additional recent fact-verification work points in a consistent direction:

- FIRE-style iterative retrieval/verification improves evidence acquisition, but
  the verification target still needs clean claim-evidence alignment.
- RAFTS-style contrastive support/refute arguments are useful when arguments are
  grounded in retrieved evidence; if retrieval/evidence is noisy, generated
  rationales amplify the noise.
- Long-context RAG systems can be competitive in shared fact-checking pipelines,
  but they do not solve label construction or consumer-perception alignment.

Implication for CLAIMARC:

- The "agent" component should be presented as an offline data-reconstruction
  and evidence-QA agent, not as a heavy inference-time multi-agent system.
- The publishable model should stay compact: attribute-conditioned claim/evidence
  dual flow, source-aware evidence coverage, and attribute-blocked
  retrieval-augmented contrastive learning.
- The newest data changes (`hardclean v1`, `P0 hp3`) should be described as
  measurement-quality improvements: transaction shortcut removal, title-only
  evidence downgrading, and explicit expectation-gap recovery.

## 2026-06-13 Addendum: Upstream Rebuild and Minimal Agent Framing

Sources rechecked on 2026-06-13:

- RAFTS (ACL 2024): https://aclanthology.org/2024.acl-long.556/
- FIRE (Findings of NAACL 2025): https://aclanthology.org/2025.findings-naacl.158/
- Conflicting evidence in retrieval-augmented LLM fact-checking (IJCAI 2025):
  https://www.ijcai.org/proceedings/2025/1073
- LiveAMR for e-commerce live-streaming regulation (NAACL Industry 2025):
  https://aclanthology.org/2025.naacl-industry.32/
- EMNLP 2025 Industry program, including fine-grained advertisement violation
  detection directions: https://2025.emnlp.org/program/ind_papers/

Implications for the current candidate rebuild:

- The task should be framed as attribute-grounded consumer-facing fact
  verification, not generic sentiment classification.  LiveAMR supports the
  premise that live-commerce regulation needs domain-specific claim
  reconstruction and LLM-assisted data generation, but our contribution should
  focus on product claim/evidence/consumer disagreement rather than morph
  normalization.
- FIRE motivates an iterative retrieval/checking agent, but the publishable
  CLAIMARC model should not become an inference-time agent.  Use the agent as
  an offline data reconstruction and evidence QA module; keep the deployed
  architecture as retrieval-augmented contrastive learning.
- RAFTS motivates support/refute views, but they should be generated only from
  aligned product evidence.  In noisy rows, arguments amplify label noise, so
  the latest hard-clean and upstream claim-veto steps are prerequisites.
- The IJCAI 2025 conflicting-evidence result supports keeping params, title,
  OCR, and VLM evidence as separate source channels and building hard negatives
  from source conflicts rather than simply concatenating all evidence.

Concrete next model tweak after the data line finishes:

- Keep the current RACL backbone.
- Add source-aware hard-negative sampling: negatives are sampled preferentially
  from the same attribute family and same evidence-source/confidence bucket.
- Add a very small source-consistency auxiliary head only if hardclean CV shows
  that the main model still loses to BGE-LR on AP; do not add a heavy multi-agent
  inference loop.

## 2026-06-13 Addendum: Data-First Expansion After Full Candidate Rerun

The completed candidate rebuild supports the same conclusion:

- More raw LLM-extracted claims are not automatically better.  The candidate
  full-rerun increases coverage but weakens AUROC/Macro-F1 when used directly,
  especially on visual/color attributes.
- The publishable framing should distinguish three states:
  1. clean benchmark rows;
  2. high-precision train-only auxiliary rows;
  3. recall-oriented candidate rows used for offline QA and ablation.
- Agentic LLM use is best justified as an offline evidence-QA/reconstruction
  loop: schema contraction, exact-quote recovery, evidence-source alignment,
  deterministic validation, and consumer-signal adjudication.

Method constraint for the next model sweep:

- Prefer one interpretable RACL modification at a time:
  - source-aware hard-negative filtering (`same_evtype_conf`); or
  - a small source/confidence prototype auxiliary.
- Do not combine hard negatives, prototype loss, evidence-view consistency, and
  BGE distillation in one run until each has an isolated ablation.  The paper
  needs clean causal attribution, not a bag of tricks.

## 2026-06-13 Addendum: 2025-2026 Evidence and Ad Moderation Check

Additional sources checked on 2026-06-13:

- Face the Facts! Evaluating RAG-based Pipelines for Professional Fact-Checking
  (INLG 2025): https://aclanthology.org/2025.inlg-main.50/
- Resolving Conflicting Evidence in Automated Fact-Checking (IJCAI 2025):
  https://www.ijcai.org/proceedings/2025/1073
- Retrieval-Augmented Generation with Conflicting Evidence (COLM 2025):
  https://openreview.net/forum?id=z1MHB2m3V9
- Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking (arXiv 2026):
  https://arxiv.org/html/2603.00267v1
- RAVEN++: Pinpointing Fine-Grained Violations in Advertisement Videos with
  Active Reinforcement Reasoning (EMNLP Industry 2025):
  https://aclanthology.org/2025.emnlp-industry.1/

Method implication after our hard-negative ablations:

- Recent work treats evidence reliability as a set-level/source-level property,
  especially under conflicting or incomplete evidence.  This supports
  source/confidence-aware prototype regularization and evidence sufficiency
  reporting more strongly than hard filtering of negative neighbors.
- RAVEN++ reinforces fine-grained violation localization, but its active RL stack
  would be too heavy for CLAIMARC.  The transferable idea is curriculum/active
  focus on difficult localized claims, which we can implement as offline data QA
  and boundary-set evaluation rather than inference-time RL.
- Multi-agent retrieval should remain an offline reconstruction protocol: use it
  to improve raw-data evidence coverage and adjudicate suspicious rows, while the
  submitted model remains an end-to-end attribute-grounded RACL verifier.

## 2026-06-13 Addendum: Fine-Grained Verification and Data Quality

Additional sources checked on 2026-06-13:

- FactLens: Benchmarking Fine-Grained Fact Verification (Findings ACL 2025):
  https://aclanthology.org/2025.findings-acl.929/
- LogiCoL: Logically-Informed Contrastive Learning for Set-based Dense
  Retrieval (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.608/
- Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?
  (ACL 2025): https://aclanthology.org/2025.acl-long.779/
- Structure Trumps Size: Rethinking Data Quality for LLM Reasoning (Findings
  EMNLP 2025): https://aclanthology.org/2025.findings-emnlp.616/

Implications for CLAIMARC:

- FactLens supports the paper's move from coarse product-level labels to
  fine-grained claim/evidence units.  For our data, the analogous quality
  criterion is whether a short live-stream claim is supported by an exact
  product evidence span and a specific consumer-response signal.
- LogiCoL supports constrained contrastive geometry.  In CLAIMARC, the useful
  constraint is not generic logic over entities but attribute-, source-, and
  label-conditioned neighbor sets.  This favors soft reliability constraints
  over brittle hard-negative filtering.
- External-validation work supports using independent tools/teachers to audit
  LLM-generated labels.  In this project, a fold-local text/BGE teacher should
  be used as a data-QA signal, not as the sole ground truth.
- Structure-over-size evidence argues against expanding the auxiliary set simply
  past 3,000 rows.  The expansion has to preserve positive boundary samples,
  source coverage, and per-attribute balance; otherwise the extra rows improve
  AP-like lexical separability while weakening Macro-F1 on consumer-risk labels.

Current design consequence:

- Keep `hardclean v1` as the trusted auxiliary anchor.
- Treat `hardclean + candidate hpmerge` and value-gated rows as recall pools.
- Admit additional rows only through structured filters: teacher agreement,
  source coverage, positive-boundary preservation, and per-attribute/label caps.

## 2026-06-13 Addendum: 2026 Retrieval Feedback and Ad Localization

Additional sources checked on 2026-06-13:

- KG-CRAFT: Knowledge Graph-based Contrastive Reasoning with LLMs for Claim
  Verification (EACL 2026): https://aclanthology.org/2026.eacl-long.302.pdf
- Test-time Corpus Feedback: From Retrieval to RAG (Findings EACL 2026):
  https://aclanthology.org/2026.findings-eacl.298.pdf
- RAVEN++: Pinpointing Fine-Grained Violations in Advertisement Videos with
  Active Reinforcement Reasoning (EMNLP Industry 2025):
  https://aclanthology.org/2025.emnlp-industry.1.pdf
- LiveAMR / Chinese Morph Resolution in E-commerce Live Streaming Scenarios
  (NAACL Industry 2025): https://aclanthology.org/2025.naacl-industry.32.pdf
- Face the Facts! Evaluating RAG-based Pipelines for Professional Fact-Checking
  (INLG 2025): https://aclanthology.org/2025.inlg-main.50.pdf

Implications after the current hardclean OOF diagnosis:

- 2026 retrieval-feedback work supports iterative evidence acquisition and
  verifier feedback, but the high-cost agent loop is best used offline for
  repairing claim/evidence coverage.  The submitted verifier should stay
  compact and deterministic at inference time.
- KG-style contrastive reasoning reinforces the value of structured neighbor
  constraints.  In CLAIMARC, the structure is attribute family, evidence source,
  exact value compatibility, and consumer-signal confidence rather than a
  general open-domain knowledge graph.
- RAVEN++ supports fine-grained violation localization and difficulty-aware
  curricula, but its RL stack is heavier than needed here.  A publishable
  transfer is a targeted repair/curriculum queue built from OOF failures:
  high-confidence false positives, exact-value rows, and source-specific
  failures.
- LiveAMR supports the domain premise that live-commerce false advertising
  needs domain-specific reconstruction of noisy live speech.  For CLAIMARC, the
  analogous contribution is claim span normalization plus raw product-evidence
  recovery, not generic sentiment or morph classification.
- Professional fact-checking RAG evaluations emphasize that verdict quality is
  bounded by evidence sufficiency.  This matches our OOF mechanism finding:
  CLAIMARC helps when two evidence sources are available, but fails on generic
  OCR/title-like evidence and exact-parameter compatibility cases.

Resulting next step:

- Run LLM/VLM review only on the mechanism-driven repair queue, withholding
  current labels and model predictions.
- Convert the review into deterministic data actions: bad claim span drop,
  exact-value contradiction/support repair, evidence recovery rerun, or
  consumer-signal review.
- Then rerun a small fold-0 screen before committing to a full grouped-CV run.

## 2026-06-13 Addendum: Utility-Aware Retrieval and Pair-Keyed QA

Additional sources checked on 2026-06-13:

- LogiCoL: Logically-Informed Contrastive Learning for Set-based Dense
  Retrieval (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.608/
- SCARLet / Training a Utility-based Retriever Through Shared Context
  Attribution for Retrieval-Augmented Language Models (EMNLP 2025):
  https://aclanthology.org/2025.emnlp-main.33/
- CoEvo: Coevolution of LLM and Retrieval Model for Domain-Specific
  Information Retrieval (EMNLP 2025):
  https://aclanthology.org/2025.emnlp-main.757/
- External validation tools for LLM-as-a-Judge annotation quality (ACL 2025):
  https://aclanthology.org/2025.acl-long.779/

Transferable ideas:

- LogiCoL shows that contrastive retrievers can be improved with soft structural
  constraints over result sets.  For CLAIMARC, the analogous constraints are:
  same attribute family, compatible/incompatible value relation, source family,
  and consumer-signal confidence.  This supports a small constrained RACL loss
  rather than a heavy new architecture.
- SCARLet argues that retriever supervision should optimize downstream utility,
  not only semantic relevance.  CLAIMARC's repair reviews can provide exactly
  such utility labels: `supports`, `contradicts`, `insufficient`, and
  `not_verifiable` product evidence.  These should become retrieval-positive,
  retrieval-negative, or ignore masks for contrastive training.
- CoEvo supports alternating between LLM-generated domain data and retriever
  updates.  In CLAIMARC, this should remain an offline loop:
  OOF diagnosis -> pair-aligned LLM/VLM review -> deterministic repair ->
  retriever/verifier retraining -> new OOF diagnosis.
- External-validation work reinforces that LLM labels need independent checks.
  Our implementation response is a three-gate protocol: schema audit, queue
  alignment audit, and fold-local model/teacher disagreement audit before any
  row enters training.

Immediate method consequence:

- Treat the pair-aligned mechanism queue as a utility-attribution pilot.  Each
  reviewed row should produce a structured product-evidence utility state,
  not a direct final label.
- The next publishable model tweak should be `RACL-U`: retrieval-augmented
  contrastive learning with utility masks derived from evidence relation and
  source confidence.  Keep it minimal:
  1. positive evidence views: exact/compatible supports for clean rows and
     contradictions for risk rows;
  2. masked negatives: same attribute family but incompatible value or
     insufficient evidence;
  3. ignore masks for bad claim spans and unverified raw-image-only evidence.
- The paper can present the agent as an offline utility-labeling and data-QA
  loop, while the verifier remains a compact end-to-end RACL model.

## 2026-06-13 Addendum: Calibration Gap After Mechanism Repair

Additional sources checked on 2026-06-13:

- Reliable Decision-Making via Calibration-Oriented Retrieval-Augmented
  Generation / CalibRAG (NeurIPS 2025):
  https://proceedings.neurips.cc/paper_files/paper/2025/file/7a03c2bf486e56d8a25e8d5bb72ff1a2-Paper-Conference.pdf
- Training a Utility-based Retriever Through Shared Context Attribution for
  Retrieval-Augmented Language Models / SCARLet (EMNLP 2025):
  https://aclanthology.org/2025.emnlp-main.33/

Observed CLAIMARC connection:

- The mechanism-repaired candidate improves ranking but not deployed
  classification: CLAIMARC v2 reaches AP 0.9182 / AUROC 0.9507 on fold 0, but
  Macro-F1 is 0.8688 under the saved threshold, below BGE-LR's 0.8892.
- OOF oracle analysis shows CLAIMARC v2 could reach Macro-F1 0.8921 with a
  different global threshold, while BGE-LR has a much smaller oracle gap.  This
  is exactly a retrieval-augmented calibration problem, not simply a failure of
  representation learning.

Method implication:

- CalibRAG motivates adding a compact forecasting/calibration function over
  claim, retrieved evidence, source count, evidence relation, and confidence.
  For CLAIMARC this should be a fold-local calibration head or reliability gate,
  not an LLM-in-the-loop inference module.
- SCARLet motivates using LLM/VLM review states as evidence utility labels.
  The key shift is from semantic relevance (`claim` looks similar to evidence)
  to downstream utility (`evidence` actually supports/contradicts the consumer
  risk decision).
- The next architecture should therefore be `RACL-U+C`: utility-masked
  contrastive retrieval plus a small evidence-conditioned calibration head.
  This remains simple enough for an ACL/EMNLP-style method section and directly
  explains the fold-0 failure mode.

Empirical update:

- Full400 mechanism review produced an explicit utility signal:
  support/contradiction rows are useful evidence, while insufficient,
  not-verifiable, generic-evidence, and bad-claim rows are low-utility for
  contrastive alignment.
- A minimal RACL-U pilot operationalized this without a new architecture:
  `cl_c_min=0.2` and `cl_neg_c_min=0.2` mask low-confidence rows from
  contrastive anchors and negatives.
- On the same softdropbad full400 v3 fold-0 screen, this improves CLAIMARC PCLS
  from AP 0.8759 / AUROC 0.9612 / Macro-F1 0.9155 to AP 0.9136 / AUROC 0.9675
  / Macro-F1 0.9304.  The saved-vs-oracle Macro-F1 gap falls to 0.0027.
- This is a compact publishable bridge between SCARLet-style utility retrieval
  and CalibRAG-style calibration: the offline LLM/VLM agent supplies utility
  attribution, while the submitted verifier remains a deterministic RACL model.

Follow-up empirical confirmation:

- `src/models/cv_oof_raclu_calibrate.py` implements the compact RACL-U+C
  diagnostic suggested above.  It is fold-safe stacked calibration: the
  calibrator is fitted on OOF predictions from all other folds and applied to
  the held-out fold.
- On `softdropbad full400 v3`, RACL-U+C reaches AP 0.8953 / AUROC 0.9678 /
  Macro-F1 0.9209 / wF1 0.8491, outperforming both CLAIMARC selectiveRKC
  (0.8819 / 0.9490 / 0.9076 / 0.8221) and BGE-LR
  (0.8914 / 0.9606 / 0.9006 / 0.8143).
- Against BGE-LR, paired bootstrap gives significant gains on AUROC
  (`+0.0071`, 95% CI `[0.0014, 0.0127]`), Macro-F1 (`+0.0204`,
  `[0.0089, 0.0319]`), and wF1 (`+0.0350`, `[0.0052, 0.0653]`).  AP is
  positive but not significant.
- The ablation is methodologically useful: score-only calibration proves
  CLAIMARC/BGE complementarity (Macro-F1 0.9230), while source-conditioned
  calibration improves AP to 0.9091 and wF1 to 0.8576.  This is exactly the
  reliability-calibration mechanism predicted by the literature scan.
