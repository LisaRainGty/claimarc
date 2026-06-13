# Expansion Candidate Pools v4

## Summary
- `claim_reextract`: `3404` rows; priority={'P0': 1559, 'P1': 1845}; scope={'objective_product_attribute': 630, 'uncertain_attribute': 2243, 'process_or_evaluation_noise': 530, 'mixed_needs_remap': 1}
- `claim_verify_queue`: `629` rows; priority={'P0': 619, 'P1': 10}; scope={}
- `evidence_recovery`: `300` rows; priority={'P1': 115, 'P2': 185}; scope={'objective_product_attribute': 65, 'uncertain_attribute': 209, 'process_or_evaluation_noise': 26}
- `schema_remap`: `10652` rows; priority={'P0': 2038, 'P1': 2308, 'P2': 6306}; scope={'mixed_needs_remap': 424, 'process_or_evaluation_noise': 3672, 'uncertain_attribute': 5224, 'objective_product_attribute': 1332}
- `auxiliary_or_noise`: `244` rows; priority={}; scope={}

## Recommended Order
1. Verify top claim-reextract rows that are objective or mixed-remap and have negative trigger comments.
2. Verify remaining evidence-recovery rows not covered by strict P0 source recovery.
3. Use schema-remap rows to contract A_cmt(p) before a full Stage B/C rebuild.
4. Keep aux/noise rows outside clean supervision; use them only for contrastive diagnostics or appendix analysis.
