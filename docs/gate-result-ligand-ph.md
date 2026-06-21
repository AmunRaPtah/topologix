# Signed gate result — ligand-only PH (track B) on hERG

**Date:** 2026-06-21  **Split:** TDC `admet_group` hERG, official scaffold split
**Aligned rows:** train 518 / test 132 (97 positive) — intersection of descriptor-valid
and conformer-valid molecules, identical across all three tracks.
**Config:** VR PH maxdim=1 (H0+H1), 8 Å filtration window, persistence images + H0
death histogram + summary stats (B = 88 dims); RDKit 2D descriptors (A = 217 dims);
XGBoost, seed 0. Raw numbers: `benchmarks/results/ligand_ph_gate.json`.

## Tracks
| track | AUROC | AUPRC | MCC | BACC | ACC |
|---|---|---|---|---|---|
| A — RDKit 2D descriptors | **0.8367** | 0.9239 | **0.5752** | 0.7730 | 0.8409 |
| B — ligand-only PH | 0.7124 | 0.8605 | 0.2613 | 0.6306 | 0.7121 |
| AB — A ⊕ B (descriptor + topology booster) | 0.8356 | 0.9255 | 0.5603 | 0.7365 | 0.8409 |

## Gate (paired bootstrap, PASS iff 95% CI on delta excludes 0)
| comparison | metric | delta | 95% CI | verdict |
|---|---|---|---|---|
| B vs A | AUROC | −0.1243 | [−0.2167, −0.0292] | **FAIL** |
| AB vs A | AUROC | −0.0010 | [−0.0293, +0.0252] | **FAIL** (within noise) |
| B vs A | MCC | −0.3139 | [−0.5254, −0.1076] | **FAIL** |
| AB vs A | MCC | −0.0149 | [−0.1499, +0.1067] | **FAIL** (within noise) |

## Verdict — NO PULSE (for this formulation)
Vanilla ligand-only 3D persistent homology does **not** beat the descriptor baseline,
and adds **no orthogonal signal** when concatenated (AB ≈ A, CIs straddle 0). The pre-
registered Phase-3 criteria (i) and (ii) both fail on both AUROC and MCC.

This is the expected-cheapest kill experiment, run before any protein/docking work. It
confirms the research prior: cheap descriptors are hard to beat on hERG, and a molecule's
*own* topology carries little that the descriptor block doesn't already encode.

## Scope — what this does and does NOT kill
Killed: **vanilla all-atom ligand-only VR-PH** as a hERG booster.
NOT yet tested, and where the real IP lives:
1. **Element-specific PH channels** (ESPH, Cang & Wei) on the ligand — all-atom PH is
   known-weak; element-pair channels are the formulation with literature support. Cheap
   to add (re-run B with per-element-pair point clouds).
2. **Track C — interface-restricted bipartite PH** (the actual novelty claim): needs hERG
   cryo-EM pocket (5VA1/7CN1) + docked poses. Expensive (structure + Vina). The marginal
   value would have to come entirely from protein-interface geometry, since the ligand's
   own topology just showed none.

Per BUILD_PLAN honesty rail #5, this failed gate is a logged result that redirects the
build — not a setback to paper over.
