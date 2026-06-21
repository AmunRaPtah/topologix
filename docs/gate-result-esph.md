# Signed gate result — element-specific PH (ESPH, track B′) on hERG

**Date:** 2026-06-21  **Split:** TDC `admet_group` hERG, official scaffold split
**Aligned rows:** train 518 / test 132 (97 positive), identical across tracks.
**Config:** ESPH = 7 element channels {C},{N},{O},{C,N},{C,O},{N,O},all-heavy, each VR PH
maxdim=1 → 88-dim block, concat = 616 dims. A = RDKit 2D descriptors (217). XGBoost, seed 0.
Raw: `benchmarks/results/esph_gate.json`.

## Tracks
| track | AUROC | AUPRC | MCC | BACC | ACC |
|---|---|---|---|---|---|
| A — descriptors | 0.8367 | 0.9239 | **0.5752** | 0.7730 | 0.8409 |
| B′ — ESPH (ligand only, **no descriptors, no protein**) | **0.8523** | **0.9399** | 0.4682 | 0.6976 | 0.8106 |
| AB′ — descriptors ⊕ ESPH | 0.8458 | 0.9318 | 0.5224 | 0.7353 | 0.8258 |

## Gate (paired bootstrap, PASS iff 95% CI on delta excludes 0)
| comparison | metric | delta | 95% CI | p(method>A) | verdict |
|---|---|---|---|---|---|
| B′ vs A | AUROC | +0.0156 | [−0.0411, +0.0723] | 0.71 | **FAIL** (not significant) |
| AB′ vs A | AUROC | +0.0091 | [−0.0178, +0.0347] | 0.76 | **FAIL** (not significant) |
| B′ vs A | MCC | −0.1070 | [−0.2552, +0.0373] | 0.07 | **FAIL** |
| AB′ vs A | MCC | −0.0528 | [−0.1600, +0.0373] | 0.12 | **FAIL** |

## Verdict — PULSE, but gate still FAILS
Two things are simultaneously true and must both be reported (honesty rail #2):

1. **There is a real topological pulse.** Element-specific PH **alone** — pure 3D topology,
   *zero* cheminformatics descriptors, *no* protein — matches/edges the strong descriptor
   baseline on AUROC (0.852 vs 0.837) and AUPRC (0.940 vs 0.924). Vanilla all-atom PH (track B)
   scored 0.71; the element channels are what carry the signal, exactly as Cang & Wei predict.
   So topology is **not** dead for hERG — it is a competitive *independent* representation.

2. **The pre-registered product thesis still fails.** The locked decision judges the product on
   "descriptor ⊕ topology **beats** descriptor alone." It does not: AB′ ≈ A (CI straddles 0),
   and ESPH is **worse on MCC** (the imbalanced soft-spot metric we said would matter most).
   The two representations are largely redundant — ESPH re-encodes what descriptors already know,
   it does not add orthogonal signal.

## The binding caveat: the gate is underpowered
n_test = 132 → the AUROC bootstrap CI is ≈ ±0.06. A real edge of ~0.015 AUROC is **undetectable**
at this sample size. The TDC hERG split *cannot* settle a small-booster hypothesis either way.
The build plan already named the right venue: a **powered MCC test on an imbalanced HTS split**
(Sato/Karim, thousands of compounds) — currently blocked on the HTS data file (TODO).

## What this means for the build
- The cheap, on-target ligand experiments are exhausted: ligand topology is descriptor-parity at
  best, not a booster, on the data we can currently test.
- The genuine remaining IP — **interface-restricted bipartite PH (track C)** — is untested but now
  rests on a weaker prior: the ligand's own topology carries no information beyond descriptors, so
  any marginal product value must come *entirely* from protein-interface geometry.
- Decision is a go/no-go for the human: (a) source the powered HTS set and re-run the MCC gate,
  (b) commit to the expensive track-C interface build (structure + docking), or (c) pivot the
  endpoint to a binding-affinity task where interface PH is already proven.
