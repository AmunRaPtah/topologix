# Pre-registration: does element-specific interface PH add signal for ΔΔG regression?

**Written before any data was downloaded, any feature computed, or any score seen.**
Committed to git before the first run. Combination `C-11`, problem `P9`, hypothesis `H9.1`.

## Why this experiment

Invariant `I-3`, derived from four negative verdicts (V-001, V-003, V-005, V-006), states
that persistent homology is redundant with cheap descriptors for classification endpoints
that are not geometrically continuous, and proposes that the boundary condition is the
endpoint's *geometric sensitivity* rather than the presence of a protein-ligand interface.

Every failure on this board so far is a classifier (hERG blockade, resistance). PH's
established successes are affinity *regression* (ESPH in 10 of 26 D3R Grand Challenge
tasks). P9 is the untouched problem on the predicted-favourable side of that line.

This is a genuine test and not a rerun of the known D3R result. The established PH wins are
on **absolute** affinity across diverse complexes. This tests **relative** binding free
energy within congeneric series, where the ligands are nearly identical by construction and
the topological differences are correspondingly tiny. I-3 predicts success; the congeneric
structure predicts failure. Those pull in opposite directions, which is what makes it worth
running.

## Data

OpenFE `protein-ligand-benchmark`, 15 targets (cdk2, cdk8, cmet, eg5, hif2a, mcl1, p38,
pde2, pfkfb3, ptp1b, shp2, syk, thrombin, tnks2, tyk2). Per target: a prepared protein PDB,
aligned ligand poses in SDF, and experimental IC50/Ki with stated error.

This is deliberately the same benchmark family on which the free incumbent `INC-OPENFE` was
measured, so the context bar and the experiment share a substrate.

**Unit handling.** Experimental values are converted to ΔG via ΔG = RT·ln(value in molar),
R = 0.0019872 kcal/mol/K, T = 297 K. Mixed measurement types (ic50, ki, pic50) are converted
per record from the declared `type` and `unit` fields; any record whose type or unit is not
handled is dropped and the count reported.

**Pair construction.** Each unordered ligand pair within a target appears exactly once, in
canonical order by ligand name. Label is ΔΔG = ΔG(B) − ΔG(A). Feature blocks are (B − A)
differences, which are antisymmetric under swap by construction. No sign augmentation.
Secondary analysis repeats on the published `03_edges` perturbation set only.

## Feature blocks

- **S (cheap baseline, the incumbent class):** RDKit 2D descriptor differences between the
  two ligands. This is the same descriptor family that beat topology on hERG (V-001, V-003),
  used here as the thing to beat.
- **T (topology):** element-specific bipartite opposition-distance interface PH per ligand
  pose, differenced between the pair. Reuses `esph_interface.py` from the Topologix
  benchmark unchanged in its geometry: 8 Å interface cut, 16 Å filtration cap, fixed-range
  persistence imager (birth/persistence 0-16 Å, 4×4 grid) with range fixed before fitting,
  six element-pair channels.
- **T+S:** concatenation.

## Split

`GroupKFold` grouped by **target**. Non-negotiable per invariant `I-2`: this is the exact
axis on which P23's 0.804 collapsed to 0.545, and a random split here would let the model
memorise the protein system, since every pair within a target shares one protein.

The dataset declares two grouping axes (target, and the congeneric series within a target).
Series is nested inside target, so grouping on target covers both; this is recorded as a
waiver rather than silently ignored.

## Decision rule, fixed now

**PRIMARY (decides PASS/FAIL).** PASS iff a paired bootstrap (2,000 resamples) over
held-out predictions gives a 95% CI on ΔRMSE that excludes zero in the direction favouring
topology, for **T vs S** or for **T+S vs S**.

**SECONDARY (reported, not decisive, per invariant `I-5`).** Spearman ρ on the same
out-of-fold predictions, with its own paired-bootstrap CI. Reported because the OpenFE
assessment found ranking statistics much closer between methods than error statistics, so
RMSE and ranking can disagree here. This is the ΔΔG analogue of the AUROC-versus-MCC
divergence that produced the V-002 pulse. **If the two metrics disagree, the result is
recorded as a metric artifact and the primary stands.**

**NULL REFERENCE.** A constant predictor (ΔΔG = 0) is scored alongside. Any model that does
not beat the constant predictor is reported as such regardless of the paired comparison.

**CONTEXT BAR, EXPLICITLY NOT THE GATE.** OpenFE reaches 1.73 kcal/mol weighted RMSE on the
public set and Schrödinger FEP+ roughly 0.86 on a buried-water subset. A fast ML model is
not expected to reach these, and reaching them is not required to pass. They define whether
the result is *useful*, which is a separate question from whether topology *adds signal*.
Both will be stated separately in the verdict. Conflating them would be exactly the
category error that made P1 look like a business.

## What each outcome means, decided in advance

- **T or T+S passes, and RMSE is near 1.73:** I-3 confirmed and P9 is a live venture question.
- **T or T+S passes, but RMSE is far above 1.73:** I-3 confirmed as science, no product.
  Publishable as the positive half of the negative-results manuscript. This is the most
  likely good outcome and must not be oversold.
- **Neither passes, and T+S ≥ S:** I-3 is wrong, or the congeneric structure defeats it.
  P9 orphaned for topology, invariant I-3 downgraded or restated.
- **Neither passes, and T+S < S (topology drags the baseline down):** the V-006 signature.
  I-3 is refuted in its current form; PH is redundant for this endpoint too, and the
  boundary is not geometric sensitivity. Topology closes across the whole board.

## Compute

CPU only, 4 vCPU. ESPH is computed per ligand pose, not per pair, so the cost is roughly
15 targets × ~20 ligands ≈ 300 PH computations, not thousands.
