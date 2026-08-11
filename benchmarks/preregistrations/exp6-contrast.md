# Pre-registration 2: the controlled contrast that tests invariant I-6

**Written after V-029/V-030/V-031 were recorded and committed, and before the contrast run.**
Hypothesis `H9.2`, problem `P9`, invariant under test `I-6`.

## What I-6 claims, and why it needs this

I-6 replaced I-3 after V-029/V-030 falsified it. I-3 said the boundary for persistent
homology was the *continuity of the endpoint*. I-6 says the boundary is **the size of the
topological difference relative to the size of the label difference**: PH adds value when
the compared structures differ substantially, and nothing when they are near-identical or
when the label is ligand-intrinsic.

That restatement currently rests entirely on re-reading five existing negatives plus one
external positive (D3R). Re-reading old results to fit a new law is the cheapest and least
trustworthy kind of evidence, and this board has already been burned once by a pattern that
looked real until it was tested properly (V-002). I-6 needs a prediction it could fail.

## The contrast

The same 370 ligands, the same S and T feature matrices already on disk, the same
RandomForest, the same GroupKFold by target. **Only the prediction target changes.**

- **Task A (done, V-029/V-030):** within-target congeneric ΔΔG. Compared ligands differ by
  one substituent. I-6 says near-identical, predicts FAIL. Observed: FAIL.
- **Task B (this run):** absolute ΔG per ligand, across 370 structurally diverse complexes
  spanning 15 unrelated proteins. This is the D3R/PDBbind regime where PH is externally
  established to work. I-6 predicts topology CONTRIBUTES here.

Nothing differs between A and B except whether the label is a difference or a level. If
topology helps in B and not in A, I-6 survives a test it could have failed. If it fails in
both, I-6 is wrong too, or these ESPH features are simply uninformative, and the honest
conclusion is that this portfolio's topology implementation carries no signal anywhere.

## Decision rule, fixed now

**PRIMARY.** PASS iff the target-level **cluster** bootstrap (2,000 resamples, resampling
targets not ligands) gives a 95% CI on ΔRMSE excluding zero in the direction favouring
topology, for T vs S or T+S vs S. The cluster bootstrap governs from the outset this time;
V-030 showed the pair-level version manufactures significance from grouping structure, and
that lesson is not being relearned.

**NULL REFERENCE.** Constant predictor (mean training ΔG). Reported alongside.

**SECONDARY, non-decisive (I-5).** Spearman ρ with its own cluster-bootstrap CI.

**PRE-DECLARED OUTCOMES.**

- *T or T+S passes in B, having failed in A:* I-6 confirmed by controlled contrast. It
  becomes the strongest law on the board and is worth the work to make enforceable.
- *Neither passes in B:* I-6 is not supported. Either the mechanism is wrong, or these ESPH
  features carry no signal in any regime. The second reading is testable and would be worse
  news, because it would mean the four earlier topology negatives are partly evidence about
  this implementation rather than about topology.
- *T alone fails but T+S passes:* topology is a weak complement, not a substitute. Report as
  a qualified pass with the effect size stated in kcal/mol, not as a headline.

## The diagnostic, computed alongside but not gating

I-6 will only become enforceable if "topological difference relative to label difference"
is a number you can compute before fitting anything. Defined here, reported for both tasks,
and **not** part of the decision rule on this run:

    R = median over compared entities of  ||T_b − T_a|| / median_{i,j} ||T_i − T_j||
        ────────────────────────────────────────────────────────────────────────────
        median over compared entities of  |y_b − y_a| / median_{i,j} |y_i − y_j|

R near 1 means the compared entities are as topologically distinct as any random pair,
relative to how distinct their labels are. R well below 1 means the comparison is
topologically degenerate: the structures being told apart are far more similar than the
labels they must explain. I-6 predicts low R for Task A and R near 1 for Task B.

Two calibration points do not make a threshold. They make it possible to state one honestly
as provisional, which is the most this run can support.
