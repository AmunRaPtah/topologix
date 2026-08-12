# Pre-registration: is the negative result specific to Vietoris-Rips persistence images?

**Written before any featurization or model fitting.** Committed before the first run.
Experiment 7, hypothesis `H1.4`, problem `P1`.

## Why

The strongest available objection to the six experiments already reported is that their
conclusion is about one topological pipeline, not about topology: **Vietoris-Rips filtration,
homology, persistence-image vectorization**. A reviewer can reasonably say the result would
differ under a different filtration, a different algebraic object, or a different
vectorization, and nothing in the existing work answers that.

This experiment varies all three axes on the endpoint where the baseline is strongest and
best characterised, so a failure of the baseline cannot confound the reading (the flaw that
made Experiment 5's first attempt uninformative).

## Data and split

hERG_Karim, 13,445 molecules, balanced (positive rate 0.500). The **full** set is used. No
subsampling: V-002 established on this exact board that an underpowered split produces a
result that reverses on replication, and a 20-fold larger sample is what killed it.

3D conformers generated with RDKit ETKDGv3, fixed seed 0, falling back to random coordinates
on embedding failure. One conformer per molecule. Molecules that fail to embed are dropped
and counted.

Bemis-Murcko scaffold split, deterministic: scaffold groups sorted by size descending,
assigned to train until 80% of molecules are allocated, remainder to test. Target ratio
matches Experiment 2's powered replication (train 10,688 / test 2,679).

## Feature blocks

Baseline **A**: 217 RDKit 2D descriptors, identical to Experiments 1, 2 and 4.

All topological blocks use the **same seven element channels as Experiment 2**
({C}, {N}, {O}, {C,N}, {C,O}, {N,O}, all-heavy), so the comparison is like-for-like with the
ESPH result that motivated this paper.

| Block | Filtration | Algebraic object | Vectorization | Axis varied |
|---|---|---|---|---|
| **V** (control) | Vietoris-Rips | homology | persistence image | none, replication |
| **X** | **alpha complex** | homology | persistence image | filtration |
| **L** | Vietoris-Rips | homology | **persistence landscape** | vectorization |
| **S** | epsilon-graph sweep | **persistent Laplacian spectrum** | eigenvalue statistics | algebraic object |
| **U** | union of X, L, S | | | all three |

**S is dimension-0 only.** At each of 16 filtration values the epsilon-neighbourhood graph's
combinatorial Laplacian L0 = D − A is formed and summarised by its zero-eigenvalue
multiplicity (which equals the Betti-0 number, so homology is nested inside), its smallest
non-zero eigenvalue, largest eigenvalue, mean, sum, and spectral gap. The non-harmonic
spectrum is precisely the information homology discards, which is what makes this a test of
the algebraic object rather than of the filtration. Higher-dimensional persistent Laplacians
(L1, L2) are **not** computed, and the conclusion will be scoped accordingly.

Classifier: XGBoost, fixed seed, identical hyperparameters across all blocks.

## Decision rule, fixed now

**Family.** Eight pre-registered comparisons: each of X, L, S, U, tested both alone against A
and concatenated with A against A. The control block V is **not** in the family; it is a
replication check, not a test.

**Multiplicity correction.** With eight comparisons at a nominal 95% level, the probability
of at least one spurious exclusion of zero is substantial. **Holm-Bonferroni** correction is
applied across the family at family-wise alpha = 0.05. A comparison PASSES only if its
corrected interval excludes zero in the direction favouring topology. Uncorrected intervals
are reported alongside so the correction's effect is visible.

**Resampling unit.** Cluster bootstrap over **scaffolds**, 2,000 resamples, per invariant I-2.
Molecules sharing a Bemis-Murcko scaffold are not independent. The row-level bootstrap is
reported beside it; where they disagree the cluster version governs. This is not a new
decision, it is the rule that decided V-030.

**Secondary metric, non-decisive, per invariant I-5.** MCC reported for every comparison.
V-002's pulse was visible in AUROC and absent in MCC at the same sample size; if AUROC and
MCC disagree here the result is recorded as a metric artifact and the primary stands.

**Validity precondition.** Block A alone must reach AUROC within 0.03 of Experiment 2's
0.878 on this split. If it does not, the split is not comparable to prior experiments, and
the run is reported as a split-construction failure rather than as evidence about topology.

## Outcomes, decided in advance

- **No block passes.** The negative result generalises across filtration, algebraic object
  and vectorization. The reviewer objection is answered and the paper's scope claim
  strengthens from "Vietoris-Rips persistence images" to "four encodings spanning three axes".
- **One or more blocks pass.** The negative result is an artifact of the encoding, not a
  property of topology on this endpoint. This would be the most consequential outcome on the
  board: it would partially reopen P1, require the manuscript's central claim to be narrowed
  to the specific encoding, and demand that invariant I-7 be withdrawn. It would be reported
  as prominently as the negatives.
- **A block passes uncorrected but not after Holm-Bonferroni.** Recorded as not passing, and
  reported explicitly as the multiplicity trap the correction exists to catch. This is the
  V-002 shape and is the single most likely way to be fooled here.
- **Block A fails the validity precondition.** Split-construction failure. No claim about
  topology either way.

## Compute

CPU only, 4 vCPU. Conformer generation approximately 40 minutes for 13,445 molecules;
topological featurization across seven channels and four encodings approximately 40 minutes.

---

## Addendum, logged before any featurization completed or any score was seen

Two changes, both forced by profiling rather than by results.

**1. Conformers are embedded without explicit hydrogens.** The pre-registration said
"RDKit ETKDGv3, fixed seed 0". Profiling on 25 randomly drawn molecules from this dataset
showed `AddHs` then embed costs 2,950 ms/mol, which is 11 hours serially for 13,445
molecules and not runnable here. Embedding the heavy-atom graph directly costs 67-71 ms/mol,
a 40-fold speedup, with a *higher* success rate in the probe (25/25 vs 24/25). Every encoder
in this experiment consumes heavy atoms only, and the pipeline called `RemoveHs` immediately
after embedding in any case, so no information used downstream is lost. The cost is that
heavy-atom geometry is no longer relaxed against explicit hydrogen sterics. This is recorded
as a deviation because it is one, and it applies identically to all five feature blocks, so
it cannot favour any block over another.

**2. A bug was found and fixed before any run.** The new encoders passed raw (n, 3) point
clouds to `ripser`. When an element channel contains exactly three atoms the array is 3x3
and square, and ripser cannot distinguish a point cloud from a distance matrix; it emitted a
warning to that effect. All Vietoris-Rips calls now pass an explicit distance matrix with
`distance_matrix=True`. **Experiments 1-6 are unaffected**: their code passes distance
matrices explicitly, which was checked rather than assumed.

Featurization is parallelised across 4 worker processes with per-chunk checkpointing. This
changes wall-clock only; the computation, the seed, and the decision rule are unchanged.
