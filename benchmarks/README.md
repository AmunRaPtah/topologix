# Benchmarks: the six pre-registered gates

Every experiment here had its decision rule fixed in writing before it ran. Raw output is in
`results/`, one JSON per gate, unedited. Where a result changed after a correction, both the
original and the corrected version are kept.

## Experiments 1-2: ligand-only PH on hERG

| File | Role |
|---|---|
| `baseline.py` | 217 RDKit 2D descriptors + XGBoost, the baseline to beat |
| `experiment_ligand_ph.py` | Vanilla all-atom ligand PH (Exp 1) and ESPH channels (Exp 2) |
| `harness.py` | Paired-bootstrap gate: PASS iff the 95% CI on ΔAUROC excludes zero |
| `results/ligand_ph_gate.json` | Exp 1: AUROC 0.712 vs 0.837, CI [-0.217, -0.029] |
| `results/esph_gate.json` | Exp 2 small split (n=132): 0.852 vs 0.837, CI [-0.041, +0.072] |
| `results/esph_gate_karim.json` | Exp 2 powered (n=2,679): 0.843 vs 0.878, CI [-0.046, -0.025] |

The small-split ESPH result is the methodologically important one. Taken at face value,
0.852 vs 0.837 reads as a positive finding. The CI straddled zero and MCC already favoured
the baseline, so the pre-registered response was replication rather than publication. On a
20× larger sample the effect reversed sign.

## Experiment 3: element-specific bipartite interface PH on drug resistance

| File | Role |
|---|---|
| `experiment_esph_interface.py` | Element channels + leakage-free vectorizer on the Platinum interface |
| `results/esph_interface_gate.json` | AUROC 0.485 vs cheap mCSM-style 0.582; concatenation 0.517 |

Concatenation scoring *below* the baseline alone is the sharpest available evidence that no
orthogonal signal exists: a genuinely present but underpowered signal would be expected to
preserve baseline performance, not reduce it.

## Experiments 4-6: ΔΔG and ΔG on the OpenFE protein-ligand-benchmark

Pre-registrations: `preregistrations/exp4-5-ddg.md`, `preregistrations/exp6-contrast.md`.

| File | Role |
|---|---|
| `download.py` | Fetches 15 targets: protein PDB, aligned ligand SDF, experimental values |
| `esph.py` | ESPH interface featurizer, geometry unchanged from Experiment 3 |
| `featurize.py` | Per-ligand descriptors (S) and topology (T); 370 ligands, ~50 s |
| `experiment_ddg_congeneric.py` | Exps 4-5: within-target ΔΔG, 5,084 pairs, held-out target |
| `experiment_ddg_contrast_heldout.py` | Exp 6 first attempt, held-out target only |
| `experiment_ddg_contrast.py` | Exp 6 corrected: both split regimes on identical rows |

Results: `ddg_congeneric_gate.json`, `ddg_contrast_heldout.json`,
`ddg_contrast_bothsplits.json`.

**The geometry in `esph.py` is carried over from Experiment 3 unchanged** (8 Å interface
cut, 16 Å filtration cap, fixed-range persistence imager with its range fixed before
fitting, six element-pair channels). This is deliberate: only the endpoint changes between
Experiment 3 and Experiments 4-6, so a different outcome could not be attributed to a
re-tuned featurizer.

### Two corrections made in the open

**The resampling unit (Exps 4-5).** The pre-registration said "paired bootstrap over
held-out predictions" without naming the unit. Pairs within a target share a protein and a
congeneric series and are not independent. Both bootstraps are computed and reported; the
cluster version was declared governing in the script docstring before the run. Read
literally, the pre-registration would have passed this gate on a +0.009 kcal/mol effect.

**The control regime (Exp 6).** The first version of the contrast used a held-out-target
split throughout, out of caution about grouping structure. That made it strictly harder than
the D3R/PDBbind regime whose published result motivated the whole hypothesis, and every
feature block fell below a constant predictor with negative rank correlation. When the
comparator collapses, the run says nothing about the marginal value of anything.
`ddg_contrast_heldout.json` is retained as the record of that failed control;
`ddg_contrast_bothsplits.json` supersedes it and reports both splits on identical rows.

Neither correction rescued the hypothesis. Both are documented because the discipline is the
point of this repository.

## Experiment 6: is the negative specific to Vietoris-Rips persistence images?

Pre-registration: `preregistrations/exp7-encodings.md`, including two addenda logged before
any score was seen.

| File | Role |
|---|---|
| `encodings_alt.py` | Four encoders varying filtration, vectorization and algebraic object |
| `featurize_encodings.py` | 3D conformers + descriptors + all four encodings, 13,434 molecules |
| `experiment_encodings.py` | Gate: 8 comparisons, Holm-Bonferroni, scaffold cluster bootstrap |
| `experiment_encodings_official.py` | Confirmatory rerun on TDC's split and Experiment 2's hyperparameters |
| `align_indices.py` | Recovers source-index alignment for the cached feature matrix |
| `results/encodings_gate_ownsplit.json` | First run. Precondition FAILED, see below |
| `results/encodings_gate_official.json` | **The result to cite.** Precondition passes |

**Cite the official-split run, not the first one.** The first run used a deterministic
largest-scaffold-group-first split, which pushed every singleton scaffold into test (2,687
scaffolds for 2,687 molecules) and made the descriptor baseline land at 0.8311 against
Experiment 2's 0.8781, outside the pre-registered ±0.03 tolerance. It also made the scaffold
cluster bootstrap degenerate into a row bootstrap, so a safeguard that is stated in the
method did nothing. Both runs are kept; the first is a valid within-split comparison and an
invalid cross-experiment one.

The confirmatory run uses TDC's own `create_scaffold_split` and the hyperparameters from
`baseline.py`. Baseline 0.8858 vs 0.8781, test positive rate 0.522 vs 0.521, 1,796 scaffolds
across 2,688 test molecules, so the cluster bootstrap is meaningful.

**What moved between the two runs.** `A+S` was +0.0050 on the first split, the only block
nominally above baseline, and is +0.0003 with CI [−0.0047, +0.0051] on the correct one. An
apparent effect that exists only under a mis-specified split is the same failure mode as the
ESPH pulse in Experiment 2, caught here before it was reported rather than after.

**A dependency changed the feature space mid-experiment.** Installing PyTDC to fetch the
official split silently downgraded rdkit, changing `Descriptors._descList` from 217 entries
to 210. Experiment 2 used 217. Had the confirmation run on 210 it would have compared a
different baseline while appearing to satisfy its own validity precondition, and nothing in
the pre-registration or the gate would have caught it. rdkit was restored and featurization
re-run, after which it reproduced the first run exactly at 13,434 kept / 11 dropped. The
lesson is to record library versions and feature-block dimensions alongside any cached
feature matrix and to treat a dimension change between runs as an error, not a curiosity.
