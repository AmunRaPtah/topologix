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
