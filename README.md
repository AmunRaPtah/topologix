# TOPOLOGIX

Interface persistent homology for drug-protein property prediction: bipartite simplicial
complex at the drug-protein interface, opposition-distance metric, persistent homology
yielding Internuclear Persistence Contours (IPCs).

## Outcome

**The validation gate this repository was built around did not pass, on any endpoint
tested.** The repository is retained as the code and evidence record for a pre-registered
negative result. Nothing here should be read as a working screening product.

The gate, fixed before any experiment ran, was: *IPC-derived features must beat an
RDKit-descriptor baseline on a held-out split. If they do not, the core formulation is
wrong and the chain halts at validation.* Six experiments across four endpoints and two
split regimes all failed it.

| # | Endpoint | Split | Topology vs cheap baseline | Gate |
|---|---|---|---|---|
| 1 | hERG cardiotoxicity, vanilla ligand PH | scaffold | AUROC 0.712 vs 0.837 | FAIL |
| 2 | hERG cardiotoxicity, ESPH (n=132) | scaffold | 0.852 vs 0.837, CI straddles 0 | no pass |
| 2b | hERG cardiotoxicity, ESPH (n=2,679) | scaffold | 0.843 vs 0.878 | FAIL |
| 3 | Drug resistance, ESPH bipartite interface | GroupKFold, protein | 0.485 vs 0.582; T+S 0.517 | FAIL |
| 4 | Congeneric ΔΔG | GroupKFold, target | RMSE 1.550 vs 1.399 | FAIL |
| 5 | Congeneric ΔΔG, concatenation | GroupKFold, target | +0.009 kcal/mol, CI straddles 0 | FAIL |
| 6 | Absolute ΔG | random 5-fold | RMSE 1.094 vs 0.960; T+S adds +0.003 | FAIL |

Experiment 6 is the informative one. In the regime where the baseline works, topology alone
reaches Spearman **ρ = +0.690** and beats a constant predictor by 27%. **The features carry
real signal and the implementation is sound.** That signal is simply a subset of what cheap
2D descriptors already provide, in every regime tested.

Two candidate boundary conditions were proposed and both were then falsified by experiment
rather than argued away: that the boundary is endpoint continuity (killed by Experiments 4
and 5, where ΔΔG is continuous and topology still failed), and that it is structural
diversity of the compared complexes (killed by Experiment 6, which is that regime).

## A note on the motivating result

This project was started on the finding that element-specific PH models ranked first in
10 of 26 official tasks in **D3R Grand Challenge 3**. Checked against the primary reports
(Nguyen et al., *J Comput Aided Mol Des* 33(1):71-82, 2019; MathDL, 34(2):131-147, 2020):

- The models **integrate ESPH with multiscale weighted colored graph** features. MathDL
  mixes further, combining algebraic topology with differential geometry in one model and
  topology, geometry and algebraic graph in another.
- **Neither paper reports an ablation isolating the topological block**, so the share of
  performance attributable to PH alone is not established there.
- D3R is a blind community challenge: submissions are ranked against **other participants**,
  not against a fixed panel of published baselines.
- **Neither paper benchmarks against cheap 2D descriptors** (RDKit descriptor sets, ECFP).
  The comparison that would establish PH's marginal value over standard cheap practice was
  not run.

None of that is a criticism of those papers, which make no such claim. It is a caution about
how the result gets read downstream, including here. A version of the caveat was recorded in
this project's own literature review in June 2026 and was not applied to any downstream
decision. It should have been.

## Layout

| Path | Contents |
|---|---|
| `src/topologix/` | Library: complex construction, opposition metric, homology, features |
| `benchmarks/` | The six experiments, runnable, plus raw gate output as JSON |
| `benchmarks/preregistrations/` | Decision rules fixed in writing before each run |
| `benchmarks/results/` | Raw gate output, one JSON per experiment |
| `scripts/` | ESM-2 pipelines from the resistance follow-on work |

See `benchmarks/README.md` for what each experiment does and how to reproduce it.

## Reproducing

```
pip install -e ".[dev]"
pytest -q
```

Experiments 4-6 need their benchmark downloaded first:

```
python benchmarks/download.py           # OpenFE protein-ligand-benchmark, 15 targets
python benchmarks/featurize.py          # 370 ligands -> descriptors + ESPH vectors
python benchmarks/experiment_ddg_congeneric.py
python benchmarks/experiment_ddg_contrast.py
```

CPU only. Featurization takes about a minute; the congeneric gate takes roughly half an
hour on 4 vCPU.

## Method note on the statistics

Experiments 3-6 resample **clusters** (proteins or targets), not rows, when bootstrapping
confidence intervals. This is not cosmetic. In Experiment 5 the row-level bootstrap gives
95% CI [+0.0034, +0.0145] on a +0.009 kcal/mol effect, excluding zero and passing the gate;
the cluster bootstrap on the same predictions gives [-0.0070, +0.0291] and fails it. Rows
drawn from 15 protein systems are not 15 × n independent observations.
