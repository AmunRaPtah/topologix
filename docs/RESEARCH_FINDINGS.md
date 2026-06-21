# TOPOLOGIX — research findings (de-risking, 2026-06-21)

Deep, adversarially-verified literature review (103-agent harness; 21 sources, 25 claims
verified, 22 confirmed / 3 refuted). Raw report: `docs/research-report-raw.json`. This file is
the decision-grade summary that the build plan is built on.

## TL;DR
- **The math operators are published, not novel.** Opposition distance + Vietoris–Rips + PH for
  bipartite protein–ligand matching is exactly PATH/PATH+ (2025), and opposition distance itself
  originates in Cang & Wei's TopologyNet/TNet-BP (2017). "Persistence contours/surfaces" =
  Adams et al. persistence images/surfaces. **Novelty must rest on the application** —
  interface-restricted bipartite PH for *toxicity (hERG)* — not on the constructs.
- **The central bet is at serious risk.** On hERG specifically, cheap 2D ligand features
  (fingerprints / RDKit descriptors + gradient boosting) are competitive-to-superior, and the one
  near-direct datapoint has topology *underperforming* descriptors. The decisive PH-vs-descriptor
  hERG experiment **has never been published** — it is exactly the gate this repo defines.
- **Numbers to beat:** TDC hERG (648, scaffold split): SOTA 0.880 AUROC; **RDKit2D+MLP = 0.841**.
  HTS-scale Sato set (291,219 cpds, ~28:1 imbalance): XGBoost ACC 0.87–0.90 but **MCC only ~0.4**.

## 1. TDA on molecules — proven, but on AFFINITY not toxicity (high confidence)
- ESPH (element-specific PH), TopologyNet (PLOS Comp Biol 2017), MathDL: SOTA on PDBbind binding
  affinity; won 10/26 D3R Grand Challenge tasks. **Caveat:** D3R wins were ESPH **+** Multiscale
  Weighted Colored Graphs hybrids, not pure PH; "SOTA" is vs 2017–18 baselines.
- So PH genuinely carries protein–ligand *interaction* signal — but the evidence base is
  binding affinity / pose, **not** a ligand-toxicity endpoint like hERG.

## 2. Grounding the three TOPOLOGIX constructs (high confidence)
| TOPOLOGIX term | Established equivalent | Source | Implication |
|---|---|---|---|
| opposition-distance metric | d_op: Euclidean across protein↔ligand, ∞ within affiliation, into a VR filtration | PATH/PATH+ PMC12226026; origin TopologyNet PLOS CB 2017 | reuse the published recipe; cite it; it is **not** our invention |
| bipartite simplicial complex at interface | VR complex on a cross-distance / bipartite point cloud (no source uses the literal phrase) | PATH 2025 | the *interface restriction* is the only arguably-new twist |
| Internuclear Persistence Contours | persistence image / persistent surface (2D matrix, CNN-able); ESPH binning is image-like | Adams et al. 2017; Pun/Xia/Lee arXiv:1811.00252 | IPC must add something beyond a renamed persistence image to be IP |

**Where the unclaimed space actually is:** interface-restricted **bipartite** PH applied to a
**toxicity** endpoint (hERG), and whatever the IPC vectorization does beyond a standard
persistence image. The operators are prior art — position the IP on application + formulation.

## 3. hERG benchmarks & the baseline to beat (high confidence)
- **TDC hERG** (`admet_group`): 648 drugs, binary, AUROC, scaffold split. SOTA = MapLight+GNN
  0.880; CFA 0.875; **RDKit2D+MLP 0.841** (within 0.04 of SOTA). ← primary gate target.
- **hERG_Karim**: 13,445 drugs, binary (<10 µM). **hERG Central**: 306,893 drugs, % inhibition
  @1/10 µM + binary.
- **Sato et al.**: 291,219 molecules (9,890 inhibitors, ~28–29:1), IC50≤10 µM. 22-descriptor
  XGBoost ensemble: ACC 0.87–0.90, SE 0.83 / SP 0.91, **MCC ~0.4** (≤0.74 on balanced subsets).
  → On imbalanced HTS data, **MCC is the soft underbelly**, not the saturated ACC/AUROC.

## 4. CENTRAL RISK — direct evidence topology may NOT beat descriptors on hERG (high confidence)
- **Feng & Wei topological-Laplacian** on the Sato/Ogura external test: BACC **0.75** (SE 0.51)
  vs Ogura's **descriptor SVM BACC 0.80** (SE 0.67) — topology *underperformed* (raw ACC/AUC
  near-tied; the gap is a sensitivity/imbalance effect).
- **CToxPred (JCIM 2024)** + PMC12756696: plain fingerprints beat physicochemical descriptors
  **and** graph-learned reps on hERG; a 10-feature decision tree nearly matched a full model.
- **Creanza et al. (JCIM 2021)**: best structure-based hERG classifier (5VA1 IFD/MD) AUC
  0.85–0.86, only "comparable" to ligand-based (0.93–0.95). **Structure aids interpretability,
  not accuracy.**
- Weak positive signal: TopoLearn (J Cheminf 2025) — PH descriptors of a feature space correlate
  with generalization error (r=0.62) — but on general property prediction, not hERG, and a
  non-topological sibling scored higher. Medium confidence at best.

**Refuted (excluded):** "hERG needs no structure at all" (0-3 — structure still helps
coverage/interpretability); "AUC is uninformative because it clusters 0.80–0.95" (1-2);
PATH+ RMSE superiority over TNet-BP (1-2).

## 5. Feasibility on CPU (verified empirically by us, not by the literature)
The literature section on tooling returned **no verified claim** — treat as open. BUT we ran the
spike directly: ripser 0.6.15 + gudhi 3.12.0 + persim on this 4-vCPU host compute VR H0/H1/H2 on a
60-point (drug+pocket-scale) cloud in **22 ms**; persistence-image vectorization works. CPU-TDA is
**not** the bottleneck. (Ripser fastest for VR H0/H1 on small clouds; GUDHI/giotto-tda for
vectorizations — to confirm at interface scale during Phase 0.)

## 6. The four open questions that the MVP must answer
1. The decisive experiment — interface/bipartite PH vs RDKit+XGBoost on a standard hERG split —
   **has never been run.** This IS the project's core de-risking experiment.
2. Real CPU cost of VR PH (H0/H1/H2) on a *combined drug + hERG-pocket interface* cloud (we have
   the small-scale number; confirm at interface scale).
3. Does interface-restriction **add** signal over whole-pocket ESPH, or **discard** it — for an
   endpoint (hERG) that may be predictable from the ligand alone? (Feature or bug?)
4. Can topology move **MCC** on the imbalanced set (where descriptors plateau at ~0.4) rather than
   the already-saturated ACC/AUROC? Pre-register that as the success criterion.
