# TOPOLOGIX — build plan (math → prototype → validated → MVP)

## LOCKED DECISION (2026-06-21) — product-first, topology-as-booster
The goal is a hERG screener that actually works and gets used, not a benchmark stunt. Research
says topology will not beat cheap descriptors on raw hERG accuracy, so:
- **Floor = a strong descriptor/fingerprint + gradient-boosting screener** (~0.84 AUROC, shippable
  on its own). The product cannot fail to exist.
- **Topology earns its place as an orthogonal booster**: the gate is "descriptor ⊕ IPC beats
  descriptor alone", judged on **MCC (imbalanced HTS) + novel-scaffold generalization**, not the
  saturated AUROC. Interface-PH also supplies the structural interpretability chemists trust.
- **IP**: interface-restricted bipartite PH for *toxicity* is the genuinely novel claim; it stands
  or falls on the booster gate. Affinity-task (PDBbind) validation is the reserve pivot if hERG
  topology shows no pulse at all.

Evidence basis: `docs/RESEARCH_FINDINGS.md`. Governing principle, taken from the research and the
repo's own design: **front-load the one experiment that can kill the project.** The literature
says cheap descriptors are hard to beat on hERG and the decisive PH-vs-descriptor hERG experiment
has never been run. So we do not build the elaborate framework and *then* test it — we run the
cheapest honest version of the gate as early as possible, and only invest in depth once topology
shows a pulse. This matches the repo's "validation gate halts the chain" philosophy.

Two work types (from README): **Derivation** (math core; acceptance = constraint set + the
validation gate) and **Engineering** (known specs; acceptance = tests pass).

---

## Phase 0 — Foundations & the baseline to beat  [Engineering]  ✓ DONE 2026-06-21
Goal: a green toolchain, the data, and the **exact descriptor number the gate must beat**.
- [x] Env + CPU-TDA feasibility spike — ripser/gudhi/persim install; VR H0/H1/H2 in 22 ms. ✓
- [x] `data`: TDC hERG loader (admet_group, official scaffold split: 523 train / 132 test) ✓.
      Sato/Karim HTS CSV loader present (`load_smiles_csv`); HTS data file still TODO.
- [x] `benchmarks`: RDKit 2D descriptors (~200) + XGBoost baseline. **TDC hERG scaffold split:
      AUROC 0.8075, AUPRC 0.9057, MCC 0.5752, BACC 0.7730, ACC 0.8409.** This is the line in the
      sand. NB: ran 0.8075 vs the ~0.84 lit target (lit = RDKit2D+MLP); honest floor, not tuned.
- [x] `run_gate` implemented: AUROC + MCC, paired bootstrap CI on the delta (PASS iff CI_lo>0).
      Smoke-tested on dummy inputs — strong signal PASSes, noise-level delta FAILs, MCC works.
Exit: `pytest` green ✓; baseline metrics logged ✓; gate runs on dummy inputs ✓.
Note: editable install upgraded rdkit→2026.3.x, sklearn→1.9 (matches pyproject pins; pytdc's
older pins bind only at resolution, runs fine). setuptools<81 needed for pytdc's pkg_resources.

## Phase 1 — Math framework  [Derivation]
Goal: rigorous, testable definitions — grounded in the published operators, claiming novelty only
where it is real (interface restriction + toxicity application + IPC vectorization).
- [ ] `metric`: opposition distance d_op (Euclidean across protein↔ligand, ∞ within affiliation).
      Cite PATH/TopologyNet. Constraint set: symmetry, non-negativity, correct ∞-handling,
      reproduces a hand-checked tiny example. (Unit-testable; an *engineering*-flavored derivation.)
- [ ] `complex`: bipartite interface complex = VR on the opposition-distance cloud, **restricted
      to interface atoms** (ligand + protein atoms within a cutoff of the ligand). The interface
      restriction is the arguably-novel twist — define it precisely (cutoff, atom selection).
- [ ] `homology`: filtration + PH → diagrams (Ripser/GUDHI). Element-specific channels (ESPH-style
      atom-type pairs) as an option.
- [ ] `features` / **IPC**: vectorize diagrams. Baseline = persistence image (Adams). Then define
      what "Internuclear Persistence Contours" adds beyond a renamed persistence image — e.g.
      per-element-pair ("internuclear") binning, or contour/level-set features of the persistence
      surface restricted to the interface. **If IPC ≠ persistence image, state the difference and
      test it; if it is, drop the name and say so.** Constraint set: invariance to global
      rotation/translation, stability sanity (small perturbation → small feature change).
Exit: each construct has a constraint set that passes + a docstring citing its prior art.

## Phase 2 — Prototype (thin vertical slice FIRST)  [Engineering + Derivation]  ◑ B DONE 2026-06-21
Goal: smallest end-to-end pipeline that produces features for real hERG compounds — built to
enable the honest **three-way comparison**, because hERG may be ligand-only predictable.
- [x] Track B shipped: `homology` (VR PH + fixed-length vectorize) + `features.ligand_ph_features`
      (SMILES→ETKDGv3 conformer→heavy-atom cloud→PH vector, 88 dims). Constraint set passes
      (`tests/test_homology.py`: fixed length, exact rotation/translation invariance, perturbation
      stability, degenerate/invalid handling).
- [x] **Gate run on B early (the cheap kill experiment)** — `benchmarks/experiment_ligand_ph.py`.
      Result signed in `docs/gate-result-ligand-ph.md`: **NO PULSE**. B alone AUROC 0.71/MCC 0.26
      (≪ A's 0.84/0.58); A⊕B ≈ A within noise (both CIs straddle 0). Vanilla ligand-only PH adds
      no orthogonal hERG signal. Track C (interface PH) still untested.
- [x] ESPH variant shipped (`features.element_ph_features`, 7 element channels) + gate re-run.
      Signed in `docs/gate-result-esph.md`: **PULSE but gate FAILS**. ESPH *alone* (no descriptors,
      no protein) hits AUROC 0.852 ≥ baseline 0.837 — topology is a competitive *independent*
      representation (vanilla PH was just the wrong formulation). BUT A⊕ESPH ≈ A (CI straddles 0)
      and ESPH is worse on MCC → no orthogonal boost, the redundancy thesis. Key caveat: n_test=132
      is underpowered (AUROC CI ≈ ±0.06); a powered MCC test needs the imbalanced HTS set (TODO).
- [ ] Three feature tracks on the same TDC split:
      (A) descriptor baseline (Phase 0) ✓,
      (B) **ligand-only PH** ✓ — vanilla FAILED; ESPH = descriptor-parity standalone, no boost,
      (C) **interface bipartite PH** (the full opposition-distance/interface construct) — GO/NO-GO.
- [ ] Structure track needs hERG pocket coordinates: cryo-EM **5VA1/7CN1** + ligand poses
      (dock with the Merck-project Vina path or use a fixed reference pose). Keep the point cloud
      interface-restricted (hundreds of atoms) to stay in the 22 ms regime.
- [ ] Confirm CPU cost at true interface scale (open question #2).
Exit: feature matrices A, B, C for the full TDC-648 set, cached + reproducible.

## Phase 3 — VALIDATED (the gate — the whole project hinges here)  [Derivation gate]
Goal: settle the central bet, honestly, with a **pre-registered** success criterion.
- [ ] Pre-register BEFORE looking at test scores. Success = **any** of:
      (i)  interface-PH (C) beats descriptor baseline (A) on AUROC, TDC scaffold split, CI-separated; OR
      (ii) **PH-augmented** (A⊕C concatenated) beats A alone — PH adds orthogonal signal; OR
      (iii) on the imbalanced Sato split, C or A⊕C beats A on **MCC** (the descriptors' soft spot).
      Also report B (ligand-only PH): if B≈C, the protein interface adds nothing for hERG — a
      publishable finding in itself, and a signal to change endpoint.
- [ ] Run `benchmarks/harness.py`. Bootstrap CIs. No metric shopping after the fact.
- [ ] **Decision branch (honest, per repo philosophy):**
      - PASS → Phase 4.
      - FAIL but PH-augmentation helps → narrow MVP to "descriptors + topological booster".
      - FAIL outright → **halt and reformulate** (loop to Phase 1), or pivot the endpoint to a
        binding-**affinity** task where PH is *proven* (PDBbind), then return to toxicity. Do not
        ship a platform that only looks finished.
Exit: a signed gate result (pass/fail + numbers) committed to the repo.

## Phase 4 — MVP (only if Phase 3 passes)  [Engineering]
Goal: `screen` — SMILES/structure in → calibrated hERG risk out, reproducible.
- [ ] `screen`: end-to-end (load → pose/interface → PH → IPC → classifier → calibrated score).
- [ ] Model card: the gate numbers, the applicability domain, honest limitations (incl. the
      ligand-only-vs-interface result), one-command reproduction.
- [ ] Defer non-core infra: `llm` router and `wolfram` harness are *not* on the critical path to
      validating the science — build only if/when they pay for themselves.
Exit: `topologix screen <smiles>` returns a calibrated risk + provenance; model card published.

---

## Sequencing logic (why this order)
The repo lists `complex → metric → homology → features → screen`. The research says invert the
*risk*: the cheapest path to truth is Phase 0 baseline → a **minimal** B/C feature set → Phase 3
gate, before perfecting the math. Build the thin slice end-to-end first (Phases 0→2→3 minimal),
get a yes/no on whether topology has *any* edge on hERG, and only then deepen the derivation
(Phase 1 refinements) if the signal is there. Spend derivation effort where the gate says it pays.

## Pre-registered honesty rails (the IP is the rigor)
1. The gate compares against a **strong** baseline (0.84 AUROC), not a strawman.
2. Success criterion fixed **before** seeing test scores; report MCC + AUROC + CIs, no cherry-pick.
3. Always report ligand-only PH (B) alongside interface PH (C) — the interface must *earn* its place.
4. Cite the prior art (Cang & Wei, PATH, Adams) in the math docstrings; claim novelty only for the
   interface-restricted-bipartite-PH-for-toxicity application + any genuine IPC delta.
5. A failed gate is a *result*, not a setback — it's logged and it redirects the build.
