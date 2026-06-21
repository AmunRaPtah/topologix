"""The kill experiment: does topology have ANY pulse on hERG?  (Phase 2 -> Phase 3)

Three feature tracks on the identical TDC hERG scaffold split:
    A  = RDKit 2D descriptors (the strong baseline / product floor)
    B  = ligand-only persistent homology (3D conformer atom cloud -> PH vector)
    AB = A concatenated with B (descriptor ⊕ topological booster)

Pre-registered success (docs/BUILD_PLAN.md, Phase 3), judged by the paired-bootstrap
gate (CI on the metric delta excludes 0), on AUROC and MCC:
    (i)  B beats A           -> topology stands alone, or
    (ii) AB beats A          -> topology adds orthogonal signal (the booster thesis).
B is always reported next to A so the topology has to *earn* its place.

This is the cheapest honest version of the experiment that can kill the project:
no protein, no docking. If even the augmentation shows no pulse, the interface
construct (track C) is unlikely to rescue hERG and the endpoint should be revisited.

All three tracks are trained and tested on the INTERSECTION of the descriptor- and
conformer-valid rows, so every track sees identical molecules and the gate compares
aligned scores on the same held-out set.

Run (from repo root): python -m benchmarks.experiment_ligand_ph
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from topologix.data import load_herg_tdc
from topologix.metrics import classification_metrics
from benchmarks.baseline import featurize as descriptor_featurize, train_xgb
from benchmarks.harness import run_gate
from topologix.features import ligand_ph_features, element_ph_features

OUT = Path(__file__).resolve().parent / "results"

# Track-B variant registry: name -> (featurizer, output-tag)
B_VARIANTS = {
    "vanilla": (ligand_ph_features, "ligand_ph_gate"),   # all-atom ligand PH
    "esph": (element_ph_features, "esph_gate"),           # element-specific channels
}


def _featurize_aligned(smiles, y, b_featurizer, maxdim=1):
    """Return (Xa, Xb, y) over the intersection of descriptor-valid and PH-valid rows."""
    y = np.asarray(y).astype(int)
    Xa_v, mA = descriptor_featurize(smiles)
    Xb_v, mB = b_featurizer(smiles, maxdim=maxdim)
    keep = mA & mB
    Xa = Xa_v[keep[mA]]            # rows of the A-valid block that are also B-valid
    Xb = Xb_v[keep[mB]]           # rows of the B-valid block that are also A-valid
    return Xa, Xb, y[keep], keep


def run(variant: str = "vanilla", seed: int = 0, maxdim: int = 1) -> dict:
    b_featurizer, tag = B_VARIANTS[variant]
    split = load_herg_tdc()
    print(f"loaded ({variant}):", split)

    Xa_tr, Xb_tr, ytr, ktr = _featurize_aligned(split.train_smiles, split.train_y, b_featurizer, maxdim)
    Xa_te, Xb_te, yte, kte = _featurize_aligned(split.test_smiles, split.test_y, b_featurizer, maxdim)
    print(f"aligned rows: train {ktr.sum()}/{len(ktr)}  test {kte.sum()}/{len(kte)}  "
          f"(test pos={int(yte.sum())})")
    print(f"feature dims: A={Xa_tr.shape[1]}  B={Xb_tr.shape[1]}  AB={Xa_tr.shape[1]+Xb_tr.shape[1]}")

    tracks = {
        "A_descriptor": (Xa_tr, Xa_te),
        "B_ligandPH": (Xb_tr, Xb_te),
        "AB_concat": (np.hstack([Xa_tr, Xb_tr]), np.hstack([Xa_te, Xb_te])),
    }
    scores, metrics = {}, {}
    for name, (Xtr, Xte) in tracks.items():
        clf = train_xgb(Xtr, ytr, seed=seed)
        s = clf.predict_proba(Xte)[:, 1]
        scores[name] = s
        metrics[name] = classification_metrics(yte, s)

    print("\nPER-TRACK METRICS (TDC hERG scaffold split, aligned rows):")
    hdr = ["track", "auroc", "auprc", "mcc", "bacc", "acc"]
    print("  " + "  ".join(f"{h:>12}" for h in hdr))
    for name, m in metrics.items():
        print("  " + f"{name:>12}" + "  " + "  ".join(f"{m[k]:>12.4f}" for k in hdr[1:]))

    print("\nGATE (paired bootstrap, PASS iff 95% CI on delta excludes 0):")
    gates = {}
    for metric in ("auroc", "mcc"):
        for cand in ("B_ligandPH", "AB_concat"):
            g = run_gate(scores[cand], scores["A_descriptor"], yte, metric=metric)
            gates[f"{cand}_vs_A::{metric}"] = {
                "passes": g.passes, "delta": g.delta, "ci_lo": g.ci_lo, "ci_hi": g.ci_hi,
                "baseline": g.baseline_value, "method": g.method_value,
                "p_method_gt_baseline": g.detail.get("p_b_gt_a"),
            }
            print(f"  {cand:>11} vs A  [{metric}]  {g}")

    result = {
        "split": "TDC admet_group hERG scaffold",
        "n_train": int(ktr.sum()), "n_test": int(kte.sum()), "n_test_pos": int(yte.sum()),
        "maxdim": maxdim, "seed": seed,
        "feature_dims": {"A": int(Xa_tr.shape[1]), "B": int(Xb_tr.shape[1])},
        "metrics": metrics, "gates": gates,
        "verdict_any_pass": any(g["passes"] for g in gates.values()),
    }
    result["variant"] = variant
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{tag}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"\nsigned result -> {path}")
    print(f"VERDICT: {'PULSE — at least one gate PASSES' if result['verdict_any_pass'] else 'NO PULSE — all gates FAIL on this split'}")
    return result


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "vanilla")
