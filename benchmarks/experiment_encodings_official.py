"""Confirmatory rerun: TDC's own scaffold split, and Experiment 2's own hyperparameters.

Both changes are anchored externally (TDC's create_scaffold_split; benchmarks/baseline.py in
the topologix repo) rather than chosen after seeing scores, and both apply identically to
every feature block. Decision rule, family, Holm-Bonferroni correction, cluster-bootstrap
rule and validity precondition are unchanged from PREREGISTRATION.md. See Addendum 2.
"""
import json, pickle
import numpy as np
import provenance, pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier
from experiment_encodings import boot_delta, holm, FAMILY, EXP2_BASELINE_AUROC, PRECONDITION_TOL, SEED

SRC = "/root/topologix/repo/benchmarks/data/herg_karim.tab"

def fit_predict(X, y, tr, te):
    """Experiment 2's baseline configuration, verbatim from benchmarks/baseline.py."""
    pos = float(y[tr].sum()); neg = float(len(tr) - pos)
    m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                      scale_pos_weight=neg / pos, n_jobs=4, random_state=SEED,
                      eval_metric="logloss", tree_method="hist")
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1]

def _check_provenance():
    """Fail loudly rather than silently compare matrices from different environments."""
    provenance.assert_compatible("results/features.pkl")


def main():
    _check_provenance()
    d = pickle.load(open("results/features.pkl", "rb"))
    rows = d["rows"]
    # Source indices recovered by align.py (deterministic re-run of the embedding decision,
    # verified against 13,434 rows with label agreement 1.0) rather than stored at
    # featurization time: the edit that stored them was reverted from the working tree.
    idx = np.array(pickle.load(open("results/kept_indices.pkl", "rb")))
    y = np.array([r["y"] for r in rows])
    scaf = np.array([r["scaffold"] for r in rows])
    blocks = {k: np.nan_to_num(np.array([r[k] for r in rows]), posinf=0.0, neginf=0.0)
              for k in ("A", "V", "X", "L", "S")}
    blocks["U"] = np.concatenate([blocks["X"], blocks["L"], blocks["S"]], axis=1)

    # Split cached by TDC's own create_scaffold_split (seed 42, frac [0.7,0.1,0.2]) before
    # rdkit was restored. Cached rather than recomputed because installing PyTDC downgraded
    # rdkit and changed Descriptors._descList from 217 to 210, silently altering the feature
    # space; the descriptor count is now back to 217, matching Experiment 2's recorded
    # feature_dims and this experiment's first run.
    sp = pickle.load(open("results/official_split.pkl", "rb"))
    test_src = set(sp["test_idx"])
    pos_in = {v: i for i, v in enumerate(idx)}
    te = np.array(sorted(pos_in[s] for s in test_src if s in pos_in))
    tr = np.array(sorted(set(range(len(y))) - set(te.tolist())))
    print(f"{len(y)} featurized | official TDC split -> train {len(tr)} / test {len(te)}", flush=True)
    print(f"test scaffolds {len(np.unique(scaf[te]))} for {len(te)} molecules "
          f"(cluster bootstrap is {'MEANINGFUL' if len(np.unique(scaf[te])) < len(te) else 'DEGENERATE'})", flush=True)
    print(f"positive rate: train {y[tr].mean():.3f}  test {y[te].mean():.3f}\n", flush=True)

    yt, st = y[te], scaf[te]
    preds, scores = {}, {}
    for name in ("A", "V", "X", "L", "S", "U"):
        preds[name] = fit_predict(blocks[name], y, tr, te)
        scores[name] = {"auroc": float(roc_auc_score(yt, preds[name])),
                        "mcc": float(matthews_corrcoef(yt, (preds[name] >= 0.5).astype(int))),
                        "dim": int(blocks[name].shape[1])}
        print(f"  {name:2s} dim {scores[name]['dim']:5d}  AUROC {scores[name]['auroc']:.4f}"
              f"  MCC {scores[name]['mcc']:.4f}", flush=True)
    for name in ("X", "L", "S", "U"):
        cat = np.concatenate([blocks["A"], blocks[name]], axis=1)
        k = f"A+{name}"
        preds[k] = fit_predict(cat, y, tr, te)
        scores[k] = {"auroc": float(roc_auc_score(yt, preds[k])),
                     "mcc": float(matthews_corrcoef(yt, (preds[k] >= 0.5).astype(int))),
                     "dim": int(cat.shape[1])}
        print(f"  {k:4s} dim {scores[k]['dim']:5d}  AUROC {scores[k]['auroc']:.4f}"
              f"  MCC {scores[k]['mcc']:.4f}", flush=True)

    ok = abs(scores["A"]["auroc"] - EXP2_BASELINE_AUROC) <= PRECONDITION_TOL
    print(f"\nvalidity precondition: A = {scores['A']['auroc']:.4f} vs {EXP2_BASELINE_AUROC}"
          f" -> {'OK' if ok else 'FAILED'}", flush=True)

    results, pvals, labels = {}, [], []
    for name, concat in FAMILY:
        key = f"A+{name}_vs_A" if concat else f"{name}_vs_A"
        pb = preds[f"A+{name}"] if concat else preds[name]
        cl = boot_delta(yt, preds["A"], pb, st, cluster=True)
        rw = boot_delta(yt, preds["A"], pb, st, cluster=False)
        results[key] = {"cluster": {k: v for k, v in cl.items() if k != "boot"},
                        "row": {k: v for k, v in rw.items() if k != "boot"},
                        "uncorrected_pass_cluster": bool(cl["ci_lo"] > 0)}
        pvals.append(cl["p_one_sided"]); labels.append(key)
    reject, levels = holm(np.array(pvals))
    for i, key in enumerate(labels):
        results[key].update({"holm_level": float(levels[i]),
                             "p_one_sided_cluster": float(pvals[i]),
                             "PASS": bool(reject[i])})

    out = {"split": "TDC create_scaffold_split seed=42 frac=[0.7,0.1,0.2], train+valid merged",
           "hyperparameters": "Experiment 2 baseline.py verbatim",
           "n": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
           "n_test_scaffolds": int(len(np.unique(scaf[te]))),
           "scores": scores, "precondition_ok": bool(ok),
           "family": labels, "results": results, "verdict_pass": bool(reject.any())}
    json.dump(out, open("results/gate_official.json", "w"), indent=2)
    print("\n" + "=" * 78, flush=True)
    for key in labels:
        r = results[key]
        print(f"  {key:12s} delta={r['cluster']['delta']:+.4f} "
              f"cluster95%[{r['cluster']['ci_lo']:+.4f},{r['cluster']['ci_hi']:+.4f}] "
              f"p={r['p_one_sided_cluster']:.4f} PASS={r['PASS']}", flush=True)
    print(f"\n  control V: AUROC {scores['V']['auroc']:.4f}")
    print(f"  VERDICT_PASS = {out['verdict_pass']}")


if __name__ == "__main__":
    main()
