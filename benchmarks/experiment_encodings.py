"""Experiment 7 gate: eight pre-registered comparisons, Holm-Bonferroni corrected.

Decision rule fixed in PREREGISTRATION.md before featurization. Two things about this gate
exist because of prior mistakes on this board and are not optional:

  - The bootstrap resamples SCAFFOLDS, not molecules (invariant I-2). Row-level intervals are
    reported beside them; where they disagree the cluster version governs. That rule decided
    V-030.
  - Eight comparisons are corrected with Holm-Bonferroni at family-wise alpha = 0.05
    (invariant I-5's multiplicity cousin). At a nominal 95% level, one spurious exclusion of
    zero across eight tests is likely rather than exceptional, and V-002 is what an
    uncorrected near-miss looks like when it is believed.
"""

import json
import pickle

import numpy as np
from scipy.stats import norm
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier

SEED = 0
N_BOOT = 2000
ALPHA = 0.05
TRAIN_FRAC = 0.8
EXP2_BASELINE_AUROC = 0.878     # validity precondition
PRECONDITION_TOL = 0.03

FAMILY = [("X", False), ("X", True), ("L", False), ("L", True),
          ("S", False), ("S", True), ("U", False), ("U", True)]


def load():
    d = pickle.load(open("results/features.pkl", "rb"))
    rows = d["rows"]
    y = np.array([r["y"] for r in rows])
    scaf = np.array([r["scaffold"] for r in rows])
    blocks = {k: np.nan_to_num(np.array([r[k] for r in rows]), posinf=0.0, neginf=0.0)
              for k in ("A", "V", "X", "L", "S")}
    blocks["U"] = np.concatenate([blocks["X"], blocks["L"], blocks["S"]], axis=1)
    return blocks, y, scaf, d["dropped"]


def scaffold_split(scaf, n_total):
    """Deterministic Bemis-Murcko split: largest scaffold groups fill train first."""
    groups = {}
    for i, s in enumerate(scaf):
        groups.setdefault(s, []).append(i)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), scaf[g[0]]))
    train, target = [], int(TRAIN_FRAC * n_total)
    test = []
    for g in ordered:
        (train if len(train) < target else test).extend(g)
    return np.array(train), np.array(test)


def fit_predict(X, y, tr, te):
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                      n_jobs=-1, eval_metric="logloss", tree_method="hist")
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1]


def boot_delta(y, pa, pb, clusters, cluster, n=N_BOOT):
    """delta = AUROC(b) - AUROC(a); positive favours b (topology)."""
    rng = np.random.RandomState(SEED)
    if cluster:
        uniq = np.unique(clusters)
        idx = {c: np.where(clusters == c)[0] for c in uniq}
    deltas = []
    for _ in range(n):
        if cluster:
            s = np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        else:
            s = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        deltas.append(roc_auc_score(y[s], pb[s]) - roc_auc_score(y[s], pa[s]))
    d = np.array(deltas)
    return {"delta": float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
            "boot": d, "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p_one_sided": float((d <= 0).mean())}


def holm(pvals, alpha=ALPHA):
    """Holm-Bonferroni. Returns per-test rejection flags and the corrected level applied."""
    order = np.argsort(pvals)
    m = len(pvals)
    reject = np.zeros(m, bool)
    levels = np.zeros(m)
    for rank, i in enumerate(order):
        lvl = alpha / (m - rank)
        levels[i] = lvl
        if pvals[i] <= lvl:
            reject[i] = True
        else:
            break          # Holm stops at the first failure
    return reject, levels


def main():
    blocks, y, scaf, dropped = load()
    tr, te = scaffold_split(scaf, len(y))
    print(f"{len(y)} molecules kept, dropped {dropped} | "
          f"train {len(tr)} / test {len(te)} | "
          f"{len(np.unique(scaf))} scaffolds, {len(np.unique(scaf[te]))} in test", flush=True)
    print(f"positive rate: train {y[tr].mean():.3f}  test {y[te].mean():.3f}\n", flush=True)

    yt, st = y[te], scaf[te]
    preds, scores = {}, {}
    for name in ("A", "V", "X", "L", "S", "U"):
        preds[name] = fit_predict(blocks[name], y, tr, te)
        scores[name] = {"auroc": float(roc_auc_score(yt, preds[name])),
                        "mcc": float(matthews_corrcoef(yt, (preds[name] >= 0.5).astype(int))),
                        "dim": int(blocks[name].shape[1])}
        print(f"  {name:2s} dim {scores[name]['dim']:5d}  "
              f"AUROC {scores[name]['auroc']:.4f}  MCC {scores[name]['mcc']:.4f}", flush=True)

    for name in ("X", "L", "S", "U"):
        cat = np.concatenate([blocks["A"], blocks[name]], axis=1)
        key = f"A+{name}"
        preds[key] = fit_predict(cat, y, tr, te)
        scores[key] = {"auroc": float(roc_auc_score(yt, preds[key])),
                       "mcc": float(matthews_corrcoef(yt, (preds[key] >= 0.5).astype(int))),
                       "dim": int(cat.shape[1])}
        print(f"  {key:4s} dim {scores[key]['dim']:5d}  AUROC {scores[key]['auroc']:.4f}  "
              f"MCC {scores[key]['mcc']:.4f}", flush=True)

    precondition_ok = abs(scores["A"]["auroc"] - EXP2_BASELINE_AUROC) <= PRECONDITION_TOL
    print(f"\nvalidity precondition: A = {scores['A']['auroc']:.4f} vs Experiment 2's "
          f"{EXP2_BASELINE_AUROC} -> {'OK' if precondition_ok else 'FAILED'}", flush=True)

    results, pvals, labels = {}, [], []
    for name, concat in FAMILY:
        key = f"A+{name}_vs_A" if concat else f"{name}_vs_A"
        pb = preds[f"A+{name}"] if concat else preds[name]
        cl = boot_delta(yt, preds["A"], pb, st, cluster=True)
        rw = boot_delta(yt, preds["A"], pb, st, cluster=False)
        results[key] = {
            "cluster": {k: v for k, v in cl.items() if k != "boot"},
            "row": {k: v for k, v in rw.items() if k != "boot"},
            "uncorrected_pass_cluster": bool(cl["ci_lo"] > 0),
            "uncorrected_pass_row": bool(rw["ci_lo"] > 0),
        }
        pvals.append(cl["p_one_sided"])
        labels.append(key)

    reject, levels = holm(np.array(pvals))
    for i, key in enumerate(labels):
        results[key]["holm_level"] = float(levels[i])
        results[key]["p_one_sided_cluster"] = float(pvals[i])
        results[key]["PASS"] = bool(reject[i])

    out = {"n": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
           "n_scaffolds": int(len(np.unique(scaf))), "dropped": dropped,
           "scores": scores, "precondition_ok": bool(precondition_ok),
           "family": labels, "results": results,
           "verdict_pass": bool(reject.any())}
    json.dump(out, open("results/gate.json", "w"), indent=2)

    print("\n" + "=" * 78)
    for key in labels:
        r = results[key]
        print(f"  {key:12s} delta={r['cluster']['delta']:+.4f} "
              f"cluster95%[{r['cluster']['ci_lo']:+.4f},{r['cluster']['ci_hi']:+.4f}] "
              f"p={r['p_one_sided_cluster']:.4f} holm<={r['holm_level']:.4f} "
              f"PASS={r['PASS']}", flush=True)
    print(f"\n  control V (replication, not in family): AUROC {scores['V']['auroc']:.4f}")
    print(f"  VERDICT_PASS = {out['verdict_pass']}")
    print("wrote results/gate.json")


if __name__ == "__main__":
    main()
