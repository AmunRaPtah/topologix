"""The pre-registered gate: does ESPH interface topology add signal for ddG regression?

Decision rule fixed in PREREGISTRATION.md before any data was downloaded.

One deficiency in that pre-registration was found while writing this, and is handled in the
open rather than quietly: it specified "paired bootstrap over held-out predictions" without
naming the resampling UNIT. Pairs within a target are not independent (they share a protein
and a congeneric series), so a pair-level bootstrap understates uncertainty. Both are
reported: the pair-level bootstrap as literally pre-registered, and a target-level cluster
bootstrap as the more conservative check. If they disagree, the cluster bootstrap governs.
This mirrors V-014, where the robustness check the pre-registration did not require halved
the headline effect.
"""

import json
import pickle
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

SEED = 0
N_BOOT = 2000
N_FOLDS = 5
OPENFE_BAR = 1.73   # kcal/mol, free incumbent INC-OPENFE, public set
FEPPLUS_BAR = 0.86  # kcal/mol, commercial INC-FEPPLUS, buried-water subset


def load_pairs():
    d = pickle.load(open("benchmarks/results/features.pkl", "rb"))
    seen, rows = set(), []
    for r in d["rows"]:                      # dedupe: 5 ligands carry multiple SDF poses
        key = (r["target"], r["name"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)

    by_target = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)

    S, T, y, groups, meta = [], [], [], [], []
    for tgt, ligs in sorted(by_target.items()):
        ligs = sorted(ligs, key=lambda r: r["name"])   # canonical order
        for i in range(len(ligs)):
            for j in range(i + 1, len(ligs)):
                a, b = ligs[i], ligs[j]
                S.append(b["S"] - a["S"])
                T.append(b["T"] - a["T"])
                y.append(b["dg"] - a["dg"])
                groups.append(tgt)
                meta.append((tgt, a["name"], b["name"]))
    return (np.nan_to_num(np.array(S), posinf=0.0, neginf=0.0),
            np.nan_to_num(np.array(T), posinf=0.0, neginf=0.0),
            np.array(y), np.array(groups), meta, len(rows))


def oof(X, y, groups):
    gkf = GroupKFold(n_splits=N_FOLDS)
    pred = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        m = RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=SEED)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def boot_delta(y, pa, pb, groups, cluster, n=N_BOOT):
    """delta = RMSE(a) - RMSE(b); positive favours b (the new method)."""
    rng = np.random.RandomState(SEED)
    if cluster:
        uniq = np.unique(groups)
        idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    deltas = []
    for _ in range(n):
        if cluster:
            gs = rng.choice(uniq, len(uniq), replace=True)
            s = np.concatenate([idx_by_g[g] for g in gs])
        else:
            s = rng.choice(len(y), len(y), replace=True)
        deltas.append(rmse(y[s], pa[s]) - rmse(y[s], pb[s]))
    d = np.array(deltas)
    return {"delta": rmse(y, pa) - rmse(y, pb),
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p_better": float((d > 0).mean())}


def main():
    S, T, y, groups, meta, n_lig = load_pairs()
    TS = np.concatenate([S, T], axis=1)
    print(f"{n_lig} ligands | {len(y)} pairs | {len(np.unique(groups))} targets "
          f"| S dim {S.shape[1]} | T dim {T.shape[1]}", flush=True)
    print(f"ddG range {y.min():.2f} .. {y.max():.2f} kcal/mol, sd {y.std():.2f}\n", flush=True)

    preds = {"null": np.zeros(len(y))}
    for name, X in (("S", S), ("T", T), ("TS", TS)):
        preds[name] = oof(X, y, groups)
        print(f"  {name:5s} RMSE {rmse(y, preds[name]):.3f}  "
              f"rho {spearmanr(y, preds[name]).statistic:+.3f}", flush=True)
    print(f"  {'null':5s} RMSE {rmse(y, preds['null']):.3f}", flush=True)

    res = {
        "n_ligands": n_lig, "n_pairs": int(len(y)), "n_targets": int(len(np.unique(groups))),
        "dims": {"S": int(S.shape[1]), "T": int(T.shape[1])},
        "rmse": {k: rmse(y, p) for k, p in preds.items()},
        "spearman": {k: float(spearmanr(y, p).statistic) for k, p in preds.items() if k != "null"},
        "context_bars": {"openfe_free": OPENFE_BAR, "fepplus_commercial": FEPPLUS_BAR},
        "gates": {},
    }
    for label, a, b in (("T_vs_S", "S", "T"), ("TS_vs_S", "S", "TS")):
        res["gates"][label] = {
            "pair_bootstrap": boot_delta(y, preds[a], preds[b], groups, cluster=False),
            "cluster_bootstrap": boot_delta(y, preds[a], preds[b], groups, cluster=True),
        }
    for label, a, b in (("S_vs_null", "null", "S"), ("T_vs_null", "null", "T")):
        res["gates"][label] = {
            "cluster_bootstrap": boot_delta(y, preds[a], preds[b], groups, cluster=True)
        }

    def passes(g):
        return g["cluster_bootstrap"]["ci_lo"] > 0
    res["verdict_pass"] = bool(passes(res["gates"]["T_vs_S"]) or passes(res["gates"]["TS_vs_S"]))
    res["beats_null"] = bool(passes(res["gates"]["S_vs_null"]))

    print()
    for label in ("T_vs_S", "TS_vs_S", "S_vs_null", "T_vs_null"):
        g = res["gates"][label]
        for kind in ("pair_bootstrap", "cluster_bootstrap"):
            if kind not in g:
                continue
            b = g[kind]
            print(f"  {label:12s} {kind:18s} delta={b['delta']:+.4f} "
                  f"95%CI[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}] pass={b['ci_lo'] > 0}", flush=True)
    print(f"\n  VERDICT_PASS = {res['verdict_pass']}  (cluster bootstrap governs)")
    print(f"  cheap baseline beats the null predictor = {res['beats_null']}")
    print(f"  context: OpenFE free {OPENFE_BAR} kcal/mol | FEP+ commercial {FEPPLUS_BAR}")

    json.dump(res, open("results/gate.json", "w"), indent=2)
    np.savez("results/oof_predictions.npz", y=y, groups=groups, **preds)
    print("\nwrote results/gate.json and results/oof_predictions.npz")


if __name__ == "__main__":
    main()
