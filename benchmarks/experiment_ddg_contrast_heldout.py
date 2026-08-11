"""Task B: absolute dG across 370 structurally diverse complexes, held-out target.

Same ligands, same S and T matrices, same model, same split axis as Task A. Only the label
changes from a within-target difference to a level. Decision rule in
PREREGISTRATION-CONTRAST.md, fixed before this ran.

The cluster bootstrap governs from the outset. V-030 established that the pair-level version
manufactures significance out of grouping structure.
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


def load():
    d = pickle.load(open("results/features.pkl", "rb"))
    seen, rows = set(), []
    for r in d["rows"]:
        key = (r["target"], r["name"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    S = np.nan_to_num(np.array([r["S"] for r in rows]), posinf=0.0, neginf=0.0)
    T = np.nan_to_num(np.array([r["T"] for r in rows]), posinf=0.0, neginf=0.0)
    y = np.array([r["dg"] for r in rows])
    g = np.array([r["target"] for r in rows])
    return S, T, y, g


def oof(X, y, groups, constant=False):
    gkf = GroupKFold(n_splits=N_FOLDS)
    pred = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        if constant:
            pred[te] = y[tr].mean()
        else:
            m = RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=SEED)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
    return pred


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def cluster_boot(y, pa, pb, groups, n=N_BOOT):
    rng = np.random.RandomState(SEED)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    d = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        d.append(rmse(y[s], pa[s]) - rmse(y[s], pb[s]))
    d = np.array(d)
    return {"delta": rmse(y, pa) - rmse(y, pb),
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p_better": float((d > 0).mean())}


def diagnostic_R(T, y, pairs=None):
    """R = (relative topological separation) / (relative label separation).

    pairs=None means every entity is compared against the global pool (Task B).
    """
    def med_pairwise(M, sample=4000, rng=np.random.RandomState(SEED)):
        n = len(M)
        i = rng.randint(0, n, sample)
        j = rng.randint(0, n, sample)
        keep = i != j
        return np.median(np.linalg.norm(M[i[keep]] - M[j[keep]], axis=1))

    glob_T = med_pairwise(T)
    glob_y = np.median(np.abs(y[np.random.RandomState(SEED).randint(0, len(y), 4000)]
                              - y[np.random.RandomState(SEED + 1).randint(0, len(y), 4000)]))
    if pairs is None:
        return 1.0, {"note": "global pool compared against itself; R is 1 by construction"}
    dT = np.median([np.linalg.norm(T[b] - T[a]) for a, b in pairs])
    dy = np.median([abs(y[b] - y[a]) for a, b in pairs])
    rel_T, rel_y = dT / glob_T, dy / glob_y
    return float(rel_T / rel_y), {"rel_topo": float(rel_T), "rel_label": float(rel_y),
                                  "median_pair_topo_dist": float(dT),
                                  "median_global_topo_dist": float(glob_T)}


def main():
    S, T, y, g = load()
    TS = np.concatenate([S, T], axis=1)
    print(f"Task B: {len(y)} ligands, {len(np.unique(g))} targets, "
          f"dG {y.min():.2f}..{y.max():.2f} kcal/mol, sd {y.std():.2f}\n", flush=True)

    preds = {"null": oof(S, y, g, constant=True)}
    for name, X in (("S", S), ("T", T), ("TS", TS)):
        preds[name] = oof(X, y, g)
    for k in ("null", "S", "T", "TS"):
        r = spearmanr(y, preds[k]).statistic if k != "null" else float("nan")
        print(f"  {k:5s} RMSE {rmse(y, preds[k]):.3f}  rho {r:+.3f}", flush=True)

    res = {"task": "absolute dG, held-out target", "n": int(len(y)),
           "n_targets": int(len(np.unique(g))),
           "rmse": {k: rmse(y, p) for k, p in preds.items()},
           "spearman": {k: float(spearmanr(y, preds[k]).statistic)
                        for k in ("S", "T", "TS")},
           "gates": {}}
    for label, a, b in (("T_vs_S", "S", "T"), ("TS_vs_S", "S", "TS"),
                        ("S_vs_null", "null", "S"), ("T_vs_null", "null", "T")):
        res["gates"][label] = cluster_boot(y, preds[a], preds[b], g)

    res["verdict_pass"] = bool(res["gates"]["T_vs_S"]["ci_lo"] > 0
                               or res["gates"]["TS_vs_S"]["ci_lo"] > 0)

    # Diagnostic R for Task A (within-target congeneric pairs) vs Task B (global pool).
    by_t = {}
    for i, t in enumerate(g):
        by_t.setdefault(t, []).append(i)
    pairs_A = [(a, b) for idxs in by_t.values()
               for n, a in enumerate(sorted(idxs)) for b in sorted(idxs)[n + 1:]]
    R_A, det_A = diagnostic_R(T, y, pairs_A)
    R_B, det_B = diagnostic_R(T, y, None)
    res["diagnostic_R"] = {"task_A_within_target": R_A, "task_A_detail": det_A,
                           "task_B_global": R_B, "task_B_detail": det_B}

    print()
    for label, gate in res["gates"].items():
        print(f"  {label:12s} delta={gate['delta']:+.4f} "
              f"95%CI[{gate['ci_lo']:+.4f},{gate['ci_hi']:+.4f}] pass={gate['ci_lo'] > 0}",
              flush=True)
    print(f"\n  VERDICT_PASS (Task B) = {res['verdict_pass']}")
    print(f"  diagnostic R: Task A (congeneric) = {R_A:.3f} | Task B (diverse) = {R_B:.3f}")
    print(f"    Task A detail: {det_A}")

    json.dump(res, open("results/contrast.json", "w"), indent=2)
    print("\nwrote results/contrast.json")


if __name__ == "__main__":
    main()
