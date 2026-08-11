"""Task B': the same absolute-dG contrast under BOTH split regimes.

Task B used held-out-target GroupKFold out of I-2 discipline. That was the wrong control
for this question: it is strictly HARDER than the D3R/PDBbind regime in which persistent
homology's external positive result was obtained, where related targets appear in both
train and test. Under held-out-target every feature block fell below the null predictor,
so the comparator collapsed and the run said nothing about topology's marginal value.

Fix: report both splits on identical rows, as B-HELDOUT-SPLIT-SUITE does for P23. The
random split is NOT a generalization claim; it reproduces the regime the external claim
was measured in, so that topology's contribution can be assessed where the baseline works.
"""
import json, pickle, numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from experiment_ddg_contrast_heldout import load, rmse, cluster_boot, SEED, N_FOLDS

def oof(X, y, groups, scheme, constant=False):
    sp = (GroupKFold(n_splits=N_FOLDS).split(X, y, groups) if scheme == "group"
          else KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X))
    pred = np.zeros(len(y))
    for tr, te in sp:
        if constant: pred[te] = y[tr].mean()
        else:
            m = RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=SEED)
            m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred

def main():
    S, T, y, g = load(); TS = np.concatenate([S, T], axis=1)
    out = {"n": int(len(y)), "n_targets": int(len(np.unique(g))), "splits": {}}
    for scheme, label in (("random", "random 5-fold (D3R-comparable regime)"),
                          ("group", "held-out target (generalization regime)")):
        preds = {"null": oof(S, y, g, scheme, constant=True)}
        for n, X in (("S", S), ("T", T), ("TS", TS)):
            preds[n] = oof(X, y, g, scheme)
        r = {"label": label,
             "rmse": {k: rmse(y, p) for k, p in preds.items()},
             "spearman": {k: float(spearmanr(y, preds[k]).statistic) for k in ("S","T","TS")},
             "gates": {}}
        for lab, a, b in (("T_vs_S","S","T"), ("TS_vs_S","S","TS"),
                          ("S_vs_null","null","S"), ("T_vs_null","null","T")):
            r["gates"][lab] = cluster_boot(y, preds[a], preds[b], g)
        r["verdict_pass"] = bool(r["gates"]["T_vs_S"]["ci_lo"] > 0
                                 or r["gates"]["TS_vs_S"]["ci_lo"] > 0)
        r["baseline_beats_null"] = bool(r["gates"]["S_vs_null"]["ci_lo"] > 0)
        out["splits"][scheme] = r
        print(f"\n== {label}")
        for k in ("null","S","T","TS"):
            rr = r["spearman"].get(k)
            print(f"  {k:5s} RMSE {r['rmse'][k]:.3f}" + (f"  rho {rr:+.3f}" if rr else ""))
        for lab, gt in r["gates"].items():
            print(f"  {lab:12s} delta={gt['delta']:+.4f} "
                  f"95%CI[{gt['ci_lo']:+.4f},{gt['ci_hi']:+.4f}] pass={gt['ci_lo']>0}")
        print(f"  VERDICT_PASS={r['verdict_pass']}  baseline_beats_null={r['baseline_beats_null']}")
    json.dump(out, open("results/contrast2.json","w"), indent=2)
    print("\nwrote results/contrast2.json")

main()
