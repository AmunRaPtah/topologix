"""Shared classification metrics + bootstrap utilities for the validation gate.

hERG is imbalanced, so we never judge on accuracy alone. AUROC is the TDC headline; MCC is the
metric descriptors plateau on and where a better representation can actually help.
"""
from __future__ import annotations

import numpy as np


def classification_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 matthews_corrcoef, roc_auc_score, average_precision_score)
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    out = {}
    # AUROC/AUPRC need both classes present
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, scores))
        out["auprc"] = float(average_precision_score(y_true, scores))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    out["mcc"] = float(matthews_corrcoef(y_true, pred))
    out["bacc"] = float(balanced_accuracy_score(y_true, pred))
    out["acc"] = float(accuracy_score(y_true, pred))
    return out


def bootstrap_delta(y_true, scores_a, scores_b, metric: str = "auroc",
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap of (metric(b) - metric(a)) over the SAME resampled test rows.

    Returns the observed delta, a 95% CI, and the fraction of resamples where b>a. "b beats a"
    means the CI excludes 0 (delta_lo > 0) — i.e. the gain is not noise.
    """
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    obs = classification_metrics(y_true, b)[metric] - classification_metrics(y_true, a)[metric]
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yi = y_true[idx]
        if len(np.unique(yi)) < 2:
            continue
        d = classification_metrics(yi, b[idx])[metric] - classification_metrics(yi, a[idx])[metric]
        deltas.append(d)
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"metric": metric, "delta": float(obs), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_b_gt_a": float(np.mean(deltas > 0)), "n_boot": len(deltas)}
