"""Descriptor baseline: RDKit 2D descriptors + XGBoost.

This is BOTH the product floor (a shippable hERG screener on its own) and the line the
topological features must beat. Per the research it should reach ~0.84 AUROC on the TDC
hERG scaffold split — a strong, non-strawman baseline.

Run: python benchmarks/baseline.py
"""
from __future__ import annotations
import warnings

import numpy as np

# the canonical RDKit 2D descriptor block (~200 features)
def _descriptor_fns():
    from rdkit.Chem import Descriptors
    return Descriptors._descList  # [(name, fn), ...]


def featurize(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """SMILES -> (X descriptor matrix, valid-mask). Invalid SMILES are dropped via the mask.
    NaN/inf are sanitized to 0 so a single bad descriptor doesn't poison a row."""
    from rdkit import Chem
    fns = _descriptor_fns()
    rows, mask = [], []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            mask.append(False)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals = []
            for _name, fn in fns:
                try:
                    vals.append(float(fn(m)))
                except Exception:  # noqa: BLE001
                    vals.append(0.0)
        rows.append(vals)
        mask.append(True)
    X = np.asarray(rows, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.asarray(mask, dtype=bool)


def train_xgb(X, y, seed: int = 0):
    from xgboost import XGBClassifier
    pos = max(1, int(np.sum(y)))
    neg = max(1, len(y) - pos)
    clf = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
        scale_pos_weight=neg / pos, n_jobs=4, random_state=seed,
    )
    clf.fit(X, y)
    return clf


def run_baseline(split):
    """Fit on train, score test. Returns (test_scores, y_true, metrics)."""
    from topologix.metrics import classification_metrics
    Xtr, mtr = featurize(split.train_smiles)
    Xte, mte = featurize(split.test_smiles)
    ytr = np.asarray(split.train_y)[mtr]
    yte = np.asarray(split.test_y)[mte]
    clf = train_xgb(Xtr, ytr)
    scores = clf.predict_proba(Xte)[:, 1]
    return scores, yte, classification_metrics(yte, scores)


def main():
    from topologix.data import load_herg_tdc
    split = load_herg_tdc()
    print("loaded:", split)
    _scores, _y, m = run_baseline(split)
    print("\nDESCRIPTOR BASELINE (RDKit 2D + XGBoost) on TDC hERG scaffold split:")
    for k, v in m.items():
        print(f"  {k:8} {v:.4f}")
    print(f"\n  target from literature: AUROC ~0.84 (RDKit2D+MLP), SOTA 0.880")


if __name__ == "__main__":
    main()
