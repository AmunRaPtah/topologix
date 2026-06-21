"""Data loaders: hERG benchmarks (the validation target).

The MVP gate is judged on the TDC hERG benchmark (648 compounds, official scaffold split),
with the larger Karim/Central sets reserved for HTS-scale MCC evaluation. We load through
PyTDC so the split is the canonical, citable one — no home-rolled split that could leak.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "benchmarks" / "data"


@dataclass
class Split:
    """A train/test split as parallel SMILES + binary-label lists."""
    train_smiles: list[str]
    train_y: list[int]
    test_smiles: list[str]
    test_y: list[int]

    def __repr__(self) -> str:
        return (f"Split(train={len(self.train_smiles)} [{sum(self.train_y)} pos], "
                f"test={len(self.test_smiles)} [{sum(self.test_y)} pos])")


def load_herg_tdc(seed: int = 1) -> Split:
    """TDC hERG (admet_group) benchmark with its official scaffold split.

    Binary hERG blockade, AUROC task, 648 drugs. The train_val/test partition is fixed by
    TDC; the TEST set is canonical and identical across runs, so the gate is reproducible
    and leak-free.
    """
    from tdc.benchmark_group import admet_group
    CACHE.mkdir(parents=True, exist_ok=True)
    group = admet_group(path=str(CACHE))
    bench = group.get("hERG")
    train_val, test = bench["train_val"], bench["test"]
    return Split(
        train_smiles=train_val["Drug"].tolist(), train_y=train_val["Y"].astype(int).tolist(),
        test_smiles=test["Drug"].tolist(), test_y=test["Y"].astype(int).tolist(),
    )


def load_herg_karim(seed: int = 1) -> Split:
    """TDC hERG_Karim (Karim et al. 2021): ~13.4k compounds, binary blockade.

    The powered companion to the 648-compound admet_group set: a scaffold split with
    ~2.7k test compounds shrinks the gate's bootstrap CI from ~±0.06 to ~±0.013 AUROC,
    enough to resolve the small ligand-topology edge the small set could not. train+valid
    are merged into train_val to mirror the admet_group convention. Roughly class-balanced
    (~49% positive), so AUROC and MCC are both well-estimated.
    """
    from tdc.single_pred import Tox
    CACHE.mkdir(parents=True, exist_ok=True)
    data = Tox(name="hERG_Karim", path=str(CACHE))
    sp = data.get_split(method="scaffold", seed=seed, frac=[0.7, 0.1, 0.2])
    import pandas as pd
    train_val = pd.concat([sp["train"], sp["valid"]], ignore_index=True)
    test = sp["test"]
    return Split(
        train_smiles=train_val["Drug"].tolist(), train_y=train_val["Y"].astype(int).tolist(),
        test_smiles=test["Drug"].tolist(), test_y=test["Y"].astype(int).tolist(),
    )


def load_smiles_csv(path: str, smiles_col: str = "smiles", label_col: str = "y",
                    test_frac: float = 0.2, seed: int = 0) -> Split:
    """Generic loader for an HTS-scale CSV (e.g. Karim/Sato) with a stratified holdout.

    For honest imbalanced-MCC evaluation; scaffold splitting for these is a later refinement.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(path)
    tr, te = train_test_split(df, test_size=test_frac, random_state=seed,
                              stratify=df[label_col])
    return Split(
        train_smiles=tr[smiles_col].tolist(), train_y=tr[label_col].astype(int).tolist(),
        test_smiles=te[smiles_col].tolist(), test_y=te[label_col].astype(int).tolist(),
    )
