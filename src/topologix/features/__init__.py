"""IPC feature extraction for downstream classification.

Track B (ligand-only PH): SMILES -> 3D conformer -> heavy-atom point cloud ->
persistent homology -> fixed-length vector. No protein, no docking — the cheapest
honest topological feature set, used to front-load the gate (does topology have
*any* pulse on hERG?) before investing in the interface construct (track C).

Conformer generation is seeded (ETKDGv3 + MMFF94 minimization) so the features
are reproducible. Rotation/translation invariance is inherited from VR PH, which
depends only on the pairwise-distance matrix of the cloud.
"""
from __future__ import annotations

import numpy as np

from topologix.homology import ph_vector


def conformer_cloud(smiles: str, seed: int = 0xC0FFEE) -> np.ndarray | None:
    """SMILES -> (n_heavy_atoms, 3) coordinates of an embedded 3D conformer.

    Returns None if parsing or embedding fails (caller drops the row via the mask).
    Heavy atoms only: hydrogens are added for a chemically sound embedding, then
    stripped so the cloud reflects the molecular skeleton.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    mh = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mh, params) != 0:
        # retry with random coordinates as a fallback for awkward graphs
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mh, params) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mh)
    except Exception:  # noqa: BLE001 — minimization is best-effort; raw embed still usable
        pass
    mh = Chem.RemoveHs(mh)
    conf = mh.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mh.GetNumAtoms())],
                      dtype=float)
    return coords if coords.shape[0] >= 2 else None


def ligand_ph_features(smiles: list[str], maxdim: int = 1,
                       seed: int = 0xC0FFEE) -> tuple[np.ndarray, np.ndarray]:
    """SMILES list -> (X PH-feature matrix, valid-mask). Mirrors baseline.featurize.

    Rows for molecules that fail conformer embedding are dropped via the mask, so
    X aligns to labels[mask] exactly as the descriptor baseline does.
    """
    rows, mask = [], []
    for smi in smiles:
        cloud = conformer_cloud(smi, seed=seed)
        if cloud is None:
            mask.append(False)
            continue
        rows.append(ph_vector(cloud, maxdim=maxdim))
        mask.append(True)
    X = np.asarray(rows, dtype=float) if rows else np.empty((0, 0))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.asarray(mask, dtype=bool)
