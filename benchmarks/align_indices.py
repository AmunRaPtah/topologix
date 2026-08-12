"""Recompute which source indices survive conformer embedding, to align the cached features.

featurize.py appended rows in source order and dropped 11 molecules without recording which.
Embedding is deterministic and cheap on its own, so the kept-index list is recoverable.
"""
import pickle
from multiprocessing import Pool
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem

SRC = "/root/topologix/repo/benchmarks/data/herg_karim.tab"

def ok(arg):
    idx, smi = arg
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return (idx, False)
    m = Chem.Mol(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 0; p.numThreads = 1
    if AllChem.EmbedMolecule(m, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(m, p) != 0: return (idx, False)
    if m.GetNumConformers() == 0: return (idx, False)
    elems = np.array([a.GetSymbol() for a in m.GetAtoms()])
    return (idx, int((elems != "H").sum()) >= 2)

if __name__ == "__main__":
    df = pd.read_csv(SRC, sep="\t"); df["Drug"] = df["Drug"].str.strip('"')
    with Pool(4) as pool:
        res = pool.map(ok, list(enumerate(df.Drug)), chunksize=64)
    kept = [i for i, good in sorted(res) if good]
    feats = pickle.load(open("results/features.pkl", "rb"))
    n_rows = len(feats["rows"])
    print(f"kept indices: {len(kept)} | cached feature rows: {n_rows} | "
          f"match: {len(kept) == n_rows}")
    assert len(kept) == n_rows, "alignment failed; do not proceed"
    # verify alignment on labels, which are stored in both
    ys_src = df.Y.values[kept]
    ys_feat = np.array([r["y"] for r in feats["rows"]])
    print("label agreement:", float((ys_src == ys_feat).mean()))
    assert (ys_src == ys_feat).all(), "label mismatch; alignment is wrong"
    pickle.dump(kept, open("results/kept_indices.pkl", "wb"))
    print("wrote results/kept_indices.pkl")
