"""3D conformers, RDKit descriptors, and four topological encodings. Parallel, checkpointed."""
import os, pickle, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import encodings_alt as E

warnings.filterwarnings("ignore")
DESC = Descriptors._descList
SRC = "/root/topologix/repo/benchmarks/data/herg_karim.tab"
NPROC = 4
CHUNK = 500

def descriptors(mol):
    v = np.zeros(len(DESC))
    for i, (_n, fn) in enumerate(DESC):
        try:
            x = fn(mol); v[i] = x if np.isfinite(x) else 0.0
        except Exception: v[i] = 0.0
    return v

def one(arg):
    idx, smi, y = arg
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return ("bad_smiles", None)
    m = Chem.Mol(mol)                      # heavy-atom embedding, see addendum
    p = AllChem.ETKDGv3(); p.randomSeed = 0; p.numThreads = 1
    if AllChem.EmbedMolecule(m, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(m, p) != 0: return ("no_conformer", None)
    if m.GetNumConformers() == 0: return ("no_conformer", None)
    P = m.GetConformer().GetPositions()
    elems = np.array([a.GetSymbol() for a in m.GetAtoms()])
    keep = elems != "H"
    P, elems = P[keep], elems[keep]
    if len(P) < 2: return ("no_conformer", None)
    blocks = E.encode_all_channels(P, elems)
    try: scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception: scaf = ""
    return ("ok", {"y": int(y), "scaffold": scaf or f"__none_{idx}",
                   "A": descriptors(mol), **blocks})

def main():
    df = pd.read_csv(SRC, sep="\t"); df["Drug"] = df["Drug"].str.strip('"')
    args = [(i, r.Drug, r.Y) for i, r in enumerate(df.itertuples())]
    rows, dropped = [], {"bad_smiles": 0, "no_conformer": 0}
    t0 = time.time()
    with Pool(NPROC) as pool:
        for n, (status, rec) in enumerate(pool.imap(one, args, chunksize=32), 1):
            if status == "ok": rows.append(rec)
            else: dropped[status] += 1
            if n % CHUNK == 0:
                el = time.time() - t0
                print(f"  {n}/{len(args)} kept {len(rows)} {el/60:.1f} min "
                      f"(eta {el/n*(len(args)-n)/60:.0f} min)", flush=True)
    with open("results/features.pkl", "wb") as fh:
        pickle.dump({"rows": rows, "dropped": dropped,
                     "dims": {k: len(rows[0][k]) for k in ("A","V","X","L","S")}}, fh)
    print(f"\nkept {len(rows)}/{len(args)} | dropped {dropped} | "
          f"dims { {k: len(rows[0][k]) for k in ('A','V','X','L','S')} } | "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
