"""Per-ligand featurization: experimental dG, RDKit 2D descriptors (S), ESPH interface (T).

Runs once per ligand, not once per pair. 370 ligands across 15 targets.
"""

import glob
import os
import pickle
import sys
import time

import numpy as np
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

import esph

RDLogger.DisableLog("rdApp.*")

R = 0.0019872041   # kcal/mol/K
T_KELVIN = 297.0

DESC = [(n, f) for n, f in Descriptors.descList]


def to_dg(meas):
    """Experimental measurement -> binding free energy in kcal/mol.

    Returns None for any type/unit combination not explicitly handled, so unhandled
    records are dropped and counted rather than silently coerced.
    """
    t = str(meas.get("type", "")).lower()
    u = str(meas.get("unit", "")).strip()
    v = meas.get("value")
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if t in ("ic50", "ki"):
        if u == "uM":
            molar = v * 1e-6
        elif u == "nM":
            molar = v * 1e-9
        elif u == "M":
            molar = v
        else:
            return None
        if molar <= 0:
            return None
        return R * T_KELVIN * np.log(molar)
    if t == "pic50":
        return -R * T_KELVIN * np.log(10.0) * v
    return None


def descriptors(mol):
    out = np.zeros(len(DESC))
    for i, (_, fn) in enumerate(DESC):
        try:
            val = fn(mol)
            out[i] = val if np.isfinite(val) else 0.0
        except Exception:
            out[i] = 0.0
    return out


def heavy_atoms(mol):
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    return pos[keep], np.array([mol.GetAtomWithIdx(i).GetSymbol() for i in keep])


def main():
    rows = []
    dropped = {"no_dg": 0, "no_pose": 0, "no_interface": 0, "unmatched": 0}
    t0 = time.time()

    for ypath in sorted(glob.glob("data/*/ligands.yml")):
        tgt = os.path.basename(os.path.dirname(ypath))
        meta = yaml.safe_load(open(ypath))
        prot, prot_e = esph.parse_protein(os.path.join(os.path.dirname(ypath), "protein.pdb"))
        supp = Chem.SDMolSupplier(os.path.join(os.path.dirname(ypath), "ligands.sdf"),
                                  removeHs=False)
        n_ok = 0
        for mol in supp:
            if mol is None:
                continue
            name = mol.GetProp("_Name")
            if name not in meta:
                dropped["unmatched"] += 1
                continue
            dg = to_dg(meta[name]["measurement"])
            if dg is None:
                dropped["no_dg"] += 1
                continue
            if mol.GetNumConformers() == 0:
                dropped["no_pose"] += 1
                continue
            lig, lig_e = heavy_atoms(mol)
            iface = esph.interface_of(lig, prot, prot_e)
            if iface is None:
                dropped["no_interface"] += 1
                continue
            inter, inter_e = iface
            rows.append({
                "target": tgt,
                "name": name,
                "dg": dg,
                "S": descriptors(mol),
                "T": esph.esph_interface_vector(lig, lig_e, inter, inter_e),
                "n_lig_atoms": len(lig),
                "n_iface_atoms": len(inter),
            })
            n_ok += 1
        print(f"{tgt:10s} {n_ok:3d} ligands  ({time.time()-t0:6.1f}s elapsed)", flush=True)

    with open("results/features.pkl", "wb") as fh:
        pickle.dump({"rows": rows, "dropped": dropped,
                     "n_desc": len(DESC), "n_topo": esph.VECTOR_LEN}, fh)
    print(f"\ntotal {len(rows)} ligands | dropped {dropped} | "
          f"S dim {len(DESC)} | T dim {esph.VECTOR_LEN} | {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
