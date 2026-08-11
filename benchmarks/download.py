"""Fetch the OpenFE protein-ligand-benchmark: protein PDB, ligand SDF, experimental values."""
import os, urllib.request, sys

RAW = "https://raw.githubusercontent.com/OpenFreeEnergy/protein-ligand-benchmark/main/data"
TARGETS = ["cdk2","cdk8","cmet","eg5","hif2a","mcl1","p38","pde2","pfkfb3",
           "ptp1b","shp2","syk","thrombin","tnks2","tyk2"]
FILES = [("00_data/ligands.yml","ligands.yml"),
         ("02_ligands/ligands.sdf","ligands.sdf"),
         ("01_protein/crd/protein.pdb","protein.pdb")]

for t in TARGETS:
    d = f"data/{t}"; os.makedirs(d, exist_ok=True)
    for remote, local in FILES:
        p = os.path.join(d, local)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        try:
            urllib.request.urlretrieve(f"{RAW}/{t}/{remote}", p)
        except Exception as e:
            print(f"  MISS {t}/{local}: {e}", flush=True)
    have = [f for _, f in FILES if os.path.exists(os.path.join(d, f))]
    print(f"{t:10s} {len(have)}/3  " + " ".join(
        f"{f}={os.path.getsize(os.path.join(d,f))//1024}k" for f in have), flush=True)
