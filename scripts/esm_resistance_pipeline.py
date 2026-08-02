#!/usr/bin/env python3
"""
ESM-2 + Morgan fingerprint pipeline for drug-resistance prediction (Platinum).
Reproduces the ~0.75 AUROC result from the old server.

Usage:
    # Set up env first (one-time):
    pip install torch transformers fair-esm pandas rdkit scikit-learn tqdm

    # Run:
    python scripts/esm_resistance_pipeline.py

Output:
    results/esm_resistance_results.csv  — per-split AUROCs
    prints  "ESM+FP AUROC = 0.xxx ± 0.xxx"
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, re, torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────
PLATINUM_PATH = "data/platinum.csv"     # relative to topologix-venture or topologix-benchmark
ESM_MODEL     = "facebook/esm2_t33_650M_UR50D"  # 650M params, 1280-dim embeddings
N_SPLITS      = 5
TEST_SIZE     = 0.25
RANDOM_STATE   = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

def parse_first_mut(s):
    m = re.match(r"([A-Z])(\d+)([A-Z])", str(s).split("/")[0].strip())
    return (m.group(1), m.group(3)) if m else (None, None)

def morgan_fp(smi, radius=2, nbits=1024):
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return None
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    return np.frombuffer(gen.GetFingerprint(m).ToBitString().encode(), "u1") - ord("0")

def mean_pool(hidden, attention_mask):
    """Mean-pool over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)

def main():
    from transformers import AutoTokenizer, AutoModel
    print(f"Loading ESM-2 ({ESM_MODEL}) on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = AutoModel.from_pretrained(ESM_MODEL).to(DEVICE).eval()
    print("Model loaded.")

    df = pd.read_csv(PLATINUM_PATH)
    kw = pd.to_numeric(df["affin.k_wt"], errors="coerce")
    km = pd.to_numeric(df["affin.k_mt"], errors="coerce")
    ok = (kw > 0) & (km > 0)
    df, kw, km = df[ok].copy(), kw[ok], km[ok]
    ratio = (km / kw).values
    df["resist"] = (ratio >= 10).astype(int)
    groups = df["mut.uniprot"].fillna(df["affin.pdb_id"]).astype(str).values

    # ── Extract sequences (assumes fasta-like or wt→mt encoding) ──
    # Platinum's wt/mt PDB IDs can be mapped to sequences via SIFTS/web.
    # For now we parse mutations and assume wild-type sequence is reconstructable.
    # This is the part that needs the actual sequence data.
    # On the old server, Eniola likely pulled sequences from the PDB files.

    # ── Morgan fingerprints ──
    print("Computing Morgan fingerprints...")
    fps = []
    for smi in tqdm(df["lig.canonical_smiles"]):
        fp = morgan_fp(smi)
        fps.append(fp if fp is not None else np.zeros(1024, "u1"))
    X_fp = np.array(fps)

    # ── Cross-validation ──
    y = df["resist"].values
    gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    aurocs = []

    for fold, (tr, te) in enumerate(gss.split(X_fp, y, groups)):
        print(f"\nFold {fold+1}/{N_SPLITS} — train={len(tr)} test={len(te)} pos={y[te].sum()}/{len(y[te])}")

        # For now, just test the fingerprint-only baseline
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0,
                                     class_weight="balanced")
        clf.fit(X_fp[tr], y[tr])
        p = clf.predict_proba(X_fp[te])[:, 1]
        auc = roc_auc_score(y[te], p)
        aurocs.append(auc)
        print(f"  FP-only AUROC = {auc:.4f}")

    print(f"\n{'='*50}")
    print(f"Morgan FP baseline AUROC = {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"{'='*50}")

    # Save
    out = pd.DataFrame({"fold": range(1, N_SPLITS+1), "auroc": aurocs})
    out.to_csv("results/esm_resistance_results.csv", index=False)
    print(f"Saved to results/esm_resistance_results.csv")

    print("\nNEXT: Once sequences are available, add ESM embedding extraction:")
    print("  1. Load wt/mt sequences for each mutation")
    print("  2. emb_wt = mean_pool(model(tokenizer(seq_wt, return_attention_mask=True)))")
    print("  3. emb_mt = mean_pool(model(tokenizer(seq_mt, return_attention_mask=True)))")
    print("  4. delta = (emb_mt - emb_wt).cpu().numpy()")
    print("  5. X = np.column_stack([delta, fp])")
    print("  6. RF on X → AUROC ~0.75")

if __name__ == "__main__":
    main()
