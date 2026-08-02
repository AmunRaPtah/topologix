#!/usr/bin/env python3
"""
ESM-2 + Morgan fingerprint → resistance prediction (AUROC ~0.75).
Reproduce the result from the old server.

Usage:
    /tmp/topo-env/bin/python /root/topologix/scripts/esm_pipeline.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, numpy as np, pandas as pd, torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# ── Config ──
SEQ_PATH    = "/root/topologix-benchmark/data/platinum_sequences.csv"
ESM_MODEL   = "facebook/esm2_t33_650M_UR50D"  # 650M params, 1280-dim embeddings
N_SPLITS    = 5
TEST_SIZE   = 0.25
BATCH_SIZE  = 4  # CPU batch size (reduce if OOM)

def morgan_fp(smi, radius=2, nbits=1024):
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return None
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    return np.frombuffer(gen.GetFingerprint(m).ToBitString().encode(), "u1") - ord("0")

@torch.no_grad()
def get_embeddings(model, tokenizer, sequences, batch_size=BATCH_SIZE):
    """Get per-sequence mean-pooled ESM-2 embeddings."""
    model.eval()
    device = next(model.parameters()).device
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        # Mean pool over non-padding tokens
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)

def main():
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading sequences from {SEQ_PATH}...")
    df = pd.read_csv(SEQ_PATH)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    print(f"  {len(df)} mutation pairs with sequences")

    # We need to filter to only single-point mutations
    # Multi-mutations like "D30N/N88D" can't be handled by simple apply
    from_top = df[df["mutation"].str.contains("/", na=False)]
    if len(from_top) > 0:
        print(f"  Dropping {len(from_top)} multi-mutations (contain '/')")
        df = df[~df["mutation"].str.contains("/", na=False)].reset_index(drop=True)
    print(f"  {len(df)} single-point mutations")
    print(f"  Positives: {df['resist'].sum()}/{len(df)} ({100*df['resist'].sum()/len(df):.1f}%)")

    # double-check wt_seq matches mutation
    wt_ok = 0
    for _, row in df.iterrows():
        wt_aa = row["mutation"][0]  # first char is wt amino acid
        pos = int("".join(c for c in row["mutation"][1:-1] if c.isdigit()))
        mt_aa = row["mutation"][-1]
        if pos <= len(row["wt_seq"]) and row["wt_seq"][pos-1] == wt_aa:
            wt_ok += 1

    print(f"  Sequences with correct wt at position: {wt_ok}/{len(df)}")

    # ── Compute Morgan fingerprints first (fast) ──
    print("\nComputing Morgan fingerprints...")
    fps = []
    for smi in df["lig_smiles"]:
        fp = morgan_fp(smi)
        fps.append(fp if fp is not None else np.zeros(1024, "u1"))
    X_fp = np.array(fps, dtype=np.float32)

    # ── Morgan-only baseline ──
    y = df["resist"].values.astype(int)
    # Need groups for split
    df["_group"] = df["mutation"]  # use mutation as group (simplest)
    groups = df["_group"].values

    gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=42)
    fp_aurocs = []
    for tr, te in gss.split(X_fp, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0, class_weight="balanced")
        clf.fit(X_fp[tr], y[tr])
        fp_aurocs.append(roc_auc_score(y[te], clf.predict_proba(X_fp[te])[:, 1]))
    print(f"  Morgan FP baseline AUROC: {np.mean(fp_aurocs):.4f} ± {np.std(fp_aurocs):.4f}")

    # ── Simple structural baseline (like resistance_baseline.py) ──
    # Reconstruct simple features from mutation properties
    KD = dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,
              L=3.8,K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
    VOL = dict(A=88.6,R=173.4,N=114.1,D=111.1,C=108.5,Q=143.8,E=138.4,G=60.1,H=153.2,
               I=166.7,L=166.7,K=168.6,M=162.9,F=189.9,P=112.7,S=89.0,T=116.1,W=227.8,Y=193.6,V=140.0)

    X_simple = []
    for _, row in df.iterrows():
        wt = row["mutation"][0]
        mt = row["mutation"][-1]
        X_simple.append([
            KD.get(wt, 0) - KD.get(mt, 0),   # hydropathy change
            VOL.get(wt, 0) - VOL.get(mt, 0),  # volume change
        ])
    X_simple = np.array(X_simple, dtype=np.float32)

    simple_aurocs = []
    for tr, te in gss.split(X_simple, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0, class_weight="balanced")
        clf.fit(X_simple[tr], y[tr])
        simple_aurocs.append(roc_auc_score(y[te], clf.predict_proba(X_simple[te])[:, 1]))
    print(f"  Simple features AUROC: {np.mean(simple_aurocs):.4f} ± {np.std(simple_aurocs):.4f}")

    # ── Load ESM-2 ──
    print(f"\nLoading ESM-2 ({ESM_MODEL}) on CPU...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = AutoModel.from_pretrained(ESM_MODEL)
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # ── Get wt and mt embeddings ──
    print(f"\nExtracting wt embeddings ({len(df)} sequences)...")
    t0 = time.time()
    emb_wt = get_embeddings(model, tokenizer, df["wt_seq"].tolist())
    print(f"  Done in {time.time()-t0:.1f}s ({len(emb_wt)} x {emb_wt.shape[1]})")

    print(f"\nExtracting mt embeddings ({len(df)} sequences)...")
    t0 = time.time()
    emb_mt = get_embeddings(model, tokenizer, df["mt_seq"].tolist())
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Delta embedding ──
    delta = emb_mt - emb_wt

    # ── Combined feature: ESM delta + Morgan FP ──
    X_esm = np.column_stack([delta, X_fp])
    print(f"\nFeature matrix: {X_esm.shape}")

    # ── Cross-validation ──
    print(f"\n{'='*60}")
    print(f"5-fold grouped CV (test_size={TEST_SIZE})")
    print(f"{'='*60}")

    all_results = {"fold": [], "esm+fp": [], "fp_only": [], "simple": []}

    for fold, (tr, te) in enumerate(gss.split(X_esm, y, groups)):
        if len(np.unique(y[te])) < 2:
            print(f"  Fold {fold+1}: SKIP (only 1 class in test)")
            continue

        # ESM + FP
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=fold, class_weight="balanced")
        clf.fit(X_esm[tr], y[tr])
        p_esm = clf.predict_proba(X_esm[te])[:, 1]
        auc_esm = roc_auc_score(y[te], p_esm)

        # FP only (for comparison)
        clf2 = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=fold, class_weight="balanced")
        clf2.fit(X_fp[tr], y[tr])
        p_fp = clf2.predict_proba(X_fp[te])[:, 1]
        auc_fp = roc_auc_score(y[te], p_fp)

        # Simple features
        clf3 = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=fold, class_weight="balanced")
        clf3.fit(X_simple[tr], y[tr])
        p_simple = clf3.predict_proba(X_simple[te])[:, 1]
        auc_simple = roc_auc_score(y[te], p_simple)

        all_results["fold"].append(fold+1)
        all_results["esm+fp"].append(auc_esm)
        all_results["fp_only"].append(auc_fp)
        all_results["simple"].append(auc_simple)

        n_pos = y[te].sum()
        print(f"  Fold {fold+1}: ESM+FP={auc_esm:.4f}  FP={auc_fp:.4f}  Simple={auc_simple:.4f}  (pos={n_pos}/{len(y[te])})")

    print(f"\n{'='*60}")
    if all_results["esm+fp"]:
        print(f"  ESM-2 + Morgan FP:  AUROC = {np.mean(all_results['esm+fp']):.4f} ± {np.std(all_results['esm+fp']):.4f}")
        print(f"  Morgan FP only:     AUROC = {np.mean(all_results['fp_only']):.4f} ± {np.std(all_results['fp_only']):.4f}")
        print(f"  Simple features:    AUROC = {np.mean(all_results['simple']):.4f} ± {np.std(all_results['simple']):.4f}")
        print(f"  (Published mCSM-lig ~0.70)")
    print(f"{'='*60}")

    # Save
    out = pd.DataFrame(all_results)
    out.to_csv("/root/topologix-benchmark/results/esm_resistance_results.csv", index=False)
    print(f"\nSaved results to results/esm_resistance_results.csv")

if __name__ == "__main__":
    main()
