#!/root/projects/pardalos/.venv/bin/python
"""
HIV-ESM-2 adapted for Platinum drug-resistance benchmark.
Runs ESM-2 650M on A10G via Modal, saves embeddings, trains classifier.

Usage:
    cd /root/topologix-benchmark
    MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... python /root/topologix/scripts/modal_hiv_esm.py
"""
import os, json, sys

env_path = "/root/projects/pardalos/.env.local"
for line in open(env_path):
    if "MODAL_TOKEN" in line:
        k, v = line.strip().split("=", 1)
        os.environ[k] = v

import modal; modal.enable_output()
import logging; logging.basicConfig(level=logging.INFO)

app = modal.App("topologix-platinum-esm")

# Debian slim + pip install torch (compatible with Python 3.12 on client)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy<2.0", "pandas", "scikit-learn", "scipy",
                 "tqdm", "biopython", "fair-esm")
)

@app.function(
    image=image,
    gpu="a10g",
    timeout=900,
    retries=1,
)
def run_platinum_esm(csv_data: str) -> dict:
    """Full pipeline: ESM-2 650M embeddings → attention pooling → RF classifier."""
    import pandas as pd, numpy as np, torch, re, gc, io, warnings
    warnings.filterwarnings("ignore")

    from esm import FastaBatchedDataset, pretrained
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score

    def window(s, mut, hw=400):
        m = re.match(r"([A-Z])(\d+)([A-Z])", str(mut).split("/")[0].strip())
        if not m: return s[:800]
        p = int(m.group(2))
        return s[max(0,p-hw):min(len(s),p+hw)]

    def mean_pool(embeddings):
        return np.array([e.mean(axis=0) for e in embeddings])

    # Load data
    df = pd.read_csv(io.StringIO(csv_data))
    df = df[df["status"]=="ok"].reset_index(drop=True)
    df = df[~df["mutation"].str.contains("/", na=False)].reset_index(drop=True)
    y = df["resist"].values.astype(int)
    print(f"Data: {len(df)} mutations, {y.sum()} positives ({100*y.sum()/len(df):.1f}%)")

    # Parse pre-computed fingerprints from CSV
    X_fp = np.array([np.fromstring(str(row), sep=",", dtype=np.float32) for row in df["X_fp"]])

    # Window sequences
    wt_seqs = [window(s,m) for s,m in zip(df["wt_seq"], df["mutation"])]
    mt_seqs = [window(s,m) for s,m in zip(df["mt_seq"], df["mutation"])]

    # Load ESM-2 650M via esm package
    print("Loading ESM-2 650M...")
    device = torch.device("cuda")
    model_name = "esm2_t33_650M_UR50D"
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    def extract_pooled(sequences, name="", batch_size=8):
        """Extract mean-pooled ESM-2 embeddings."""
        all_pooled = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            data = [(f"s{j}", s) for j, s in enumerate(batch)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            with torch.no_grad():
                results = model(tokens, repr_layers=[33], return_contacts=False)
                reps = results["representations"][33].cpu().numpy()
            for j, s in enumerate(batch):
                # Strip BOS/EOS tokens
                seq_emb = reps[j, 1:len(s)+1, :]
                pooled = seq_emb.mean(axis=0)  # mean pool
                all_pooled.append(pooled)
            torch.cuda.empty_cache()
            if (i // batch_size) % 10 == 0:
                print(f"  {name}: {min(i+batch_size, len(sequences))}/{len(sequences)}", flush=True)
            gc.collect()
        return np.array(all_pooled)

    emb_wt = extract_pooled(wt_seqs, "wt")
    emb_mt = extract_pooled(mt_seqs, "mt")
    delta = emb_mt - emb_wt
    print(f"Delta shape: {delta.shape}")

    X_esm = np.column_stack([delta, X_fp])
    print(f"Feature matrix: {X_esm.shape}")

    # Cross-validation
    groups = np.arange(len(y))  # per-mutation groups
    gss = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
    aurocs = []
    for fold, (tr, te) in enumerate(gss.split(X_esm, y, groups)):
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=fold, class_weight="balanced")
        clf.fit(X_esm[tr], y[tr])
        p = clf.predict_proba(X_esm[te])[:, 1]
        auc = roc_auc_score(y[te], p)
        aurocs.append(auc)
        print(f"  Fold {fold+1}: AUROC={auc:.4f}")

    mean_auc = float(np.mean(aurocs))
    std_auc = float(np.std(aurocs))
    print(f"\n{'='*50}")
    print(f"ESM-2 650M + Morgan FP on Platinum:")
    print(f"  AUROC = {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  (HIV-ESM-2 benchmark: 0.968 on HIVDB)")
    print(f"  (Published mCSM-lig: ~0.70)")
    print(f"{'='*50}")

    return {"auroc_mean": mean_auc, "auroc_std": std_auc, "per_fold": aurocs}


if __name__ == "__main__":
    import pandas as pd, numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

    csv_path = "/root/topologix-benchmark/data/platinum_sequences.csv"
    df = pd.read_csv(csv_path)
    # Pre-compute Morgan fingerprints
    fps = []
    for smi in df["lig_smiles"]:
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            fps.append(np.zeros(1024, "u1"))
        else:
            gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps.append(np.frombuffer(gen.GetFingerprint(m).ToBitString().encode(), "u1") - ord("0"))
    df["X_fp"] = [",".join(str(b) for b in fp) for fp in fps]
    csv_data = df.to_csv(index=False)

    print("Deploying to Modal GPU (A10G) — building image and running...")
    with app.run():
        result = run_platinum_esm.remote(csv_data)
        print(f"\nResult: {json.dumps(result, indent=2)}")
        # Save
        out_path = "/root/topologix-benchmark/results/esm_resistance_results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out_path}")
