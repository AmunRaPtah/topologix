#!/root/projects/pardalos/.venv/bin/python
"""
Deploy ESM-2 embedding as Modal app, then call it.
Usage:
    MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... python modal_esm_deploy.py
"""
import os, sys, json

# Add pardalos to path for modal credentials
env_path = "/root/projects/pardalos/.env.local"
for line in open(env_path):
    if "MODAL_TOKEN_ID" in line or "MODAL_TOKEN_SECRET" in line:
        k, v = line.strip().split("=", 1)
        os.environ[k] = v

import modal

# ── Define the Modal app ──
app = modal.App("topologix-esm-embed")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "pandas", "numpy", "fair-esm")
)

@app.function(
    image=image,
    gpu="a10g",
    timeout=600,
)
def embed_and_classify(csv_data: str) -> dict:
    """Load ESM-2, embed all sequences, run RF classifier, return AUROC."""
    import pandas as pd, numpy as np, torch, re, gc, io
    from transformers import AutoTokenizer, AutoModel
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def window(s, mut, hw=400):
        m = re.match(r"([A-Z])(\d+)([A-Z])", str(mut).split("/")[0].strip())
        if not m: return s[:800]
        p = int(m.group(2))
        return s[max(0,p-hw):min(len(s),p+hw)]

    df = pd.read_csv(io.StringIO(csv_data))
    df = df[df["status"]=="ok"].reset_index(drop=True)
    df = df[~df["mutation"].str.contains("/", na=False)].reset_index(drop=True)
    y = df["resist"].values.astype(int)
    print(f"{len(df)} sequences, {y.sum()} positives")

    print("Loading ESM-2 650M...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to("cuda")
    model.eval()

    for name in ["wt", "mt"]:
        seqs = df[f"{name}_seq"].tolist()
        seqs = [window(s, mut) for s, mut in zip(seqs, df["mutation"])]
        print(f"Embedding {name} ({len(seqs)} seqs)...")
        all_embs = []
        for i in range(0, len(seqs), 16):
            batch = seqs[i:i+16]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            all_embs.append(emb.cpu().numpy())
            if (i // 16) % 5 == 0:
                print(f"  {min(i+16, len(seqs))}/{len(seqs)}")
        embs = np.concatenate(all_embs, axis=0)
        # Save to Modal volume or just keep in memory
        if name == "wt":
            emb_wt = embs
        else:
            emb_mt = embs
        gc.collect()

    delta = emb_mt - emb_wt
    print(f"Delta shape: {delta.shape}")

    # Morgan fingerprints
    fps = []
    for smi in df["lig_smiles"]:
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            fps.append(np.zeros(1024, "u1"))
        else:
            gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
            fps.append(np.frombuffer(gen.GetFingerprint(m).ToBitString().encode(), "u1") - ord("0"))
    X_fp = np.array(fps, dtype=np.float32)

    X_esm = np.column_stack([delta, X_fp])
    groups = df.index.values
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
    print(f"\nESM + Morgan FP: AUROC = {mean_auc:.4f} ± {std_auc:.4f}")
    return {"auroc_mean": mean_auc, "auroc_std": std_auc, "per_fold": aurocs}


if __name__ == "__main__":
    print("Deploying Modal app & running ESM embedding on A10G GPU...")
    with app.run():
        csv_path = "/root/topologix-benchmark/data/platinum_sequences.csv"
        with open(csv_path) as f:
            csv_data = f.read()
        result = embed_and_classify.remote(csv_data)
        print(f"\nResult: {json.dumps(result, indent=2)}")
