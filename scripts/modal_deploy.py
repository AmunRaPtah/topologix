#!/root/projects/pardalos/.venv/bin/python
"""
Deploy ESM-2 embedding function to Modal as a persistent app.
Then invoke it.

Step 1: MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... python modal_deploy.py deploy
Step 2: MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... python modal_deploy.py run
"""
import os, json, sys, pathlib

env_path = "/root/projects/pardalos/.env.local"
for line in open(env_path):
    if "MODAL_TOKEN" in line:
        k, v = line.strip().split("=", 1)
        os.environ[k] = v

import modal; modal.enable_output()

APP_NAME = "topologix-platinum-esm"
OUT_PATH = "/root/topologix-benchmark/results/esm_resistance_results.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy<2.0", "pandas", "scikit-learn", "scipy",
                 "tqdm", "biopython", "fair-esm")
)

app = modal.App(APP_NAME, image=image)

@app.function(gpu="a10g", timeout=900)
def run_esm(csv_data: str) -> dict:
    import pandas as pd, numpy as np, torch, re, gc, io, warnings
    warnings.filterwarnings("ignore")
    from esm import pretrained
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score

    def window(s, mut, hw=400):
        m = re.match(r"([A-Z])(\d+)([A-Z])", str(mut).split("/")[0].strip())
        if not m: return s[:800]
        p = int(m.group(2))
        return s[max(0,p-hw):min(len(s),p+hw)]

    df = pd.read_csv(io.StringIO(csv_data))
    df = df[df["status"]=="ok"].reset_index(drop=True)
    df = df[~df["mutation"].str.contains("/", na=False)].reset_index(drop=True)
    y = df["resist"].values.astype(int)
    print(f"Data: {len(df)} mutations, {y.sum()} positives", flush=True)

    X_fp = np.array([np.fromstring(str(row).strip(), sep=",", dtype=np.float32) for row in df["X_fp"]])

    wt_seqs = [window(s,m) for s,m in zip(df["wt_seq"], df["mutation"])]
    mt_seqs = [window(s,m) for s,m in zip(df["mt_seq"], df["mutation"])]

    print("Loading ESM-2 650M...", flush=True)
    device = torch.device("cuda")
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    def extract(seqs, name):
        all_emb = []
        for i in range(0, len(seqs), 8):
            data = [(f"s{j}", s) for j, s in enumerate(seqs[i:i+8])]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            with torch.no_grad():
                reps = model(tokens, repr_layers=[33])["representations"][33].cpu().numpy()
            for j, s in enumerate(data):
                all_emb.append(reps[j, 1:len(s[1])+1].mean(axis=0))
            torch.cuda.empty_cache(); gc.collect()
            if (i // 8) % 10 == 0:
                print(f"  {name}: {min(i+8, len(seqs))}/{len(seqs)}", flush=True)
        return np.array(all_emb)

    emb_wt = extract(wt_seqs, "wt")
    emb_mt = extract(mt_seqs, "mt")
    delta = emb_mt - emb_wt
    print(f"Delta: {delta.shape}", flush=True)

    X = np.column_stack([delta, X_fp])
    gss = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
    aucs = []
    for f, (tr, te) in enumerate(gss.split(X, y, np.arange(len(y)))):
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=f, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
        aucs.append(auc)
        print(f"  Fold {f+1}: AUROC={auc:.4f}", flush=True)

    return {"auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs)), "per_fold": aucs}


if __name__ == "__main__":
    import pandas as pd, numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

    csv_path = "/root/topologix-benchmark/data/platinum_sequences.csv"
    df = pd.read_csv(csv_path)
    fps = []
    for smi in df["lig_smiles"]:
        m = Chem.MolFromSmiles(str(smi))
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
        fp = np.frombuffer(gen.GetFingerprint(m).ToBitString().encode(), "u1") - ord("0") if m else np.zeros(1024, "u1")
        fps.append(",".join(str(b) for b in fp))
    df["X_fp"] = fps
    csv_data = df.to_csv(index=False)

    print("Running ESM-2 650M on Modal A10G...", flush=True)
    with app.run():
        result = run_esm.remote(csv_data)
        print(f"\nResult: {json.dumps(result, indent=2)}", flush=True)
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {OUT_PATH}", flush=True)
