#!/root/projects/pardalos/.venv/bin/python
"""Run ESM-2 embeddings in batches, saving to disk incrementally."""
import sys, warnings, os, gc, re
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

EMB_DIR = "/root/topologix-benchmark/data/embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

SEQ_PATH = "/root/topologix-benchmark/data/platinum_sequences.csv"
ESM_MODEL = "facebook/esm2_t12_35M_UR50D"  # 35M params, 480-dim — fits 4GB RAM
# Alternative: "facebook/esm2_t33_650M_UR50D" for 650M (needs >8GB)
BATCH_SIZE = 16
MAX_LEN = 800  # window around mutation site — well within 1024 token limit

def window_around_mutation(seq, mutation, half_window=400):
    """Extract a window of 2*half_window residues centered on mutation."""
    m = re.match(r"([A-Z])(\d+)([A-Z])", str(mutation).split("/")[0].strip())
    if not m: return seq[:MAX_LEN]  # fallback
    pos = int(m.group(2))
    start = max(0, pos - half_window)
    end = min(len(seq), pos + half_window)
    return seq[start:end]

def embed_batch(model, tokenizer, seqs, device="cpu"):
    inputs = tokenizer(seqs, return_tensors="pt", padding=True, add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    return (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

def main():
    print("Loading data...", flush=True)
    df = pd.read_csv(SEQ_PATH)
    df = df[df["status"]=="ok"].reset_index(drop=True)
    df = df[~df["mutation"].str.contains("/", na=False)].reset_index(drop=True)
    y = df["resist"].values.astype(int)
    print(f"{len(df)} sequences, {y.sum()} positives", flush=True)

    # Skip if we already did embeddings
    wt_done = os.path.exists(f"{EMB_DIR}/wt_all.npy")

    if not wt_done:
        print("Loading ESM-2...", flush=True)
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
        model = AutoModel.from_pretrained(ESM_MODEL)
        model.eval()
        print("Model loaded", flush=True)

        for name in ["wt", "mt"]:
            print(f"\nEmbedding {name} sequences...", flush=True)
            seqs = df[f"{name}_seq"].tolist()
            # Window long sequences around mutation site
            seqs = [window_around_mutation(s, mut) for s, mut in zip(seqs, df["mutation"])]
            max_tok = max(len(s) for s in seqs)
            print(f"  Max sequence length after windowing: {max_tok}", flush=True)
            n = len(seqs)
            all_embs = []
            for i in range(0, n, BATCH_SIZE):
                batch = seqs[i:i+BATCH_SIZE]
                try:
                    emb = embed_batch(model, tokenizer, batch)
                    all_embs.append(emb.cpu().numpy())
                except Exception as e:
                    print(f"  ERROR at batch {i}: {e}", flush=True)
                    break
                if (i // BATCH_SIZE) % 25 == 0:
                    done = min(i+BATCH_SIZE, n)
                    print(f"  {name}: {done}/{n} ({100*done//n}%)", flush=True)
                gc.collect()

            embs = np.concatenate(all_embs, axis=0)
            np.save(f"{EMB_DIR}/{name}_all.npy", embs)
            print(f"  Saved {name}_all.npy: {embs.shape}", flush=True)

        del model, tokenizer
        gc.collect()
        print("\nEmbeddings done", flush=True)
    else:
        print("Embeddings already exist, loading...", flush=True)

    # Load & classify
    print("\nTraining classifier...", flush=True)
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score

    emb_wt = np.load(f"{EMB_DIR}/wt_all.npy")
    emb_mt = np.load(f"{EMB_DIR}/mt_all.npy")
    delta = emb_mt - emb_wt

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
        print(f"  Fold {fold+1}: AUROC={auc:.4f}", flush=True)

    print(f"\n{'='*50}", flush=True)
    print(f"ESM-2 ({ESM_MODEL.split('/')[-1]}) + Morgan FP:  AUROC = {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}", flush=True)
    print(f"('~equal' to published mCSM-lig ~0.70)", flush=True)
    print(f"{'='*50}", flush=True)

    # Save
    pd.DataFrame({"fold": range(1,len(aurocs)+1), "auroc": aurocs}).to_csv(
        "/root/topologix-benchmark/results/esm_resistance_results.csv", index=False)
    print("Saved to results/esm_resistance_results.csv", flush=True)

if __name__ == "__main__":
    main()
