#!/root/projects/pardalos/.venv/bin/python
"""
Deploy to Modal GPU: run ESM-2 650M inference on A10G, save embeddings.
Usage:
    cd /root/topologix-benchmark
    python /root/topologix/scripts/modal_esm.py
"""
import os, sys, modal, pathlib

# ── Modal app ──
app = modal.App("topologix-esm")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "pandas", "numpy", "fair-esm")
)

@app.function(
    image=image,
    gpu="a10g",
    timeout=600,
    secrets=[
        modal.Secret.from_dict({
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        })
    ],
)
def embed_sequences(csv_path: str, out_dir: str) -> dict:
    """Load ESM-2, embed all wt/mt sequences, save to disk."""
    import pandas as pd, numpy as np, torch, re, gc
    from transformers import AutoTokenizer, AutoModel

    def window(s, mut, hw=400):
        m = re.match(r"([A-Z])(\d+)([A-Z])", str(mut).split("/")[0].strip())
        if not m: return s[:800]
        p = int(m.group(2))
        return s[max(0,p-hw):min(len(s),p+hw)]

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df[df["status"]=="ok"].reset_index(drop=True)
    df = df[~df["mutation"].str.contains("/", na=False)].reset_index(drop=True)

    print(f"Loading ESM-2 650M on GPU...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to("cuda")
    model.eval()

    for name in ["wt", "mt"]:
        seqs = df[f"{name}_seq"].tolist()
        seqs = [window(s, mut) for s, mut in zip(seqs, df["mutation"])]
        print(f"Embedding {name}: {len(seqs)} sequences (batch_size=16)...")

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
                done = min(i+16, len(seqs))
                print(f"  {done}/{len(seqs)} ({100*done//len(seqs)}%)")
            gc.collect()

        embs = np.concatenate(all_embs, axis=0)
        out_path = f"{out_dir}/{name}_all.npy"
        os.makedirs(out_dir, exist_ok=True)
        np.save(out_path, embs)
        print(f"Saved {out_path}: {embs.shape}")

    return {"status": "done", "n": len(df)}


if __name__ == "__main__":
    csv_path = os.path.abspath("/root/topologix-benchmark/data/platinum_sequences.csv")
    out_dir  = os.path.abspath("/root/topologix-benchmark/data/embeddings")

    # Set credentials from Pardalos .env.local
    env_path = "/root/projects/pardalos/.env.local"
    for line in open(env_path):
        if line.startswith("MODAL_TOKEN_ID"):
            os.environ["MODAL_TOKEN_ID"] = line.split("=",1)[1].strip()
        if line.startswith("MODAL_TOKEN_SECRET"):
            os.environ["MODAL_TOKEN_SECRET"] = line.split("=",1)[1].strip()
        if line.startswith("HF_TOKEN"):
            os.environ["HF_TOKEN"] = line.split("=",1)[1].strip()

    print(f"Deploying Modal ESM task...")
    result = embed_sequences.remote(csv_path, out_dir)
    print(f"Modal task result: {result}")
