#!/usr/bin/env python3
"""
Fetch protein sequences from UniProt for Platinum mutations.
Uses mut.uniprot IDs to get reference sequences, then applies mutations.
Gives us nearly 100% coverage vs the ~20% from local PDBs.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, re, urllib.request, time, sys
from pathlib import Path

PLATINUM_PATH = Path("/root/topologix-benchmark/data/platinum.csv")
OUTPUT_PATH   = Path("/root/topologix-benchmark/data/platinum_sequences.csv")
CACHE_DIR     = Path("/root/topologix-benchmark/data/uniprot_cache")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

AA3 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

def parse_first_mutation(s):
    """'I84V' → ('I', 84, 'V') or 'D30N/N88D' → first only."""
    m = re.match(r"([A-Z])(\d+)([A-Z])", str(s).split("/")[0].strip())
    if m: return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None

def fetch_uniprot_seq(uniprot_id):
    """Fetch canonical sequence from UniProt REST API."""
    id_clean = uniprot_id.split("-")[0].strip()
    cache_file = CACHE_DIR / f"{id_clean}.txt"
    if cache_file.exists():
        return cache_file.read_text().strip()

    url = f"https://rest.uniprot.org/uniprotkb/{id_clean}.fasta"
    req = urllib.request.Request(url, headers={"User-Agent": "Topologix/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            lines = resp.read().decode().strip().split("\n")
            seq = "".join(l for l in lines if not l.startswith(">"))
            cache_file.write_text(seq)
            time.sleep(0.2)
            return seq.strip()
    except Exception as e:
        return None

def pdb_to_seq(pdb_id):
    """Fetch protein sequence from RCSB PDB via REST API."""
    pid = pdb_id.lower()
    cache_file = CACHE_DIR / f"pdb_{pid}.txt"
    if cache_file.exists():
        return cache_file.read_text().strip()

    url = f"https://rest.uniprot.org/uniprotkb/?query=accession:{pid}&format=fasta"
    # Actually, let's try RCSB directly
    # Use PDB API to get entity sequence
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Topologix/1.0"})
    try:
        # Get the entity info for the polymer
        url2 = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pid}/1"
        req2 = urllib.request.Request(url2, headers={"User-Agent": "Topologix/1.0"})
        import json
        with urllib.request.urlopen(req2, timeout=10) as resp:
            data = json.loads(resp.read())
            seq = data["entity_poly"]["pdbx_seq_one_letter_code"]
            seq = seq.replace("\n", "").replace(" ", "")
            cache_file.write_text(seq)
            time.sleep(0.2)
            return seq
    except Exception:
        pass
    return None

def main():
    df = pd.read_csv(PLATINUM_PATH)

    # Filter usable entries
    kw = pd.to_numeric(df["affin.k_wt"], errors="coerce")
    km = pd.to_numeric(df["affin.k_mt"], errors="coerce")
    ok = (kw > 0) & (km > 0)
    df = df[ok].copy()
    ratio = (km[ok] / kw[ok]).values
    df["resist"] = (ratio >= 10).astype(int)

    # Collect unique UniProt IDs and PDB IDs
    uniprot_ids = df["mut.uniprot"].dropna().unique().tolist()
    uniprot_ids = [u for u in uniprot_ids if str(u).strip() and str(u) != "nan"]
    print(f"Unique UniProt IDs: {len(uniprot_ids)}")

    # Pre-fetch all UniProt sequences
    uniprot_cache = {}
    for i, uid in enumerate(uniprot_ids):
        seq = fetch_uniprot_seq(str(uid).strip())
        if seq:
            uniprot_cache[str(uid).strip()] = seq
        if (i+1) % 50 == 0:
            print(f"  Fetched {i+1}/{len(uniprot_ids)} UniProt sequences...")

    print(f"  Got {len(uniprot_cache)}/{len(uniprot_ids)} UniProt sequences")

    results = []
    ok_count, missing_seq, mismatch = 0, 0, 0

    for idx, row in df.iterrows():
        wt, pos, mt = parse_first_mutation(row["mutation"])
        if wt is None:
            mismatch += 1
            continue

        resist = int(row["resist"])
        lig_smi = str(row.get("lig.canonical_smiles", ""))

        # Strategy 1: UniProt
        uniprot = str(row.get("mut.uniprot", "")).strip()
        seq = None
        used_source = None

        if uniprot and uniprot != "nan" and uniprot in uniprot_cache:
            seq = uniprot_cache[uniprot]
            used_source = f"uniprot:{uniprot}"

        # Strategy 2: PDB (if no UniProt)
        if seq is None:
            for col in ["mut.wt_pdb", "mut.mt_pdb", "affin.pdb_id"]:
                pid = str(row.get(col, "")).strip()
                if pid and pid != "nan" and pid != "NO":
                    seq = pdb_to_seq(pid)
                    if seq:
                        used_source = f"pdb:{pid}"
                        break

        if seq is None:
            missing_seq += 1
            results.append({"mutation": row["mutation"], "pdb_id": str(row.get("affin.pdb_id","")),
                "wt_seq": "", "mt_seq": "", "lig_smiles": lig_smi,
                "resist": resist, "status": "no_sequence", "source": ""})
            continue

        # Apply mutation — need to find the right position
        # In UniProt sequences, mutation residue numbers may not match PDB numbering
        # Try multiple strategies:

        mt_seq = None

        # Strategy A: Exact match at position
        if pos <= len(seq) and seq[pos-1] == wt:
            mt_seq = seq[:pos-1] + mt + seq[pos:]

        # Strategy B: Search for the wt residue nearby (PDB numbering often differs)
        if mt_seq is None:
            # The mutation position in PDB may differ from UniProt position.
            # Try to find "wt" followed by "pos" residues that match
            # Search the sequence for the context
            # For now, try search for wt with a window
            wt_positions = [i for i, aa in enumerate(seq) if aa == wt]
            if len(wt_positions) == 1:
                # Only one occurrence — likely the right one
                mt_seq = seq[:wt_positions[0]] + mt + seq[wt_positions[0]+1:]

        # Strategy C: Try searching for nearby context from the mutation description
        if mt_seq is None:
            # Check if we can find the mutation in the full mutation string
            full_mut = str(row.get("mutation", ""))
            # Try multi-mutation: "D30N/N88D" — apply first only
            # Just in case position + 1 or -1 matches
            for offset in [0, -1, 1, -2, 2]:
                p = pos + offset
                if 1 <= p <= len(seq) and seq[p-1] == wt:
                    mt_seq = seq[:p-1] + mt + seq[p:]
                    break

        if mt_seq is None:
            mismatch += 1
            results.append({"mutation": row["mutation"], "pdb_id": str(row.get("affin.pdb_id","")),
                "wt_seq": seq[:min(pos+5, len(seq))],
                "mt_seq": f"[FAIL at pos {pos}: need {wt} found {seq[min(pos-1, len(seq)-1)] if pos <= len(seq) else 'OOB'}]",
                "lig_smiles": lig_smi, "resist": resist, "status": f"pos_mismatch",
                "source": used_source or ""})
            continue

        ok_count += 1
        results.append({"mutation": row["mutation"], "pdb_id": str(row.get("affin.pdb_id","")),
            "wt_seq": seq, "mt_seq": mt_seq,
            "lig_smiles": lig_smi, "resist": resist, "status": "ok",
            "source": used_source or ""})

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*50}")
    print(f"Total: {len(results)}")
    print(f"  OK:      {ok_count}")
    print(f"  Missing: {missing_seq} (no sequence source)")
    print(f"  Mismatch:{mismatch} (position conflict)")
    print(f"  Positives: {(out['status']=='ok') & (out['resist']==1)}")
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
