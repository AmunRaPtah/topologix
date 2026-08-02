#!/usr/bin/env python3
"""
Extract wt/mt protein sequences from local PDB structures for Platinum mutations.
Output: data/platinum_sequences.csv (mutation, wt_seq, mt_seq, lig_smiles, resist_label)

Run from /root/topologix-benchmark:
    python /root/topologix/scripts/extract_sequences.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, re, os, sys
from pathlib import Path
from Bio.PDB import PDBParser

STRUCTURES_DIR = Path("/root/topologix-benchmark/structures")
PLATINUM_PATH  = Path("/root/topologix-benchmark/data/platinum.csv")
OUTPUT_PATH    = Path("/root/topologix-benchmark/data/platinum_sequences.csv")

# ── amino acid 3-letter → 1-letter ──
AA3 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "HID":"H","HIE":"H","HIP":"H",  # histidine variants
    "ASH":"D","GLH":"E",            # protonated variants
    "CYX":"C",                      # disulfide cysteine
}

def parse_mutation(s):
    """'I84V' → ('I', '84', 'V') or 'D30N/N88D' → first mutation."""
    m = re.match(r"([A-Z])(\d+)([A-Z])", str(s).split("/")[0].strip())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None

def get_structure_seq(pdb_id, chain_id="A"):
    """Extract protein sequence from a PDB file as a string of 1-letter codes."""
    pdb_file = STRUCTURES_DIR / f"{pdb_id}.pdb"
    if not pdb_file.exists():
        pdb_file = STRUCTURES_DIR / f"{pdb_id.upper()}.pdb"
    if not pdb_file.exists():
        return None

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(pdb_file))
    except Exception:
        return None

    # Try chain A first, then first chain found
    for model in structure:
        for chain in model:
            cid = chain.get_id()
            if cid == chain_id or cid == chain_id.upper():
                seq = []
                for residue in chain:
                    if residue.get_id()[0] == " ":  # standard residue
                        resname = residue.get_resname()
                        if resname in AA3:
                            seq.append(AA3[resname])
                if seq:
                    return "".join(seq)
        # Fallback: first chain with any residues
        for chain in model:
            seq = []
            for residue in chain:
                if residue.get_id()[0] == " ":
                    resname = residue.get_resname()
                    if resname in AA3:
                        seq.append(AA3[resname])
            if seq:
                return "".join(seq)
    return None

def apply_mutation(seq, wt, pos, mt):
    """Apply point mutation to sequence at 1-indexed position. Returns None if wt doesn't match."""
    if pos < 1 or pos > len(seq):
        return None
    if seq[pos-1] != wt:
        return None  # sequence mismatch
    return seq[:pos-1] + mt + seq[pos:]

def main():
    df = pd.read_csv(PLATINUM_PATH)

    # Filter usable entries
    kw = pd.to_numeric(df["affin.k_wt"], errors="coerce")
    km = pd.to_numeric(df["affin.k_mt"], errors="coerce")
    ok = (kw > 0) & (km > 0)
    df = df[ok].copy()
    ratio = (km[ok] / kw[ok]).values
    df["resist"] = (ratio >= 10).astype(int)

    results = []
    found, missing_pdb, failed = 0, 0, 0

    for idx, row in df.iterrows():
        mut = parse_mutation(row["mutation"])
        if mut is None:
            failed += 1
            continue
        wt, pos, mt = mut

        # Try wt PDB first, then mt PDB, then the affin PDB
        pdb_ids = []
        wt_pdb = str(row.get("mut.wt_pdb", "")).strip()
        mt_pdb = str(row.get("mut.mt_pdb", "")).strip()
        affin_pdb = str(row.get("affin.pdb_id", "")).strip()

        if wt_pdb and wt_pdb != "NO" and wt_pdb != "nan":
            pdb_ids.append(("wt", wt_pdb))
        if mt_pdb and mt_pdb != "NO" and mt_pdb != "nan":
            pdb_ids.append(("mt", mt_pdb))
        if affin_pdb and affin_pdb != "NO" and affin_pdb != "nan":
            pdb_ids.append(("affin", affin_pdb))

        seq = None
        used_pdb = None
        for role, pid in pdb_ids:
            s = get_structure_seq(pid)
            if s:
                seq = s
                used_pdb = pid
                break

        if seq is None:
            found += 1
            results.append({
                "mutation": row["mutation"], "pdb_id": f"{pdb_ids[0][1] if pdb_ids else 'N/A'}",
                "wt_seq": "[PDB_NOT_FOUND]", "mt_seq": "[PDB_NOT_FOUND]",
                "lig_smiles": "", "resist": int(row["resist"]),
                "status": "missing_structure"
            })
            missing_pdb += 1
            continue

        # Apply mutation
        mt_seq = apply_mutation(seq, wt, pos, mt)
        if mt_seq is None:
            # Position mismatch — try the other PDB for the chain numbering
            seq2 = None
            for role, pid in pdb_ids:
                if pid == used_pdb: continue
                s2 = get_structure_seq(pid)
                if s2:
                    seq2 = s2
                    used_pdb = pid
                    break
            if seq2:
                mt_seq = apply_mutation(seq2, wt, pos, mt)
                if mt_seq:
                    seq = seq2

            if mt_seq is None:
                failed += 1
                results.append({
                    "mutation": row["mutation"], "pdb_id": used_pdb,
                    "wt_seq": f"[MISMATCH:{seq[:max(pos,5)]}...]",
                    "mt_seq": f"[FAILED: wt.{wt}@{pos} vs {seq[pos-1] if pos <= len(seq) else 'OUT'}]",
                    "lig_smiles": str(row.get("lig.canonical_smiles", "")),
                    "resist": int(row["resist"]),
                    "status": f"mismatch_wt={wt}_pos={pos}_seq={seq[pos-1] if pos<=len(seq) else 'OOB'}"
                })
                continue

        results.append({
            "mutation": row["mutation"], "pdb_id": used_pdb,
            "wt_seq": seq, "mt_seq": mt_seq,
            "lig_smiles": str(row.get("lig.canonical_smiles", "")),
            "resist": int(row["resist"]),
            "status": "ok"
        })
        found += 1

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Total entries processed: {len(results)}")
    print(f"  OK (sequences extracted): {(out['status']=='ok').sum()}")
    print(f"  Missing PDB structure: {(out['status']=='missing_structure').sum()}")
    mismatches = (~((out['status'] == 'ok') | (out['status'] == 'missing_structure'))).sum()
    print(f"  Mismatch/failed: {mismatches}")
    print(f"  Resistance positives: {out['resist'].sum()}/{len(out)}")
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"\nNow run:  cd /root/topologix-benchmark && python scripts/esm_resistance_pipeline.py")

if __name__ == "__main__":
    main()
