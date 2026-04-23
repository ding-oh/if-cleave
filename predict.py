"""Predict per-residue cleavage probabilities for a single PDB chain.

Runs ESM-IF1 on the structure, calls `propka3` for the 6 PROPKA columns,
applies per-fold standardization (stats saved alongside the checkpoints),
and ensembles the 4 pre-trained folds into a per-residue probability.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.extract_features import extract_if1_embeddings, load_if1_model
from model.model import IFCleave


IONIZABLE_AA = {"ASP", "GLU", "HIS", "TYR", "LYS", "ARG", "CYS", "N+", "C-"}
# 3-letter -> 1-letter for matching IF1 sequence output
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
PROPKA_DEFAULT = np.array([7.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def run_propka(pdb_path, workdir):
    base = os.path.basename(pdb_path)
    local = os.path.join(workdir, base)
    if os.path.abspath(local) != os.path.abspath(pdb_path):
        import shutil
        shutil.copy(pdb_path, local)
    subprocess.run(["propka3", base], cwd=workdir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    pka_path = os.path.join(workdir, base.replace(".pdb", ".pka"))
    if not os.path.exists(pka_path):
        raise RuntimeError(f"propka3 failed to produce {pka_path}")
    return pka_path


_ROW_RE = re.compile(
    r"^([A-Z][A-Z0-9+\-]{1,2})\s+(\d+)\s+([A-Z])\s+"
    r"([\-0-9.]+)\s+(\d+)\s*%\s+"
    r"([\-0-9.]+)\s+(\d+)\s+"
    r"([\-0-9.]+)\s+(\d+)\s+"
    r"([\-0-9.]+)\s+[A-Z+\-]+\s+(\d+)\s+[A-Z]"
)


def parse_pka_file(pka_path, chain_id):
    """Return {(res_name, res_num): (pka, buried, hbond_partner, desolv_reg, num_vol, desolv_eff)}."""
    rows = {}
    with open(pka_path) as fh:
        for line in fh:
            m = _ROW_RE.match(line.strip())
            if not m:
                continue
            res, num, ch = m.group(1), int(m.group(2)), m.group(3)
            if ch != chain_id or res not in IONIZABLE_AA:
                continue
            pka = float(m.group(4))
            buried = float(m.group(5)) / 100.0
            desolv_reg = float(m.group(6))
            num_vol = float(m.group(7))
            desolv_eff = float(m.group(8))
            hbond_partner = float(m.group(11))
            if (res, num) not in rows:
                rows[(res, num)] = (pka, buried, hbond_partner, desolv_reg, num_vol, desolv_eff)
    return rows


def extract_residue_list(pdb_path, chain_id):
    """Return [(res_name, res_num), ...] for standard amino acids in the chain, in order."""
    seen = set()
    out = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[21] != chain_id:
                continue
            res = line[17:20].strip()
            if res not in AA3TO1:
                continue
            try:
                num = int(line[22:26])
            except ValueError:
                continue
            key = (res, num)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def build_propka_features(pdb_path, chain_id, workdir):
    residues = extract_residue_list(pdb_path, chain_id)
    pka_path = run_propka(pdb_path, workdir)
    ion_rows = parse_pka_file(pka_path, chain_id)

    features = np.tile(PROPKA_DEFAULT, (len(residues), 1))
    for i, key in enumerate(residues):
        if key in ion_rows:
            features[i] = ion_rows[key]
    return features, residues


def predict(pdb_path, chain_id, ckpt_dir, device):
    fold_stats = torch.load(os.path.join(ckpt_dir, "feat_stats.pt"), weights_only=False)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "bilstm_fold*_best.pt")))
    if len(ckpts) != len(fold_stats):
        raise RuntimeError(f"checkpoint/stat count mismatch: {len(ckpts)} vs {len(fold_stats)}")

    model_if1, alphabet = load_if1_model(device)
    if1_emb, seq = extract_if1_embeddings(pdb_path, chain_id, model_if1, alphabet, device)

    with tempfile.TemporaryDirectory() as td:
        propka, residues = build_propka_features(pdb_path, chain_id, td)

    n = min(if1_emb.shape[0], propka.shape[0], len(seq))
    if1_emb = if1_emb[:n]
    propka = propka[:n]
    seq = seq[:n]
    residues = residues[:n]

    x_raw = torch.from_numpy(np.concatenate([if1_emb, propka], axis=1)).float()

    fold_probs = []
    for ckpt, stats in zip(ckpts, fold_stats):
        x = (x_raw - stats["mean"]) / stats["std"]
        m = IFCleave(input_dim=518, hidden_dim=256, dropout=0.4).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        m.eval()
        with torch.no_grad():
            logits = m(x.to(device))
        fold_probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.mean(fold_probs, axis=0), seq, residues


def main():
    p = argparse.ArgumentParser(description="Per-residue cleavage probability for a PDB chain")
    p.add_argument("--pdb", required=True, help="Path to PDB file")
    p.add_argument("--chain", required=True, help="Chain ID")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--output", default=None, help="Optional TSV output path")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs, seq, residues = predict(args.pdb, args.chain, args.ckpt_dir, device)

    header = f"# {args.pdb} chain {args.chain} | L={len(probs)} | mean_prob={probs.mean():.4f} | n_pos={(probs>args.threshold).sum()}"
    print(header)
    print("res_num\taa\tprob\tcleaved")
    for (res_name, res_num), aa, prob in zip(residues, seq, probs):
        print(f"{res_num}\t{aa}\t{prob:.4f}\t{int(prob>args.threshold)}")

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(header + "\n")
            fh.write("res_num\taa\tprob\tcleaved\n")
            for (res_name, res_num), aa, prob in zip(residues, seq, probs):
                fh.write(f"{res_num}\t{aa}\t{prob:.4f}\t{int(prob>args.threshold)}\n")
        print(f"saved {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
