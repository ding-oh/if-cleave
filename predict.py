"""Predict per-residue cleavage probabilities for a single PDB chain.

Runs ESM-IF1 on the structure, calls `propka3` for the 6 PROPKA columns,
applies per-fold standardization (stats saved alongside the checkpoints),
and ensembles the 4 pre-trained folds into a per-residue probability.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.extract_features import extract_if1_embeddings, load_if1_model
from data.propka import build_propka_features as _build_propka_slice
from data.propka import chain_residue_order
from model.model import IFCleave


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


def build_propka_features(pdb_path, chain_id, workdir):
    pka_path = run_propka(pdb_path, workdir)
    propka = _build_propka_slice(pdb_path, pka_path, chain_id)
    residues = chain_residue_order(pdb_path, chain_id)
    return propka, residues


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
