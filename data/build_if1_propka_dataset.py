import argparse
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm


def load_if1_model(device):
    try:
        import esm  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing fair-esm. Install with: pip install fair-esm"
        ) from exc

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)
    return model, alphabet


def extract_if1_embeddings(pdb_path, chain_id, model, alphabet, device):
    try:
        from esm.inverse_folding.util import (
            load_structure,
            extract_coords_from_structure,
            CoordBatchConverter,
        )
    except Exception as exc:
        raise RuntimeError(
            "Missing inverse_folding utils. Make sure fair-esm is installed correctly."
        ) from exc

    structure = load_structure(pdb_path, chain_id)
    coords, seq = extract_coords_from_structure(structure)

    batch_converter = CoordBatchConverter(alphabet)
    confidence = np.ones((len(seq),), dtype=np.float32)
    batch = [(coords, confidence, seq)]
    coords_batch, confidence, seqs, tokens, padding_mask = batch_converter(batch)

    coords_batch = coords_batch.to(device)
    confidence = confidence.to(device)
    padding_mask = padding_mask.to(device)
    tokens = tokens.to(device)

    with torch.no_grad():
        try:
            out = model(
                coords_batch,
                padding_mask=padding_mask,
                confidence=confidence,
                prev_output_tokens=tokens,
            )
        except TypeError:
            try:
                out = model(coords_batch, padding_mask, confidence, tokens)
            except TypeError:
                out = model(coords_batch, padding_mask, confidence)

    rep = None
    if isinstance(out, (tuple, list)) and len(out) > 1:
        state_dict = out[1]
        if isinstance(state_dict, dict) and "inner_states" in state_dict:
            rep = state_dict["inner_states"][-1]
        else:
            rep = state_dict
    elif isinstance(out, dict):
        if "representations" in out:
            reps = out["representations"]
            rep = reps[max(reps.keys())] if isinstance(reps, dict) else reps
        elif "encoder_out" in out:
            rep = out["encoder_out"]

    if rep is None:
        raise RuntimeError("Unable to extract representations from IF1 output.")

    if rep.dim() == 3 and rep.shape[1] == 1:
        rep = rep[:, 0, :]
    elif rep.dim() == 3:
        rep = rep[0]

    rep = rep.detach().float().cpu()

    if rep.shape[0] == len(seq) + 1:
        rep = rep[1:]
    elif rep.shape[0] == len(seq) + 2:
        rep = rep[1:-1]
    elif rep.shape[0] > len(seq):
        rep = rep[: len(seq)]

    return rep.numpy(), seq


def main():
    parser = argparse.ArgumentParser(
        description="Build IF1 + PROPKA dataset from existing ESM2+PROPKA pickle"
    )
    parser.add_argument(
        "--input_pkl",
        default="all_datasets_fixed.pkl",
        help="Path to ESM2+PROPKA dataset pickle",
    )
    parser.add_argument(
        "--output_pkl",
        default="all_datasets_if1_propka.pkl",
        help="Output pickle path for IF1+PROPKA dataset",
    )
    parser.add_argument(
        "--pdb_root",
        default="CleavgDB_clean",
        help="Root directory containing per-protein PDB folders",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for IF1 model",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit for quick runs",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_pkl):
        raise FileNotFoundError(f"Input pickle not found: {args.input_pkl}")

    model, alphabet = load_if1_model(args.device)

    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)

    output = {}
    keys = list(data.keys())
    if args.max_samples:
        keys = keys[: args.max_samples]

    for pdb_id in tqdm(keys, desc="Extracting IF1 embeddings"):
        sample = data[pdb_id]
        features = sample["features"]
        labels = sample["labels"]
        meta = sample.get("meta", {})

        if features.shape[1] < 518:
            raise ValueError(f"Expected >=518 features for {pdb_id}, got {features.shape}")

        propka = features[:, 512:518]

        chain_id = meta.get("chain")
        if not chain_id:
            parts = pdb_id.split("_")
            chain_id = parts[1] if len(parts) > 1 else None

        pdb_path = os.path.join(args.pdb_root, pdb_id, "protein.pdb")
        if not chain_id or not os.path.exists(pdb_path):
            print(f"Skipping {pdb_id}: missing chain or PDB ({pdb_path})")
            continue

        try:
            if1_emb, seq = extract_if1_embeddings(pdb_path, chain_id, model, alphabet, args.device)
        except Exception as exc:
            print(f"Failed {pdb_id}: {exc}")
            continue

        target_len = min(len(labels), if1_emb.shape[0], propka.shape[0])
        if target_len <= 0:
            print(f"Skipping {pdb_id}: invalid lengths")
            continue

        if if1_emb.shape[0] != target_len or propka.shape[0] != target_len:
            print(
                f"Length mismatch for {pdb_id}: "
                f"labels={len(labels)} if1={if1_emb.shape[0]} propka={propka.shape[0]}"
            )

        new_features = np.concatenate(
            [if1_emb[:target_len], propka[:target_len]], axis=1
        ).astype(np.float32)
        new_labels = labels[:target_len].astype(np.float32)

        output[pdb_id] = {
            "features": new_features,
            "labels": new_labels,
            "meta": meta,
        }

    with open(args.output_pkl, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved IF1+PROPKA dataset to {args.output_pkl}")


if __name__ == "__main__":
    main()
