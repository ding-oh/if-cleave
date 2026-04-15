"""Prepare data for window size ablation study.

Reuses the SAME train/val/test split from data_if1/ (CD-HIT, seed=42)
but re-applies label expansion with different window sizes.
This ensures fair comparison across window sizes.
"""

import os
import pickle
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from data.prepare_data import expand_labels_window


def main():
    parser = argparse.ArgumentParser(description="Window ablation data preparation")
    parser.add_argument("--window_sizes", nargs="+", type=int,
                        default=[1, 3, 5, 7, 9, 11, 13, 15],
                        help="Window sizes to prepare")
    parser.add_argument("--ref_dir", default="data_if1",
                        help="Reference data directory (for split IDs)")
    parser.add_argument("--pkl_file", default="all_datasets_if1_propka.pkl",
                        help="IF1+PROPKA dataset pickle")
    parser.add_argument("--output_base", default="data_if1_w",
                        help="Output directory prefix (e.g., data_if1_w → data_if1_w9)")
    args = parser.parse_args()

    # Step 1: Load reference splits to get PDB ID lists
    print("Loading reference split PDB IDs from", args.ref_dir)
    ref_train = torch.load(os.path.join(args.ref_dir, "train_data.pt"),
                           map_location="cpu", weights_only=False)
    ref_val = torch.load(os.path.join(args.ref_dir, "val_data.pt"),
                         map_location="cpu", weights_only=False)
    ref_test = torch.load(os.path.join(args.ref_dir, "test_data.pt"),
                          map_location="cpu", weights_only=False)

    train_ids = set(d.pdb_id for d in ref_train)
    val_ids = set(d.pdb_id for d in ref_val)
    test_ids = set(d.pdb_id for d in ref_test)
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Step 2: Load raw data
    print(f"Loading raw data from {args.pkl_file}...")
    with open(args.pkl_file, "rb") as f:
        all_data = pickle.load(f)
    print(f"  Total entries: {len(all_data)}")

    # Step 3: For each window size, create expanded datasets
    for ws in args.window_sizes:
        out_dir = f"{args.output_base}{ws}"
        if (os.path.exists(os.path.join(out_dir, "train_data.pt")) and
            os.path.exists(os.path.join(out_dir, "val_data.pt")) and
            os.path.exists(os.path.join(out_dir, "test_data.pt"))):
            print(f"\n  Window {ws}: already exists at {out_dir}, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"  Window size: {ws}")
        print(f"{'='*50}")

        os.makedirs(out_dir, exist_ok=True)

        train_data, val_data, test_data = [], [], []

        for pdb_id, sample in tqdm(all_data.items(), desc=f"w={ws}"):
            features = sample["features"]
            labels = sample["labels"]

            expanded = expand_labels_window(labels, window_size=ws)

            data = Data(
                x=torch.FloatTensor(features),
                y=torch.FloatTensor(expanded),
                original_y=torch.FloatTensor(labels),
                pdb_id=pdb_id,
                seq_len=len(labels),
            )

            if pdb_id in train_ids:
                train_data.append(data)
            elif pdb_id in val_ids:
                val_data.append(data)
            elif pdb_id in test_ids:
                test_data.append(data)
            # else: PDB ID not in any split (shouldn't happen)

        print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        # Compute positive ratio
        total_pos = sum(d.y.sum().item() for d in train_data)
        total_res = sum(d.seq_len for d in train_data)
        orig_pos = sum(d.original_y.sum().item() for d in train_data)
        print(f"  Positive ratio: {total_pos/total_res:.2%} (original: {orig_pos/total_res:.2%})")
        print(f"  Expansion factor: {total_pos/orig_pos:.2f}x")

        torch.save(train_data, os.path.join(out_dir, "train_data.pt"))
        torch.save(val_data, os.path.join(out_dir, "val_data.pt"))
        torch.save(test_data, os.path.join(out_dir, "test_data.pt"))
        print(f"  Saved to {out_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
