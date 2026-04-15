import os
import json
import pickle
import argparse
import csv
import hashlib
import shutil
import subprocess
import tempfile
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def expand_labels_window(labels, window_size=9):
    """
    Expand binary labels using a window approach.
    If any position within the window is 1, mark the center as 1.

    Args:
        labels: Binary array of cleavage sites
        window_size: Size of the window (should be odd)

    Returns:
        Expanded labels
    """
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd")

    half_window = window_size // 2
    expanded = np.zeros_like(labels)

    for i in range(len(labels)):
        start = max(0, i - half_window)
        end = min(len(labels), i + half_window + 1)

        # If any position in the window is 1, mark this position as 1
        if np.any(labels[start:end]):
            expanded[i] = 1

    return expanded

def load_cleavage_data(base_path="CleavgDB_clean",
                       window_size=9,
                       max_samples=None,
                       pkl_file="all_datasets_fixed.pkl"):
    """
    Load CleavgDB_clean data with window-based label expansion.
    """
    # First check if we have the preprocessed pickle file
    if os.path.exists(pkl_file):
        print(f"Loading preprocessed data from {pkl_file}")
        with open(pkl_file, 'rb') as f:
            all_data = pickle.load(f)

        # Process data with window expansion
        processed_data = []

        for idx, (pdb_id, sample) in enumerate(tqdm(all_data.items(), desc="Processing samples")):
            if max_samples and idx >= max_samples:
                break

            features = sample['features']  # Shape: (seq_len, 518)
            labels = sample['labels']      # Shape: (seq_len,)

            # Expand labels using window approach
            expanded_labels = expand_labels_window(labels, window_size=window_size)

            # Create PyG Data object
            data = Data(
                x=torch.FloatTensor(features),
                y=torch.FloatTensor(expanded_labels),
                original_y=torch.FloatTensor(labels),
                pdb_id=pdb_id,
                seq_len=len(labels)
            )

            processed_data.append(data)

        print(f"Loaded {len(processed_data)} samples")
        return processed_data

    else:
        print(f"Preprocessed pickle not found. Loading from raw PDB directories...")

        # Get all PDB directories
        pdb_dirs = [d for d in os.listdir(base_path)
                    if os.path.isdir(os.path.join(base_path, d))]

        if max_samples:
            pdb_dirs = pdb_dirs[:max_samples]

        all_data = []

        for pdb_dir in tqdm(pdb_dirs, desc="Loading PDB data"):
            pdb_path = os.path.join(base_path, pdb_dir)

            # Load protein structure and features
            protein_pdb = os.path.join(pdb_path, "protein.pdb")
            protein_pka = os.path.join(pdb_path, "protein.pka")

            if not os.path.exists(protein_pdb):
                print(f"Warning: Missing PDB file for {pdb_dir}")
                continue

            # Get epitope directories
            epitope_dirs = [d for d in os.listdir(pdb_path)
                           if d.startswith('epitope')]

            if not epitope_dirs:
                print(f"Warning: No epitopes found for {pdb_dir}")
                continue

            # Load epitope information
            all_cleavage_positions = []

            for epitope_dir in epitope_dirs:
                info_path = os.path.join(pdb_path, epitope_dir, "info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        n_cleav = info.get('n_cleavage', -1)
                        c_cleav = info.get('c_cleavage', -1)

                        if n_cleav != -1:
                            all_cleavage_positions.append(n_cleav)
                        if c_cleav != -1:
                            all_cleavage_positions.append(c_cleav)

            if not all_cleavage_positions:
                print(f"Warning: No cleavage positions found for {pdb_dir}")
                continue

            raise NotImplementedError(
                "Feature extraction from raw PDB is not implemented in this path. "
                "Use the preprocessed dataset pipeline."
            )

        return all_data

def create_splits(data_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                  random_seed=42, group_map=None, val_ratio_of_trainval=False):
    """
    Split data into train/val/test sets.
    """
    if val_ratio_of_trainval:
        if not (0.0 < test_ratio < 1.0 and 0.0 < val_ratio < 1.0):
            raise ValueError("test_ratio and val_ratio must be in (0,1) when val_ratio_of_trainval is set.")
    else:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Get PDB IDs
    pdb_ids = [d.pdb_id for d in data_list]

    if group_map:
        # Split at sequence group level to avoid identical sequences across splits
        group_ids = sorted(set(group_map.values()))
        train_val_groups, test_groups = train_test_split(
            group_ids,
            test_size=test_ratio,
            random_state=random_seed
        )

        val_size = val_ratio if val_ratio_of_trainval else val_ratio / (train_ratio + val_ratio)
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=val_size,
            random_state=random_seed
        )

        train_ids = [pid for pid in pdb_ids if group_map.get(pid) in train_groups]
        val_ids = [pid for pid in pdb_ids if group_map.get(pid) in val_groups]
        test_ids = [pid for pid in pdb_ids if group_map.get(pid) in test_groups]
    else:
        # First split: train+val vs test
        train_val_ids, test_ids = train_test_split(
            pdb_ids,
            test_size=test_ratio,
            random_state=random_seed
        )

        # Second split: train vs val
        val_size = val_ratio if val_ratio_of_trainval else val_ratio / (train_ratio + val_ratio)
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_size,
            random_state=random_seed
        )

    # Create datasets
    train_data = [d for d in data_list if d.pdb_id in train_ids]
    val_data = [d for d in data_list if d.pdb_id in val_ids]
    test_data = [d for d in data_list if d.pdb_id in test_ids]

    return train_data, val_data, test_data

def calculate_statistics(data_list):
    """
    Calculate dataset statistics.
    """
    total_residues = sum(d.seq_len for d in data_list)
    total_positive = sum(d.y.sum().item() for d in data_list)
    total_original_positive = sum(d.original_y.sum().item() for d in data_list)

    stats = {
        'num_samples': len(data_list),
        'total_residues': total_residues,
        'avg_seq_len': total_residues / len(data_list),
        'min_seq_len': min(d.seq_len for d in data_list),
        'max_seq_len': max(d.seq_len for d in data_list),
        'positive_ratio_expanded': total_positive / total_residues,
        'positive_ratio_original': total_original_positive / total_residues,
        'window_expansion_factor': total_positive / total_original_positive if total_original_positive > 0 else 0
    }

    return stats


def _extract_protein_sequence(pdb_dir):
    for entry in os.listdir(pdb_dir):
        if not entry.startswith("epitope"):
            continue
        info_path = os.path.join(pdb_dir, entry, "info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r") as f:
                    info = json.load(f)
                seq = info.get("Protein Sequence")
            except (OSError, json.JSONDecodeError):
                seq = None
            if seq:
                return seq
    return None


def build_sequence_group_map(base_path, pdb_ids):
    """
    Build a map from pdb_id to a sequence hash using info.json in epitope folders.
    Falls back to pdb_id when sequence cannot be found.
    """
    group_map = {}
    for pdb_id in pdb_ids:
        seq = None
        pdb_dir = os.path.join(base_path, pdb_id)
        if os.path.isdir(pdb_dir):
            seq = _extract_protein_sequence(pdb_dir)
        if seq:
            seq_hash = hashlib.sha1(seq.encode("utf-8")).hexdigest()
            group_map[pdb_id] = seq_hash
        else:
            group_map[pdb_id] = f"pdb:{pdb_id}"
    return group_map


def build_cdhit_group_map(base_path, pdb_ids, output_dir, cdhit_exe="cd-hit",
                          identity=1.0, word_size=5, threads=1, memory=0):
    if shutil.which(cdhit_exe) is None:
        raise RuntimeError(f"cd-hit not found on PATH: {cdhit_exe}")

    sequences = {}
    missing = []
    for pdb_id in pdb_ids:
        pdb_dir = os.path.join(base_path, pdb_id)
        seq = _extract_protein_sequence(pdb_dir) if os.path.isdir(pdb_dir) else None
        if seq:
            sequences[pdb_id] = seq
        else:
            missing.append(pdb_id)

    if not sequences:
        raise RuntimeError("No protein sequences found for CD-HIT clustering.")

    os.makedirs(output_dir, exist_ok=True)
    fasta_path = os.path.join(output_dir, "cdhit_sequences.fasta")
    with open(fasta_path, "w") as f:
        for pdb_id, seq in sequences.items():
            f.write(f">{pdb_id}\n")
            f.write(f"{seq}\n")

    output_prefix = os.path.join(output_dir, "cdhit_clusters")
    cmd = [
        cdhit_exe,
        "-i", fasta_path,
        "-o", output_prefix,
        "-c", str(identity),
        "-n", str(word_size),
        "-T", str(threads),
        "-M", str(memory),
    ]
    subprocess.run(cmd, check=True)

    cluster_path = f"{output_prefix}.clstr"
    group_map = {}
    current_cluster = None
    with open(cluster_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = line.split()[-1]
                continue
            if ">" in line and "..." in line and current_cluster is not None:
                seq_id = line.split(">", 1)[1].split("...", 1)[0]
                group_map[seq_id] = f"cdhit:{current_cluster}"

    for pdb_id in missing:
        group_map[pdb_id] = f"pdb:{pdb_id}"

    return group_map

def main():
    """
    Main function to prepare GeoCleav dataset.
    """
    print("=" * 50)
    print("GeoCleav Dataset Preparation")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="Prepare GeoCleav dataset splits")
    parser.add_argument('--window_size', type=int, default=9,
                        help='Window size for label expansion')
    parser.add_argument('--output_dir', type=str,
                        default="data_if1",
                        help='Output directory for .pt splits')
    parser.add_argument('--pkl_file', type=str,
                        default="all_datasets_fixed.pkl",
                        help='Path to preprocessed dataset pickle')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test split ratio')
    parser.add_argument('--val_ratio_of_trainval', action='store_true',
                        help='Interpret val_ratio as fraction of (train+val) after test split')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    parser.add_argument('--split_by_sequence', action='store_true',
                        help='Ensure identical protein sequences do not cross splits')
    parser.add_argument('--split_by_cdhit', action='store_true',
                        help='Use CD-HIT to cluster sequences before splitting')
    parser.add_argument('--test_list_csv', type=str, default=None,
                        help='Optional path to save test set list as CSV')
    parser.add_argument('--sequence_base_dir', type=str,
                        default="CleavgDB_clean",
                        help='Base directory to resolve protein sequences from info.json')
    parser.add_argument('--cdhit_exe', type=str, default="cd-hit",
                        help='Path to cd-hit executable')
    parser.add_argument('--cdhit_identity', type=float, default=1.0,
                        help='CD-HIT sequence identity threshold (0.0-1.0)')
    parser.add_argument('--cdhit_word', type=int, default=5,
                        help='CD-HIT word size (use appropriate value for identity)')
    parser.add_argument('--cdhit_threads', type=int, default=1,
                        help='CD-HIT threads')
    parser.add_argument('--cdhit_memory', type=int, default=0,
                        help='CD-HIT memory in MB (0 for unlimited)')
    args = parser.parse_args()

    # Parameters
    window_size = args.window_size
    output_dir = args.output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"\nLoading data with {window_size}-window label expansion...")
    data_list = load_cleavage_data(window_size=window_size, pkl_file=args.pkl_file)

    if not data_list:
        print("Error: No data loaded!")
        return

    # Create splits
    print("\nCreating train/val/test splits...")
    group_map = None
    split_method = "random"
    if args.split_by_cdhit:
        print("\nBuilding CD-HIT sequence clusters...")
        group_map = build_cdhit_group_map(
            args.sequence_base_dir,
            [d.pdb_id for d in data_list],
            output_dir,
            cdhit_exe=args.cdhit_exe,
            identity=args.cdhit_identity,
            word_size=args.cdhit_word,
            threads=args.cdhit_threads,
            memory=args.cdhit_memory,
        )
        split_method = "cdhit"
    elif args.split_by_sequence:
        print("\nBuilding sequence hash group map...")
        group_map = build_sequence_group_map(args.sequence_base_dir, [d.pdb_id for d in data_list])
        split_method = "sequence_hash"

    train_data, val_data, test_data = create_splits(
        data_list,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        group_map=group_map,
        val_ratio_of_trainval=args.val_ratio_of_trainval
    )

    # Calculate statistics
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    for split_name, split_data in [("Train", train_data),
                                    ("Val", val_data),
                                    ("Test", test_data)]:
        stats = calculate_statistics(split_data)
        print(f"\n{split_name} Set:")
        print(f"  Samples: {stats['num_samples']}")
        print(f"  Total residues: {stats['total_residues']:,}")
        print(f"  Avg sequence length: {stats['avg_seq_len']:.1f}")
        print(f"  Sequence length range: {stats['min_seq_len']}-{stats['max_seq_len']}")
        print(f"  Positive ratio (original): {stats['positive_ratio_original']:.2%}")
        print(f"  Positive ratio (expanded): {stats['positive_ratio_expanded']:.2%}")
        print(f"  Window expansion factor: {stats['window_expansion_factor']:.2f}x")

    # Save datasets
    print(f"\nSaving datasets to {output_dir}...")

    torch.save(train_data, os.path.join(output_dir, 'train_data.pt'))
    torch.save(val_data, os.path.join(output_dir, 'val_data.pt'))
    torch.save(test_data, os.path.join(output_dir, 'test_data.pt'))

    # Save metadata
    if args.test_list_csv:
        print(f"\nSaving test set list to {args.test_list_csv}...")
        with open(args.test_list_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pdb_id", "group_id"])
            for d in test_data:
                writer.writerow([d.pdb_id, group_map.get(d.pdb_id) if group_map else "random"])

    metadata = {
        'window_size': window_size,
        'feature_dim': 518,
        'seed': args.seed,
        'split_method': split_method,
        'cdhit_params': {
            'identity': args.cdhit_identity,
            'word_size': args.cdhit_word,
            'threads': args.cdhit_threads,
            'memory': args.cdhit_memory,
        } if split_method == "cdhit" else None,
        'ratios': {
            'train': args.train_ratio,
            'val': args.val_ratio,
            'test': args.test_ratio,
            'val_ratio_of_trainval': bool(args.val_ratio_of_trainval)
        },
        'splits': {
            'train': len(train_data),
            'val': len(val_data),
            'test': len(test_data)
        },
        'stats': {
            'train': calculate_statistics(train_data),
            'val': calculate_statistics(val_data),
            'test': calculate_statistics(test_data)
        }
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset preparation complete!")
    print(f"Data saved to: {output_dir}")

    return train_data, val_data, test_data

if __name__ == "__main__":
    main()
