"""Run 4-fold ensemble inference on the test split using pre-trained checkpoints.

Recomputes per-fold train stats to standardize the test set the same way
training did, then ensembles sigmoid outputs across folds.
"""
import os
import sys
import copy
import glob
import argparse
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch_geometric.data import Batch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import IFCleave
from utils.data import compute_feature_stats, apply_standardization


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_raw = torch.load(os.path.join(args.data_dir, 'train_data.pt'), weights_only=False)
    val_raw = torch.load(os.path.join(args.data_dir, 'val_data.pt'), weights_only=False)
    test_raw = torch.load(os.path.join(args.data_dir, 'test_data.pt'), weights_only=False)
    all_raw = train_raw + val_raw
    print(f"train+val: {len(all_raw)}, test: {len(test_raw)}")

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, 'bilstm_fold*_best.pt'))
                   or glob.glob(os.path.join(args.ckpt_dir, 'bilstm_fold*_norot_best.pt')))
    if len(ckpts) != args.n_folds:
        raise RuntimeError(f"expected {args.n_folds} checkpoints in {args.ckpt_dir}, found {len(ckpts)}")

    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_splits = list(kfold.split(all_raw))

    fold_probs = []
    labels = None
    for k, (train_idx, _) in enumerate(fold_splits):
        fold_train = [all_raw[i] for i in train_idx]
        mean, std = compute_feature_stats(fold_train)

        test_std = copy.deepcopy(test_raw)
        apply_standardization(test_std, mean, std)

        model = IFCleave(input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                         dropout=args.dropout).to(device)
        model.load_state_dict(torch.load(ckpts[k], map_location=device, weights_only=False))
        model.eval()

        preds = []
        with torch.no_grad():
            for i in range(0, len(test_std), args.batch_size):
                batch = Batch.from_data_list(test_std[i:i+args.batch_size]).to(device)
                out = torch.sigmoid(model(batch.x, batch.batch)).cpu().numpy()
                preds.append(out)
                if k == 0 and labels is None:
                    pass
        fold_probs.append(np.concatenate(preds))
        print(f"fold{k}: mean_prob={fold_probs[-1].mean():.4f}")

    labels = np.concatenate([d.y.numpy() for d in test_raw])
    ensemble = np.mean(fold_probs, axis=0)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez(args.output, predictions=ensemble, labels=labels)
    print(f"saved {args.output} (predictions: {ensemble.shape}, labels: {labels.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-fold ensemble inference")
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing train/val/test_data.pt (used for fold splits and stats)')
    parser.add_argument('--output', type=str, default='results/bilstm_predictions.npz')
    parser.add_argument('--n_folds', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_dim', type=int, default=518)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.4)
    args = parser.parse_args()
    main(args)
