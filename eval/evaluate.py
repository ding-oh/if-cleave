"""
Window-based evaluation for cleavage site prediction.

Instead of exact residue matching, this evaluates:
- True Positive: Predicted positive is within ±window of any true cleavage site
- False Positive: Predicted positive is NOT within any window
- True Negative: Predicted negative is NOT within any window
- False Negative: True cleavage site has no prediction within its window
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import convert_to_native

def get_cleavage_sites(labels):
    """Get indices of original cleavage sites (window=1 labels)."""
    return np.where(labels == 1)[0]

def is_within_window(pred_idx, cleavage_sites, window_half):
    """Check if prediction index is within window of any cleavage site."""
    for site in cleavage_sites:
        if abs(pred_idx - site) <= window_half:
            return True
    return False

def window_based_evaluation(y_true_original, y_pred, eval_window):
    """
    Evaluate predictions using window-based matching.

    Args:
        y_true_original: Original cleavage site labels (window=1, no expansion)
        y_pred: Binary predictions (0 or 1)
        eval_window: Evaluation window size (e.g., 7 means ±3 residues)

    Returns:
        Dictionary of metrics
    """
    window_half = eval_window // 2
    cleavage_sites = get_cleavage_sites(y_true_original)

    # Create window-based ground truth for evaluation
    y_true_window = np.zeros_like(y_true_original)
    for site in cleavage_sites:
        start = max(0, site - window_half)
        end = min(len(y_true_original), site + window_half + 1)
        y_true_window[start:end] = 1

    # Method 1: Simple window-expanded label matching
    # This is essentially what we already do with label expansion
    tp = np.sum((y_pred == 1) & (y_true_window == 1))
    fp = np.sum((y_pred == 1) & (y_true_window == 0))
    tn = np.sum((y_pred == 0) & (y_true_window == 0))
    fn = np.sum((y_pred == 0) & (y_true_window == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    # MCC calculation (use float64 to avoid overflow)
    tp_f, fp_f, tn_f, fn_f = float(tp), float(fp), float(tn), float(fn)
    mcc_denom = np.sqrt((tp_f+fp_f) * (tp_f+fn_f) * (tn_f+fp_f) * (tn_f+fn_f))
    mcc = (tp_f*tn_f - fp_f*fn_f) / mcc_denom if mcc_denom > 0 else 0

    return {
        'eval_window': eval_window,
        'mcc': mcc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'num_cleavage_sites': len(cleavage_sites),
        'window_positive_ratio': np.mean(y_true_window)
    }


def site_based_evaluation(y_true_original, y_pred, eval_window):
    """
    Site-based evaluation: For each true cleavage site, check if there's
    at least one positive prediction within its window.

    This measures: "How many cleavage sites did we successfully detect?"
    """
    window_half = eval_window // 2
    cleavage_sites = get_cleavage_sites(y_true_original)

    detected_sites = 0
    for site in cleavage_sites:
        start = max(0, site - window_half)
        end = min(len(y_pred), site + window_half + 1)
        if np.any(y_pred[start:end] == 1):
            detected_sites += 1

    site_recall = detected_sites / len(cleavage_sites) if len(cleavage_sites) > 0 else 0

    return {
        'eval_window': eval_window,
        'total_sites': len(cleavage_sites),
        'detected_sites': detected_sites,
        'site_recall': site_recall
    }


def load_and_evaluate(pred_file, data_dir_w1, eval_windows=[1, 3, 5, 7, 9]):
    """
    Load predictions and evaluate with different window sizes.

    Args:
        pred_file: Path to {model}_predictions.npz
        data_dir_w1: Directory containing window=1 (original) test data
        eval_windows: List of evaluation window sizes to try
    """
    pred_data = np.load(pred_file)
    y_prob = pred_data['predictions']
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    test_data = torch.load(Path(data_dir_w1) / 'test_data.pt', weights_only=False)
    y_true_original = np.concatenate([data.y.numpy() for data in test_data])

    print(f"Loaded {len(y_prob)} predictions")
    print(f"Original cleavage sites: {np.sum(y_true_original)}")
    print(f"Predicted positives: {np.sum(y_pred)}")
    print()

    # Evaluate with different windows
    results = []
    site_results = []

    for window in eval_windows:
        metrics = window_based_evaluation(y_true_original, y_pred, window)
        site_metrics = site_based_evaluation(y_true_original, y_pred, window)

        results.append(metrics)
        site_results.append(site_metrics)

        print(f"=== Evaluation Window = {window} (±{window//2} residues) ===")
        print(f"  MCC:         {metrics['mcc']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1:          {metrics['f1']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Site Recall: {site_metrics['site_recall']:.4f} ({site_metrics['detected_sites']}/{site_metrics['total_sites']} sites)")
        print()

    return results, site_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True,
                        help='Path to {model}_predictions.npz from train.py')
    parser.add_argument('--data_dir_w1', type=str, default='data_if1_w1',
                        help='Directory containing window=1 test data')
    parser.add_argument('--eval_windows', type=int, nargs='+', default=[1, 3, 5, 7, 9, 11])
    args = parser.parse_args()

    results, site_results = load_and_evaluate(
        args.pred_file, args.data_dir_w1, args.eval_windows
    )

    output_file = Path(args.pred_file).parent / 'window_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump(convert_to_native({
            'pred_file': str(args.pred_file),
            'window_metrics': results,
            'site_metrics': site_results,
        }), f, indent=2)
    print(f"Saved results to {output_file}")
