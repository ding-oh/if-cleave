import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from datetime import datetime

from model.model import get_model
from utils.metrics import calculate_metrics, convert_to_native
from utils.data import CleavageDataset, custom_collate
from utils.features import apply_feature_exclusion


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_epoch(model, loader, optimizer, criterion, device,
                label_smoothing=0.0, grad_clip=0.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch.x, batch.batch)
        targets = batch.y.float()
        if label_smoothing > 0:
            targets = targets * (1 - 2 * label_smoothing) + label_smoothing
        loss = criterion(outputs, targets)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = torch.sigmoid(outputs).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(loader)

    return metrics


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)

            outputs = model(batch.x, batch.batch)
            loss = criterion(outputs, batch.y.float())

            total_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(loader)

    return metrics, all_preds, all_labels


def train_fold(fold_idx, train_data, val_data, args, device):
    """Train a single fold."""
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/{args.n_folds}")
    print(f"{'='*60}")
    print(f"Train: {len(train_data)} samples")
    print(f"Val: {len(val_data)} samples")

    train_dataset = CleavageDataset(train_data)
    val_dataset = CleavageDataset(val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=custom_collate,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=custom_collate,
        num_workers=args.num_workers
    )

    model = get_model(
        args.model,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    if args.loss == 'bce':
        num_pos = sum(d.y.sum().item() for d in train_data)
        num_neg = sum((1 - d.y).sum().item() for d in train_data)
        pos_weight = torch.tensor([num_neg / num_pos]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_mcc = -1
    best_epoch = 0
    history = {'train': [], 'val': []}

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device,
                                    label_smoothing=args.label_smoothing,
                                    grad_clip=args.grad_clip)

        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['mcc'])

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        if (epoch + 1) % 10 == 0 or val_metrics['mcc'] > best_val_mcc:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train MCC: {train_metrics['mcc']:.4f}, "
                  f"Val MCC: {val_metrics['mcc']:.4f}")

        if val_metrics['mcc'] > best_val_mcc:
            best_val_mcc = val_metrics['mcc']
            best_epoch = epoch + 1

            best_state = {
                'model_state_dict': model.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }

        if epoch + 1 - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nFold {fold_idx + 1} - Best Val MCC: {best_val_mcc:.4f} at epoch {best_epoch}")

    return {
        'fold': fold_idx,
        'best_epoch': best_epoch,
        'best_val_mcc': best_val_mcc,
        'train_metrics': best_state['train_metrics'],
        'val_metrics': best_state['val_metrics'],
        'history': history,
        'model_state_dict': best_state['model_state_dict']
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading datasets...")
    train_data = torch.load(os.path.join(args.data_dir, 'train_data.pt'), weights_only=False)
    val_data = torch.load(os.path.join(args.data_dir, 'val_data.pt'), weights_only=False)
    test_data = torch.load(os.path.join(args.data_dir, 'test_data.pt'), weights_only=False)

    exclude = [n.strip() for n in args.exclude_propka.split(',') if n.strip()] \
              if args.exclude_propka else []
    train_data, input_dim = apply_feature_exclusion(train_data, exclude)
    val_data, _           = apply_feature_exclusion(val_data, exclude)
    test_data, _          = apply_feature_exclusion(test_data, exclude)
    args.input_dim = input_dim

    all_data = train_data + val_data
    print(f"Total samples for {args.n_folds}-fold CV: {len(all_data)}")
    print(f"Test samples: {len(test_data)}")

    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    test_dataset = CleavageDataset(test_data)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=custom_collate,
        num_workers=args.num_workers
    )

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_data)):
        fold_train_data = [all_data[i] for i in train_idx]
        fold_val_data = [all_data[i] for i in val_idx]

        result = train_fold(fold_idx, fold_train_data, fold_val_data, args, device)

        print(f"\nEvaluating fold {fold_idx} on test set...")
        model = get_model(
            args.model,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        ).to(device)

        model.load_state_dict(result['model_state_dict'])

        all_labels = np.concatenate([d.y.numpy() for d in fold_train_data])
        pos_weight = torch.tensor([(1 - all_labels.mean()) / all_labels.mean()]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        result['test_metrics'] = test_metrics
        result['test_preds'] = test_preds
        result['test_labels'] = test_labels

        print(f"Fold {fold_idx} Test MCC: {test_metrics['mcc']:.4f}")

        checkpoint_path = os.path.join(args.output_dir, f"{args.model}_fold{fold_idx}_best.pt")
        torch.save(result['model_state_dict'], checkpoint_path)
        print(f"Saved model to {checkpoint_path}")

        del result['model_state_dict']
        fold_results.append(result)

    print(f"\n{'='*60}")
    print(f"{args.n_folds}-Fold Cross Validation Results")
    print(f"{'='*60}")

    avg_val_mcc = np.mean([r['best_val_mcc'] for r in fold_results])
    std_val_mcc = np.std([r['best_val_mcc'] for r in fold_results])

    print(f"\nValidation MCC per fold:")
    for i, result in enumerate(fold_results):
        print(f"  Fold {i+1}: {result['best_val_mcc']:.4f} (epoch {result['best_epoch']})")

    print(f"\nAverage Val MCC: {avg_val_mcc:.4f} +/- {std_val_mcc:.4f}")

    metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    print(f"\nAverage Validation Metrics:")
    for key in metrics_keys:
        values = [r['val_metrics'][key] for r in fold_results]
        print(f"  {key.capitalize():12s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    avg_test_mcc = np.mean([r['test_metrics']['mcc'] for r in fold_results])
    std_test_mcc = np.std([r['test_metrics']['mcc'] for r in fold_results])
    print(f"\nTest MCC per fold:")
    for i, result in enumerate(fold_results):
        print(f"  Fold {i+1}: {result['test_metrics']['mcc']:.4f}")
    print(f"\nAverage Test MCC: {avg_test_mcc:.4f} +/- {std_test_mcc:.4f}")

    print(f"\nAverage Test Metrics:")
    for key in metrics_keys:
        values = [r['test_metrics'][key] for r in fold_results]
        print(f"  {key.capitalize():12s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    all_test_preds = np.array([r['test_preds'] for r in fold_results])
    ensemble_preds = np.mean(all_test_preds, axis=0)
    test_labels = np.array(fold_results[0]['test_labels'])
    ensemble_metrics = calculate_metrics(test_labels, ensemble_preds)
    print(f"\nEnsemble Test MCC: {ensemble_metrics['mcc']:.4f}")
    print(f"Ensemble Test Metrics:")
    for key in metrics_keys:
        print(f"  {key.capitalize():12s}: {ensemble_metrics[key]:.4f}")

    fold_results_for_json = []
    for r in fold_results:
        r_copy = {k: v for k, v in r.items() if k not in ['test_preds', 'test_labels']}
        fold_results_for_json.append(r_copy)

    results = {
        'model': args.model,
        'n_folds': args.n_folds,
        'avg_val_mcc': avg_val_mcc,
        'std_val_mcc': std_val_mcc,
        'avg_test_mcc': avg_test_mcc,
        'std_test_mcc': std_test_mcc,
        'ensemble_test_mcc': ensemble_metrics['mcc'],
        'ensemble_test_metrics': ensemble_metrics,
        'fold_results': fold_results_for_json,
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
    }

    results_path = os.path.join(
        args.output_dir,
        f"{args.model}_{args.n_folds}fold_results.json"
    )
    with open(results_path, 'w') as f:
        results = convert_to_native(results)
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Fold Cross Validation for cleavage prediction")

    parser.add_argument('--n_folds', type=int, default=4)
    parser.add_argument('--model', type=str, default='bilstm',
                       choices=['simple', 'attention', 'conv', 'bilstm'])
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'focal'])
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=0.0)

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--exclude_propka', type=str, default='',
                       help='Comma-separated PROPKA features to exclude, or "all_propka"')

    args = parser.parse_args()
    args.input_dim = 518

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
