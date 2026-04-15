import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_recall_fscore_support
)


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate evaluation metrics for binary classification."""
    y_pred_binary = (y_pred > threshold).astype(int)

    mcc = matthews_corrcoef(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='binary', zero_division=0
    )

    tn = ((y_true == 0) & (y_pred_binary == 0)).sum()
    fp = ((y_true == 0) & (y_pred_binary == 1)).sum()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'mcc': mcc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    }


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    return obj
