"""
Evaluation Metrics for DDI Prediction

Comprehensive metrics computation for multi-class DDI classification.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    auc_macro: Optional[float]
    auc_weighted: Optional[float]
    mcc: float
    kappa: float
    confusion_matrix: np.ndarray
    per_class_f1: np.ndarray


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
) -> ClassificationMetrics:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels [N]
        y_pred: Predicted labels [N]
        y_prob: Prediction probabilities [N, num_classes]
        num_classes: Number of classes (inferred if None)

    Returns:
        ClassificationMetrics object
    """
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # MCC and Kappa
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # AUC-ROC (requires probabilities)
    auc_macro = None
    auc_weighted = None
    if y_prob is not None:
        try:
            # Only compute if we have more than one class in the batch
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 1:
                auc_macro = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='macro'
                )
                auc_weighted = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='weighted'
                )
        except ValueError:
            pass  # Not enough classes

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        auc_macro=auc_macro,
        auc_weighted=auc_weighted,
        mcc=mcc,
        kappa=kappa,
        confusion_matrix=cm,
        per_class_f1=per_class_f1,
    )


def compute_metrics_with_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute metrics with bootstrap confidence intervals.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Dict mapping metric name to (value, lower_bound, upper_bound)
    """
    np.random.seed(42)
    n_samples = len(y_true)

    metrics_samples = {
        'accuracy': [],
        'f1_macro': [],
        'precision_macro': [],
        'recall_macro': [],
    }

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Compute metrics
        metrics_samples['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
        metrics_samples['f1_macro'].append(
            f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0)
        )
        metrics_samples['precision_macro'].append(
            precision_score(y_true_boot, y_pred_boot, average='macro', zero_division=0)
        )
        metrics_samples['recall_macro'].append(
            recall_score(y_true_boot, y_pred_boot, average='macro', zero_division=0)
        )

    # Compute confidence intervals
    alpha = (1 - confidence) / 2
    results = {}

    for metric_name, samples in metrics_samples.items():
        samples = np.array(samples)
        value = np.mean(samples)
        lower = np.percentile(samples, alpha * 100)
        upper = np.percentile(samples, (1 - alpha) * 100)
        results[metric_name] = (value, lower, upper)

    return results


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
) -> str:
    """
    Print detailed classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: Mapping from label index to name

    Returns:
        Classification report string
    """
    if label_names:
        target_names = [label_names.get(i, f"Class_{i}") for i in sorted(label_names.keys())]
    else:
        target_names = None

    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
    )

    return report


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics for each class separately.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities

    Returns:
        Dict mapping class index to metrics dict
    """
    classes = np.unique(y_true)
    results = {}

    for cls in classes:
        # Binary metrics for this class
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        metrics = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'support': int(y_true_binary.sum()),
        }

        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true_binary, y_prob[:, cls])
                metrics['ap'] = average_precision_score(y_true_binary, y_prob[:, cls])
            except (ValueError, IndexError):
                pass

        results[int(cls)] = metrics

    return results


def compare_models(
    results: Dict[str, ClassificationMetrics],
) -> str:
    """
    Compare metrics across multiple models.

    Args:
        results: Dict mapping model name to ClassificationMetrics

    Returns:
        Formatted comparison table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Model Comparison")
    lines.append("=" * 80)

    header = f"{'Model':<20} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'AUC':>10}"
    lines.append(header)
    lines.append("-" * 80)

    for model_name, metrics in results.items():
        auc_str = f"{metrics.auc_macro:.4f}" if metrics.auc_macro else "N/A"
        line = f"{model_name:<20} {metrics.accuracy:>10.4f} {metrics.f1_macro:>10.4f} {metrics.f1_weighted:>12.4f} {auc_str:>10}"
        lines.append(line)

    lines.append("=" * 80)

    return "\n".join(lines)


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drug1_ids: List[str],
    drug2_ids: List[str],
    label_names: Optional[Dict[int, str]] = None,
    top_k: int = 20,
) -> Dict[str, List]:
    """
    Analyze prediction errors.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        drug1_ids: First drug IDs
        drug2_ids: Second drug IDs
        label_names: Label name mapping
        top_k: Number of errors to analyze

    Returns:
        Dict with error analysis
    """
    # Find errors
    error_mask = y_true != y_pred
    error_indices = np.where(error_mask)[0]

    # Get error details
    errors = []
    for idx in error_indices[:top_k]:
        true_label = int(y_true[idx])
        pred_label = int(y_pred[idx])

        error = {
            'index': int(idx),
            'drug1': drug1_ids[idx],
            'drug2': drug2_ids[idx],
            'true_label': true_label,
            'pred_label': pred_label,
            'true_name': label_names.get(true_label, f"Class_{true_label}") if label_names else str(true_label),
            'pred_name': label_names.get(pred_label, f"Class_{pred_label}") if label_names else str(pred_label),
        }
        errors.append(error)

    # Confusion patterns
    cm = confusion_matrix(y_true, y_pred)
    confusion_pairs = []

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true_class': i,
                    'pred_class': j,
                    'count': int(cm[i, j]),
                    'true_name': label_names.get(i, f"Class_{i}") if label_names else str(i),
                    'pred_name': label_names.get(j, f"Class_{j}") if label_names else str(j),
                })

    # Sort by count
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

    return {
        'total_errors': int(error_mask.sum()),
        'error_rate': float(error_mask.mean()),
        'sample_errors': errors,
        'confusion_pairs': confusion_pairs[:top_k],
    }
