#!/usr/bin/env python
"""
Evaluation Script for DDI-GNN

Usage:
    python scripts/evaluate.py --checkpoint outputs/best_model.pt --data-split test
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import DDIDataset, get_dataloader
from src.models.full_model import load_model
from src.evaluation.metrics import (
    compute_metrics,
    compute_metrics_with_confidence,
    print_classification_report,
    analyze_errors,
)
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_class_distribution,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate DDI-GNN model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save all predictions to file",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Compute bootstrap confidence intervals",
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.config, device=device)
    model.eval()

    # Load dataset
    print(f"\nLoading {args.data_split} dataset...")
    dataset = DDIDataset(
        data_source='drugbank',
        split=args.data_split,
        root=args.data_dir,
    )
    print(f"Samples: {len(dataset)}")

    loader = get_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Run evaluation
    print("\nRunning evaluation...")
    all_preds = []
    all_labels = []
    all_probs = []
    all_drug1_ids = []
    all_drug2_ids = []

    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            drug1 = batch['drug1'].to(device)
            drug2 = batch['drug2'].to(device)
            labels = batch['labels']

            outputs = model(drug1, drug2)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_drug1_ids.extend(batch['drug1_ids'])
            all_drug2_ids.extend(batch['drug2_ids'])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    metrics = compute_metrics(all_labels, all_preds, all_probs, dataset.num_classes)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:         {metrics.accuracy:.4f}")
    print(f"  Precision (macro): {metrics.precision_macro:.4f}")
    print(f"  Recall (macro):    {metrics.recall_macro:.4f}")
    print(f"  F1 (macro):        {metrics.f1_macro:.4f}")
    print(f"  F1 (weighted):     {metrics.f1_weighted:.4f}")
    if metrics.auc_macro:
        print(f"  AUC (macro):       {metrics.auc_macro:.4f}")
    print(f"  MCC:               {metrics.mcc:.4f}")
    print(f"  Cohen's Kappa:     {metrics.kappa:.4f}")

    # Bootstrap confidence intervals
    if args.bootstrap:
        print("\nComputing bootstrap confidence intervals...")
        ci_metrics = compute_metrics_with_confidence(all_labels, all_preds)

        print("\nMetrics with 95% Confidence Intervals:")
        for metric_name, (value, lower, upper) in ci_metrics.items():
            print(f"  {metric_name}: {value:.4f} [{lower:.4f}, {upper:.4f}]")

    # Classification report
    print("\nPer-Class Classification Report:")
    print(print_classification_report(all_labels, all_preds, dataset.label_names))

    # Error analysis
    print("\nError Analysis:")
    error_analysis = analyze_errors(
        all_labels, all_preds,
        all_drug1_ids, all_drug2_ids,
        dataset.label_names,
        top_k=10,
    )

    print(f"  Total errors: {error_analysis['total_errors']}")
    print(f"  Error rate: {error_analysis['error_rate']:.2%}")

    print("\n  Most confused class pairs:")
    for pair in error_analysis['confusion_pairs'][:5]:
        print(f"    {pair['true_name']} -> {pair['pred_name']}: {pair['count']} times")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Confusion matrix
    plot_confusion_matrix(
        metrics.confusion_matrix,
        class_names=[dataset.label_names.get(i, f"Class {i}") for i in range(dataset.num_classes)],
        normalize=True,
        title="Confusion Matrix (Normalized)",
        save_path=str(output_dir / "confusion_matrix.png"),
        top_k=20,
    )
    print(f"  Saved: confusion_matrix.png")

    # ROC curves
    plot_roc_curves(
        all_labels, all_probs,
        class_names=dataset.label_names,
        top_k=10,
        save_path=str(output_dir / "roc_curves.png"),
    )
    print(f"  Saved: roc_curves.png")

    # Class distribution
    plot_class_distribution(
        all_labels,
        class_names=dataset.label_names,
        top_k=20,
        save_path=str(output_dir / "class_distribution.png"),
    )
    print(f"  Saved: class_distribution.png")

    # Save metrics to JSON
    results = {
        'accuracy': float(metrics.accuracy),
        'precision_macro': float(metrics.precision_macro),
        'precision_weighted': float(metrics.precision_weighted),
        'recall_macro': float(metrics.recall_macro),
        'recall_weighted': float(metrics.recall_weighted),
        'f1_macro': float(metrics.f1_macro),
        'f1_weighted': float(metrics.f1_weighted),
        'auc_macro': float(metrics.auc_macro) if metrics.auc_macro else None,
        'mcc': float(metrics.mcc),
        'kappa': float(metrics.kappa),
        'num_samples': len(all_labels),
        'num_classes': dataset.num_classes,
        'error_analysis': error_analysis,
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: metrics.json")

    # Save predictions
    if args.save_predictions:
        import pandas as pd

        predictions_df = pd.DataFrame({
            'drug1_id': all_drug1_ids,
            'drug2_id': all_drug2_ids,
            'true_label': all_labels,
            'pred_label': all_preds,
            'true_name': [dataset.label_names.get(l, f"Class {l}") for l in all_labels],
            'pred_name': [dataset.label_names.get(l, f"Class {l}") for l in all_preds],
            'correct': all_labels == all_preds,
            'confidence': all_probs.max(axis=1),
        })

        predictions_df.to_csv(output_dir / "predictions.csv", index=False)
        print(f"  Saved: predictions.csv")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
