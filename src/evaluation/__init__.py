"""Evaluation module for DDI-GNN."""

from .metrics import compute_metrics, ClassificationMetrics
from .visualization import (
    visualize_attention,
    plot_confusion_matrix,
    visualize_molecule_attention,
    plot_roc_curves,
)
