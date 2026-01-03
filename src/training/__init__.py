"""Training module for DDI-GNN."""

from .trainer import DDITrainer, TrainingConfig
from .losses import FocalLoss, LabelSmoothingLoss, WeightedCrossEntropyLoss
