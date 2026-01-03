"""
Loss Functions for DDI Prediction

Implements various loss functions suitable for multi-class classification
with class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reduces the relative loss for well-classified examples,
    focusing training on hard examples.

    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Class weights [num_classes]
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Prevents the model from becoming too confident and
    improves generalization.
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = 'mean',
    ):
        """
        Initialize Label Smoothing Loss.

        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0 = no smoothing)
            reduction: Reduction method
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = -torch.sum(smooth_targets * log_probs, dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with class weights computed from data.
    """

    def __init__(
        self,
        class_counts: Optional[torch.Tensor] = None,
        num_classes: int = 86,
        reduction: str = 'mean',
    ):
        """
        Initialize Weighted Cross-Entropy Loss.

        Args:
            class_counts: Count of samples per class
            num_classes: Number of classes
            reduction: Reduction method
        """
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

        if class_counts is not None:
            # Inverse frequency weighting
            weights = 1.0 / (class_counts + 1e-8)
            weights = weights / weights.sum() * num_classes
            self.register_buffer('weights', weights)
        else:
            self.weights = None

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        return F.cross_entropy(
            inputs, targets,
            weight=self.weights,
            reduction=self.reduction,
        )


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning drug representations.

    Encourages drugs with similar interactions to have similar embeddings.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = 'mean',
    ):
        """
        Initialize Contrastive Loss.

        Args:
            temperature: Temperature for softmax
            reduction: Reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Drug embeddings [batch_size, hidden_dim]
            labels: Interaction labels [batch_size]

        Returns:
            Loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same label)
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()

        # Remove diagonal
        mask = mask - torch.eye(mask.size(0), device=mask.device)

        # Compute loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        loss = -mean_log_prob

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def get_loss_function(
    loss_type: str = 'cross_entropy',
    num_classes: int = 86,
    class_counts: Optional[torch.Tensor] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get loss function by type.

    Args:
        loss_type: Loss type ('cross_entropy', 'focal', 'label_smoothing', 'weighted')
        num_classes: Number of classes
        class_counts: Class counts for weighting
        **kwargs: Additional arguments

    Returns:
        Loss function module
    """
    loss_type = loss_type.lower()

    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(gamma=gamma)
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingLoss(num_classes, smoothing=smoothing)
    elif loss_type == 'weighted':
        return WeightedCrossEntropyLoss(class_counts, num_classes)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
