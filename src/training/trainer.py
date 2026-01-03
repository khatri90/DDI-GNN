"""
DDI Trainer

Comprehensive training loop with logging, checkpointing, and early stopping.
"""

import os
import time
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Any, Callable, List
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

from .losses import get_loss_function


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0

    # Scheduler
    scheduler_type: str = 'reduce_on_plateau'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001

    # Loss
    loss_type: str = 'cross_entropy'
    label_smoothing: float = 0.0

    # Logging
    log_every: int = 10
    save_dir: str = 'outputs'
    experiment_name: str = 'ddi_experiment'
    use_wandb: bool = False
    wandb_project: str = 'ddi-gnn'

    # Checkpointing
    save_best: bool = True
    save_last: bool = True
    save_every: int = 0  # Save every N epochs (0 = disabled)

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class DDITrainer:
    """
    Trainer for DDI prediction models.

    Handles training loop, evaluation, logging, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        num_classes: int = 86,
        class_counts: Optional[torch.Tensor] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: DDI prediction model
            config: Training configuration
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Optional test data loader
            num_classes: Number of interaction classes
            class_counts: Class counts for weighted loss
        """
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = config.device

        # Setup loss function
        self.criterion = get_loss_function(
            config.loss_type,
            num_classes=num_classes,
            class_counts=class_counts,
            smoothing=config.label_smoothing,
        )

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Setup output directory
        self.output_dir = Path(config.save_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1_macro': [],
            'val_f1_weighted': [],
            'learning_rate': [],
        }

        # W&B logging
        self.wandb_run = None
        if config.use_wandb:
            self._setup_wandb()

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr,
            )
        elif self.config.scheduler_type == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.scheduler_min_lr,
            )
        elif self.config.scheduler_type == 'cosine_annealing_warm_restarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=50,
                T_mult=2,
                eta_min=self.config.scheduler_min_lr,
            )
        else:
            return None

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=vars(self.config),
            )
        except ImportError:
            print("Warning: wandb not installed. Disabling W&B logging.")
            self.config.use_wandb = False

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            drug1 = batch['drug1'].to(self.device)
            drug2 = batch['drug2'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if 'drug1_kg' in batch and batch['drug1_kg'] is not None:
                drug1_kg = batch['drug1_kg'].to(self.device)
                drug2_kg = batch['drug2_kg'].to(self.device)
                outputs = self.model(drug1, drug2, drug1_kg, drug2_kg)
            else:
                outputs = self.model(drug1, drug2)

            # Handle tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Log to W&B
            if self.config.use_wandb and batch_idx % self.config.log_every == 0:
                import wandb
                wandb.log({
                    'train/loss': loss.item(),
                    'train/step': self.global_step,
                })

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        prefix: str = 'val',
    ) -> Dict[str, float]:
        """
        Evaluate model on data loader.

        Args:
            loader: Data loader
            prefix: Metric prefix ('val' or 'test')

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score,
            recall_score, roc_auc_score, confusion_matrix
        )

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Evaluating ({prefix})"):
            drug1 = batch['drug1'].to(self.device)
            drug2 = batch['drug2'].to(self.device)
            labels = batch['labels'].to(self.device)

            if 'drug1_kg' in batch and batch['drug1_kg'] is not None:
                drug1_kg = batch['drug1_kg'].to(self.device)
                drug2_kg = batch['drug2_kg'].to(self.device)
                outputs = self.model(drug1, drug2, drug1_kg, drug2_kg)
            else:
                outputs = self.model(drug1, drug2)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        metrics = {
            f'{prefix}_loss': total_loss / len(loader),
            f'{prefix}_accuracy': accuracy_score(all_labels, all_preds),
            f'{prefix}_f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            f'{prefix}_f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            f'{prefix}_precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            f'{prefix}_recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        }

        # AUC-ROC (only if we have multiple classes represented)
        try:
            if len(np.unique(all_labels)) > 1:
                metrics[f'{prefix}_auc_macro'] = roc_auc_score(
                    all_labels, all_probs,
                    multi_class='ovr', average='macro'
                )
        except ValueError:
            pass  # Not enough classes for AUC

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Training history and best metrics
        """
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.valid_loader.dataset)}")

        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)

            # Validate
            val_metrics = self.evaluate(self.valid_loader, prefix='val')
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_accuracy'].append(val_metrics['val_accuracy'])
            self.history['val_f1_macro'].append(val_metrics['val_f1_macro'])
            self.history['val_f1_weighted'].append(val_metrics['val_f1_weighted'])

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Print progress
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            print(f"  Val F1 Macro: {val_metrics['val_f1_macro']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # W&B logging
            if self.config.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'learning_rate': current_lr,
                    **val_metrics,
                })

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_f1_macro'])
                else:
                    self.scheduler.step()

            # Check for best model
            current_metric = val_metrics['val_f1_macro']
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.patience_counter = 0

                if self.config.save_best:
                    self.save_checkpoint('best_model.pt')
                    print(f"  New best model saved! (F1: {self.best_val_metric:.4f})")
            else:
                self.patience_counter += 1

            # Save last model
            if self.config.save_last:
                self.save_checkpoint('last_model.pt')

            # Save periodic checkpoint
            if self.config.save_every > 0 and epoch % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation F1: {self.best_val_metric:.4f}")

        # Final test evaluation
        test_metrics = None
        if self.test_loader is not None:
            print("\nEvaluating on test set...")
            # Load best model
            self.load_checkpoint('best_model.pt')
            test_metrics = self.evaluate(self.test_loader, prefix='test')
            print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
            print(f"Test F1 Macro: {test_metrics['test_f1_macro']:.4f}")

        # Save history
        self.save_history()

        return {
            'history': self.history,
            'best_val_f1': self.best_val_metric,
            'test_metrics': test_metrics,
            'training_time': total_time,
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': self.best_val_metric,
            'config': vars(self.config),
            'history': self.history,
        }
        path = self.output_dir / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.output_dir / filename
        if not path.exists():
            print(f"Warning: Checkpoint {path} not found")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)

    def save_history(self):
        """Save training history to JSON."""
        path = self.output_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Also save as plots
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Loss plot
            axes[0, 0].plot(self.history['train_loss'], label='Train')
            axes[0, 0].plot(self.history['val_loss'], label='Validation')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()

            # Accuracy plot
            axes[0, 1].plot(self.history['val_accuracy'])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Validation Accuracy')

            # F1 plot
            axes[1, 0].plot(self.history['val_f1_macro'], label='Macro')
            axes[1, 0].plot(self.history['val_f1_weighted'], label='Weighted')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_title('Validation F1 Scores')
            axes[1, 0].legend()

            # Learning rate plot
            axes[1, 1].plot(self.history['learning_rate'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'training_curves.png', dpi=150)
            plt.close()

        except ImportError:
            print("matplotlib not available, skipping plot generation")
