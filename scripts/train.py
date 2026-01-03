#!/usr/bin/env python
"""
Training Script for DDI-GNN

Usage:
    python scripts/train.py --config configs/best_model.yaml
    python scripts/train.py --config configs/gcn_baseline.yaml --epochs 50
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import DDIDataset, DDICollator, create_data_splits, get_dataloader
from src.data.featurizers import MoleculeFeaturizer
from src.models.full_model import DDIModel, create_model_from_config
from src.training.trainer import DDITrainer, TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DDI-GNN model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/gcn_baseline.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
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
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (fewer samples)",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        model_name = config['model'].get('name', 'ddi_model')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{model_name}_{timestamp}"

    print(f"Experiment: {experiment_name}")

    # Create datasets
    print("\nLoading datasets...")
    data_config = config.get('data', {})

    try:
        train_dataset, valid_dataset, test_dataset = create_data_splits(
            data_source=data_config.get('dataset', 'drugbank'),
            split_type=data_config.get('split_type', 'random'),
            seed=args.seed,
            root=args.data_dir,
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating dummy datasets for demonstration...")

        # Create minimal dummy data for testing
        import pandas as pd

        dummy_data = pd.DataFrame({
            'Drug1_ID': ['D1', 'D2', 'D3', 'D4', 'D5'] * 100,
            'Drug2_ID': ['D2', 'D3', 'D4', 'D5', 'D1'] * 100,
            'Drug1': ['CC(=O)OC1=CC=CC=C1C(=O)O'] * 500,  # Aspirin
            'Drug2': ['CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'] * 500,  # Ibuprofen
            'Y': np.random.randint(0, 10, 500),
        })

        # Save dummy data
        processed_dir = Path(args.data_dir) / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        for split, df in [('train', dummy_data.iloc[:350]),
                         ('valid', dummy_data.iloc[350:425]),
                         ('test', dummy_data.iloc[425:])]:
            df.to_csv(processed_dir / f'drugbank_{split}.csv', index=False)

        train_dataset = DDIDataset('drugbank', 'train', root=args.data_dir)
        valid_dataset = DDIDataset('drugbank', 'valid', root=args.data_dir)
        test_dataset = DDIDataset('drugbank', 'test', root=args.data_dir)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Debug mode: use subset
    if args.debug:
        print("\nDebug mode: using subset of data")
        train_dataset.data_df = train_dataset.data_df.head(100)
        valid_dataset.data_df = valid_dataset.data_df.head(50)
        test_dataset.data_df = test_dataset.data_df.head(50)

    # Create data loaders
    train_config = config.get('training', {})
    batch_size = train_config.get('batch_size', 64)

    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = get_dataloader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = get_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Get feature dimensions
    sample_batch = next(iter(train_loader))
    num_atom_features = sample_batch['drug1'].x.shape[1]
    num_classes = train_dataset.num_classes

    print(f"\nFeature dimensions:")
    print(f"  Atom features: {num_atom_features}")
    print(f"  Number of classes: {num_classes}")

    # Create model
    print("\nCreating model...")
    model_config = config.get('model', {})
    encoder_config = model_config.get('encoder', {})

    model = DDIModel(
        model_type=model_config.get('name', 'siamese').lower(),
        num_atom_features=num_atom_features,
        hidden_dim=encoder_config.get('hidden_dim', 128),
        num_classes=num_classes,
        encoder_type=encoder_config.get('type', 'gat'),
        num_layers=encoder_config.get('num_layers', 3),
        dropout=encoder_config.get('dropout', 0.2),
        num_heads=encoder_config.get('num_heads', 4),
        residual=encoder_config.get('residual', True),
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Create training config
    training_config = TrainingConfig(
        epochs=train_config.get('epochs', 100),
        batch_size=batch_size,
        learning_rate=train_config.get('learning_rate', 0.001),
        weight_decay=train_config.get('weight_decay', 0.0001),
        gradient_clip=train_config.get('gradient_clip', 1.0),
        scheduler_type=train_config.get('scheduler', {}).get('type', 'reduce_on_plateau'),
        scheduler_patience=train_config.get('scheduler', {}).get('patience', 10),
        scheduler_factor=train_config.get('scheduler', {}).get('factor', 0.5),
        early_stopping_patience=train_config.get('early_stopping', {}).get('patience', 20),
        early_stopping_min_delta=train_config.get('early_stopping', {}).get('min_delta', 0.001),
        loss_type=train_config.get('loss_type', 'cross_entropy'),
        label_smoothing=train_config.get('label_smoothing', 0.0),
        log_every=config.get('logging', {}).get('log_every', 10),
        save_dir=args.output_dir,
        experiment_name=experiment_name,
        use_wandb=args.wandb,
        wandb_project=config.get('logging', {}).get('wandb', {}).get('project', 'ddi-gnn'),
        save_best=True,
        save_last=True,
        device=device,
    )

    # Create trainer
    trainer = DDITrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        num_classes=num_classes,
    )

    # Train
    print("\nStarting training...")
    results = trainer.train()

    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Validation F1: {results['best_val_f1']:.4f}")
    print(f"Training Time: {results['training_time']/60:.2f} minutes")

    if results['test_metrics']:
        print(f"\nTest Results:")
        print(f"  Accuracy: {results['test_metrics']['test_accuracy']:.4f}")
        print(f"  F1 Macro: {results['test_metrics']['test_f1_macro']:.4f}")
        print(f"  F1 Weighted: {results['test_metrics']['test_f1_weighted']:.4f}")

    print(f"\nOutputs saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
