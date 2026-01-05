"""
Full DDI Model

Factory for creating complete DDI prediction models from configuration.
"""

import torch
import torch.nn as nn
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from .encoders import get_encoder
from .predictors import (
    DDIPredictor,
    SiameseDDI,
    KGEnhancedDDI,
    SubstructureAttentionDDI,
    MCDropoutWrapper,
)


class DDIModel(nn.Module):
    """
    Unified DDI Model wrapper.

    Provides a consistent interface for all DDI model variants.
    """

    def __init__(
        self,
        model_type: str = 'siamese',
        num_atom_features: int = 169,
        hidden_dim: int = 128,
        num_classes: int = 86,
        encoder_type: str = 'gat',
        num_layers: int = 3,
        dropout: float = 0.2,
        use_kg: bool = False,
        kg_embedding_dim: int = 128,
        use_substructure_attention: bool = False,
        use_uncertainty: bool = False,
        **kwargs,
    ):
        """
        Initialize DDI model.

        Args:
            model_type: Type of model ('basic', 'siamese', 'kg', 'substructure')
            num_atom_features: Dimension of atom features
            hidden_dim: Hidden dimension
            num_classes: Number of interaction types
            encoder_type: Type of GNN encoder
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_kg: Whether to use knowledge graph features
            kg_embedding_dim: Dimension of KG embeddings
            use_substructure_attention: Whether to use substructure attention
            use_uncertainty: Whether to enable uncertainty estimation
            **kwargs: Additional arguments
        """
        super().__init__()

        self.model_type = model_type
        self.use_kg = use_kg
        self.use_uncertainty = use_uncertainty

        # Create base model
        if use_substructure_attention:
            self.model = SubstructureAttentionDDI(
                num_atom_features=num_atom_features,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                encoder_type=encoder_type,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs,
            )
        elif use_kg:
            self.model = KGEnhancedDDI(
                num_atom_features=num_atom_features,
                kg_embedding_dim=kg_embedding_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                encoder_type=encoder_type,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs,
            )
        elif model_type == 'siamese':
            self.model = SiameseDDI(
                num_atom_features=num_atom_features,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                encoder_type=encoder_type,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs,
            )
        else:  # basic
            self.model = DDIPredictor(
                num_atom_features=num_atom_features,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                encoder_type=encoder_type,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs,
            )

        # Wrap with MC Dropout if uncertainty estimation is enabled
        if use_uncertainty:
            self.model = MCDropoutWrapper(self.model, dropout_rate=dropout)

    def forward(
        self,
        drug1,
        drug2,
        drug1_kg: Optional[torch.Tensor] = None,
        drug2_kg: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        n_samples: int = 30,
    ):
        """
        Forward pass.

        Args:
            drug1: Batch of first drug graphs
            drug2: Batch of second drug graphs
            drug1_kg: Optional KG embeddings for first drugs
            drug2_kg: Optional KG embeddings for second drugs
            return_attention: Whether to return attention weights
            n_samples: Number of MC samples for uncertainty

        Returns:
            Model outputs (logits, uncertainty, attention depending on config)
        """
        if self.use_uncertainty:
            if self.use_kg and drug1_kg is not None:
                return self.model(drug1, drug2, n_samples=n_samples,
                                drug1_kg=drug1_kg, drug2_kg=drug2_kg)
            return self.model(drug1, drug2, n_samples=n_samples)

        if self.use_kg and drug1_kg is not None:
            return self.model(drug1, drug2, drug1_kg, drug2_kg)

        if hasattr(self.model, 'forward'):
            # Check if model supports return_attention
            if 'return_attention' in self.model.forward.__code__.co_varnames:
                return self.model(drug1, drug2, return_attention=return_attention)

        return self.model(drug1, drug2)


def create_model_from_config(config_path: str) -> DDIModel:
    """
    Create model from YAML configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured DDI model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model', {})
    encoder_config = model_config.get('encoder', {})

    # Extract parameters
    model_type = model_config.get('name', 'siamese').lower()

    # Determine model type from config
    if 'kg' in model_type.lower():
        use_kg = True
        model_type = 'kg'
    else:
        use_kg = model_config.get('kg_encoder', {}).get('enabled', False)

    use_substructure = model_config.get('substructure_attention', {}).get('enabled', False)

    interp_config = config.get('interpretability', {})
    use_uncertainty = interp_config.get('uncertainty_estimation', False)

    # Feature dimensions (calculated from featurizer)
    # This matches MoleculeFeaturizer.get_atom_feature_dim() with use_chirality=True
    num_atom_features = 169  # Default for full feature set

    return DDIModel(
        model_type=model_type,
        num_atom_features=num_atom_features,
        hidden_dim=encoder_config.get('hidden_dim', 128),
        num_classes=86,  # DrugBank has 86 interaction types
        encoder_type=encoder_config.get('type', 'gat'),
        num_layers=encoder_config.get('num_layers', 3),
        dropout=encoder_config.get('dropout', 0.2),
        use_kg=use_kg,
        kg_embedding_dim=model_config.get('kg_encoder', {}).get('kg_embedding_dim', 128),
        use_substructure_attention=use_substructure,
        use_uncertainty=use_uncertainty,
        num_heads=encoder_config.get('num_heads', 4),
        residual=encoder_config.get('residual', True),
    )


def load_model(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = 'cpu',
) -> DDIModel:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config (inferred from checkpoint if None)
        device: Device to load model to

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    if config_path is None:
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("Config not found in checkpoint. Please provide config_path.")
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Create model
    model = create_model_from_config_dict(config)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def create_model_from_config_dict(config: Dict[str, Any]) -> DDIModel:
    """
    Create model from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Configured DDI model
    """
    model_config = config.get('model', {})
    encoder_config = model_config.get('encoder', {})

    model_type = model_config.get('name', 'siamese').lower()

    if 'kg' in model_type.lower():
        use_kg = True
    else:
        use_kg = model_config.get('kg_encoder', {}).get('enabled', False)

    use_substructure = model_config.get('substructure_attention', {}).get('enabled', False)

    interp_config = config.get('interpretability', {})
    use_uncertainty = interp_config.get('uncertainty_estimation', False)

    return DDIModel(
        model_type=model_type,
        num_atom_features=169,
        hidden_dim=encoder_config.get('hidden_dim', 128),
        num_classes=86,
        encoder_type=encoder_config.get('type', 'gat'),
        num_layers=encoder_config.get('num_layers', 3),
        dropout=encoder_config.get('dropout', 0.2),
        use_kg=use_kg,
        kg_embedding_dim=model_config.get('kg_encoder', {}).get('kg_embedding_dim', 128),
        use_substructure_attention=use_substructure,
        use_uncertainty=use_uncertainty,
        num_heads=encoder_config.get('num_heads', 4),
        residual=encoder_config.get('residual', True),
    )
