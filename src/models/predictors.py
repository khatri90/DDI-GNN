"""
DDI Predictor Architectures

Implements various prediction heads for drug-drug interaction prediction,
including Siamese networks, knowledge graph enhanced models, and
substructure attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, Dict, List

from .encoders import get_encoder, GATDrugEncoder


class DDIPredictor(nn.Module):
    """
    Basic DDI prediction model.

    Encodes two drugs using a shared GNN encoder and predicts
    interaction type using a classifier head.
    """

    def __init__(
        self,
        num_atom_features: int,
        hidden_dim: int = 128,
        num_classes: int = 86,
        encoder_type: str = 'gcn',
        num_layers: int = 3,
        dropout: float = 0.2,
        **encoder_kwargs,
    ):
        """
        Initialize DDI predictor.

        Args:
            num_atom_features: Dimension of atom features
            hidden_dim: Hidden dimension
            num_classes: Number of interaction types
            encoder_type: Type of GNN encoder
            num_layers: Number of GNN layers
            dropout: Dropout rate
            **encoder_kwargs: Additional encoder arguments
        """
        super().__init__()

        self.encoder = get_encoder(
            encoder_type,
            num_atom_features,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **encoder_kwargs,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        drug1: Batch,
        drug2: Batch,
    ) -> torch.Tensor:
        """
        Predict interaction between drug pairs.

        Args:
            drug1: Batch of first drug graphs
            drug2: Batch of second drug graphs

        Returns:
            Logits [batch_size, num_classes]
        """
        # Encode drugs
        if hasattr(self.encoder, 'forward') and 'edge_attr' in self.encoder.forward.__code__.co_varnames:
            h1 = self.encoder(drug1.x, drug1.edge_index, drug1.batch, drug1.edge_attr if hasattr(drug1, 'edge_attr') else None)
            h2 = self.encoder(drug2.x, drug2.edge_index, drug2.batch, drug2.edge_attr if hasattr(drug2, 'edge_attr') else None)
        else:
            h1 = self.encoder(drug1.x, drug1.edge_index, drug1.batch)
            h2 = self.encoder(drug2.x, drug2.edge_index, drug2.batch)

        # Handle tuple returns (e.g., from GAT with attention)
        if isinstance(h1, tuple):
            h1 = h1[0]
        if isinstance(h2, tuple):
            h2 = h2[0]

        # Concatenate and classify
        h_pair = torch.cat([h1, h2], dim=1)
        return self.classifier(h_pair)


class SiameseDDI(nn.Module):
    """
    Siamese network for DDI prediction.

    Uses shared encoder for both drugs with symmetric/antisymmetric
    combination strategies.
    """

    def __init__(
        self,
        num_atom_features: int,
        hidden_dim: int = 128,
        num_classes: int = 86,
        encoder_type: str = 'gat',
        num_layers: int = 3,
        dropout: float = 0.2,
        combination: str = 'concat',  # 'concat', 'hadamard', 'bilinear'
        **encoder_kwargs,
    ):
        """
        Initialize Siamese DDI model.

        Args:
            num_atom_features: Dimension of atom features
            hidden_dim: Hidden dimension
            num_classes: Number of interaction types
            encoder_type: Type of GNN encoder
            num_layers: Number of GNN layers
            dropout: Dropout rate
            combination: How to combine drug embeddings
            **encoder_kwargs: Additional encoder arguments
        """
        super().__init__()

        self.combination = combination

        # Shared encoder (Siamese)
        self.encoder = get_encoder(
            encoder_type,
            num_atom_features,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **encoder_kwargs,
        )

        # Combination-specific layers
        if combination == 'concat':
            classifier_input_dim = hidden_dim * 2
        elif combination == 'hadamard':
            classifier_input_dim = hidden_dim
        elif combination == 'bilinear':
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
            classifier_input_dim = hidden_dim
        else:
            raise ValueError(f"Unknown combination: {combination}")

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode_drug(self, drug: Batch) -> torch.Tensor:
        """Encode a single drug."""
        h = self.encoder(
            drug.x, drug.edge_index, drug.batch,
            drug.edge_attr if hasattr(drug, 'edge_attr') else None
        )
        if isinstance(h, tuple):
            h = h[0]
        return h

    def forward(
        self,
        drug1: Batch,
        drug2: Batch,
    ) -> torch.Tensor:
        """
        Predict interaction.

        Args:
            drug1: Batch of first drug graphs
            drug2: Batch of second drug graphs

        Returns:
            Logits [batch_size, num_classes]
        """
        h1 = self.encode_drug(drug1)
        h2 = self.encode_drug(drug2)

        # Combine embeddings
        if self.combination == 'concat':
            h_pair = torch.cat([h1, h2], dim=1)
        elif self.combination == 'hadamard':
            h_pair = h1 * h2
        elif self.combination == 'bilinear':
            h_pair = self.bilinear(h1, h2)

        return self.classifier(h_pair)


class KGEnhancedDDI(nn.Module):
    """
    Knowledge Graph Enhanced DDI Predictor.

    Combines molecular graph features with knowledge graph embeddings
    using various fusion strategies.
    """

    def __init__(
        self,
        num_atom_features: int,
        kg_embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 86,
        encoder_type: str = 'gat',
        num_layers: int = 3,
        dropout: float = 0.2,
        fusion_type: str = 'concat',  # 'concat', 'attention', 'gated'
        **encoder_kwargs,
    ):
        """
        Initialize KG-enhanced DDI model.

        Args:
            num_atom_features: Dimension of atom features
            kg_embedding_dim: Dimension of KG embeddings
            hidden_dim: Hidden dimension
            num_classes: Number of interaction types
            encoder_type: Type of GNN encoder
            num_layers: Number of GNN layers
            dropout: Dropout rate
            fusion_type: How to fuse molecular and KG features
            **encoder_kwargs: Additional encoder arguments
        """
        super().__init__()

        self.fusion_type = fusion_type

        # Molecular encoder
        self.mol_encoder = get_encoder(
            encoder_type,
            num_atom_features,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **encoder_kwargs,
        )

        # KG encoder
        self.kg_encoder = nn.Sequential(
            nn.Linear(kg_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Fusion layers
        if fusion_type == 'concat':
            fusion_dim = hidden_dim * 4  # 2 drugs * 2 modalities
            self.fusion = nn.Linear(fusion_dim, hidden_dim * 2)
        elif fusion_type == 'attention':
            self.mol_attention = nn.Linear(hidden_dim, 1)
            self.kg_attention = nn.Linear(hidden_dim, 1)
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        elif fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        drug1: Batch,
        drug2: Batch,
        drug1_kg: torch.Tensor,
        drug2_kg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict interaction using both molecular and KG features.

        Args:
            drug1: Batch of first drug graphs
            drug2: Batch of second drug graphs
            drug1_kg: KG embeddings for first drugs [batch_size, kg_dim]
            drug2_kg: KG embeddings for second drugs [batch_size, kg_dim]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Encode molecular graphs
        h1_mol = self.mol_encoder(
            drug1.x, drug1.edge_index, drug1.batch,
            drug1.edge_attr if hasattr(drug1, 'edge_attr') else None
        )
        h2_mol = self.mol_encoder(
            drug2.x, drug2.edge_index, drug2.batch,
            drug2.edge_attr if hasattr(drug2, 'edge_attr') else None
        )

        if isinstance(h1_mol, tuple):
            h1_mol = h1_mol[0]
        if isinstance(h2_mol, tuple):
            h2_mol = h2_mol[0]

        # Encode KG embeddings
        h1_kg = self.kg_encoder(drug1_kg)
        h2_kg = self.kg_encoder(drug2_kg)

        # Fusion
        if self.fusion_type == 'concat':
            h_combined = torch.cat([h1_mol, h2_mol, h1_kg, h2_kg], dim=1)
            h_fused = self.fusion(h_combined)
        elif self.fusion_type == 'attention':
            # Compute attention weights
            a1_mol = self.mol_attention(h1_mol)
            a1_kg = self.kg_attention(h1_kg)
            weights1 = F.softmax(torch.cat([a1_mol, a1_kg], dim=1), dim=1)
            h1 = weights1[:, 0:1] * h1_mol + weights1[:, 1:2] * h1_kg

            a2_mol = self.mol_attention(h2_mol)
            a2_kg = self.kg_attention(h2_kg)
            weights2 = F.softmax(torch.cat([a2_mol, a2_kg], dim=1), dim=1)
            h2 = weights2[:, 0:1] * h2_mol + weights2[:, 1:2] * h2_kg

            h_fused = self.fusion(torch.cat([h1, h2], dim=1))
        elif self.fusion_type == 'gated':
            # Gated fusion for each drug
            gate1 = self.gate(torch.cat([h1_mol, h1_kg], dim=1))
            h1 = gate1 * h1_mol + (1 - gate1) * h1_kg

            gate2 = self.gate(torch.cat([h2_mol, h2_kg], dim=1))
            h2 = gate2 * h2_mol + (1 - gate2) * h2_kg

            h_fused = self.fusion(torch.cat([h1, h2], dim=1))

        return self.classifier(h_fused)


class SubstructureAttention(nn.Module):
    """
    Attention mechanism over substructures to identify
    interaction-causing molecular fragments.
    """

    def __init__(self, hidden_dim: int):
        """
        Initialize substructure attention.

        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h1_atoms: torch.Tensor,
        h2_atoms: torch.Tensor,
        batch1: torch.Tensor,
        batch2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pairwise attention between atoms of two drugs.

        Args:
            h1_atoms: Atom embeddings for drug 1 [num_atoms1, hidden_dim]
            h2_atoms: Atom embeddings for drug 2 [num_atoms2, hidden_dim]
            batch1: Batch assignment for drug 1
            batch2: Batch assignment for drug 2

        Returns:
            Attention weights for drug 1 atoms [num_atoms1]
            Attention weights for drug 2 atoms [num_atoms2]
        """
        batch_size = batch1.max().item() + 1

        attention_1 = torch.zeros(h1_atoms.size(0), device=h1_atoms.device)
        attention_2 = torch.zeros(h2_atoms.size(0), device=h2_atoms.device)

        for b in range(batch_size):
            mask1 = batch1 == b
            mask2 = batch2 == b

            atoms1 = h1_atoms[mask1]  # [n1, hidden]
            atoms2 = h2_atoms[mask2]  # [n2, hidden]

            n1, n2 = atoms1.size(0), atoms2.size(0)

            # Compute pairwise features
            atoms1_exp = atoms1.unsqueeze(1).expand(-1, n2, -1)  # [n1, n2, hidden]
            atoms2_exp = atoms2.unsqueeze(0).expand(n1, -1, -1)  # [n1, n2, hidden]

            pair_features = torch.cat([atoms1_exp, atoms2_exp], dim=-1)  # [n1, n2, hidden*2]
            pair_attention = self.attention(pair_features).squeeze(-1)  # [n1, n2]

            # Normalize and aggregate
            pair_attention = F.softmax(pair_attention.view(-1), dim=0).view(n1, n2)

            # Marginal attention for each atom
            attention_1[mask1] = pair_attention.sum(dim=1)
            attention_2[mask2] = pair_attention.sum(dim=0)

        return attention_1, attention_2


class SubstructureAttentionDDI(nn.Module):
    """
    DDI predictor with substructure attention for interpretability.

    Identifies which molecular substructures contribute to the
    predicted interaction.
    """

    def __init__(
        self,
        num_atom_features: int,
        hidden_dim: int = 128,
        num_classes: int = 86,
        encoder_type: str = 'gat',
        num_layers: int = 3,
        dropout: float = 0.2,
        **encoder_kwargs,
    ):
        """
        Initialize substructure attention DDI model.

        Args:
            num_atom_features: Dimension of atom features
            hidden_dim: Hidden dimension
            num_classes: Number of interaction types
            encoder_type: Type of GNN encoder
            num_layers: Number of GNN layers
            dropout: Dropout rate
            **encoder_kwargs: Additional encoder arguments
        """
        super().__init__()

        # Encoder
        self.encoder = get_encoder(
            encoder_type,
            num_atom_features,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **encoder_kwargs,
        )

        # Substructure attention
        self.substructure_attention = SubstructureAttention(hidden_dim)

        # Graph-level aggregation with attention
        self.graph_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        drug1: Batch,
        drug2: Batch,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Predict interaction with optional attention weights.

        Args:
            drug1: Batch of first drug graphs
            drug2: Batch of second drug graphs
            return_attention: Whether to return attention weights

        Returns:
            Logits [batch_size, num_classes]
            Attention dict (optional)
        """
        # Get atom-level embeddings
        h1_atoms = self.encoder.input_proj(drug1.x)
        h2_atoms = self.encoder.input_proj(drug2.x)

        # Pass through GNN layers
        for i in range(self.encoder.num_layers):
            h1_new = self.encoder.convs[i](h1_atoms, drug1.edge_index)
            h1_new = self.encoder.batch_norms[i](h1_new)
            h1_new = F.relu(h1_new)

            h2_new = self.encoder.convs[i](h2_atoms, drug2.edge_index)
            h2_new = self.encoder.batch_norms[i](h2_new)
            h2_new = F.relu(h2_new)

            if self.encoder.residual:
                h1_atoms = h1_atoms + h1_new
                h2_atoms = h2_atoms + h2_new
            else:
                h1_atoms = h1_new
                h2_atoms = h2_new

        # Compute substructure attention
        attn1, attn2 = self.substructure_attention(
            h1_atoms, h2_atoms, drug1.batch, drug2.batch
        )

        # Weighted graph-level pooling
        h1_weighted = h1_atoms * attn1.unsqueeze(-1)
        h2_weighted = h2_atoms * attn2.unsqueeze(-1)

        from torch_geometric.nn import global_add_pool
        h1_graph = global_add_pool(h1_weighted, drug1.batch)
        h2_graph = global_add_pool(h2_weighted, drug2.batch)

        # Classify
        h_pair = torch.cat([h1_graph, h2_graph], dim=1)
        logits = self.classifier(h_pair)

        if return_attention:
            attention_dict = {
                'drug1_attention': attn1,
                'drug2_attention': attn2,
                'drug1_batch': drug1.batch,
                'drug2_batch': drug2.batch,
            }
            return logits, attention_dict

        return logits, None


class MCDropoutWrapper(nn.Module):
    """
    Monte Carlo Dropout wrapper for uncertainty estimation.

    Applies dropout at inference time to estimate epistemic uncertainty.
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize MC Dropout wrapper.

        Args:
            model: Base DDI prediction model
            dropout_rate: Dropout rate for MC sampling
        """
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        drug1: Batch,
        drug2: Batch,
        n_samples: int = 30,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation.

        Args:
            drug1: Batch of first drug graphs
            drug2: Batch of second drug graphs
            n_samples: Number of MC samples
            **kwargs: Additional arguments for base model

        Returns:
            Mean predictions [batch_size, num_classes]
            Uncertainty (std) [batch_size, num_classes]
        """
        self.model.train()  # Enable dropout

        predictions = []
        for _ in range(n_samples):
            logits = self.model(drug1, drug2, **kwargs)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            probs = self.dropout(probs)  # Apply dropout to outputs
            predictions.append(probs)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean_pred, uncertainty
