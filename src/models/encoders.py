"""
GNN Encoder Architectures

Implements various graph neural network architectures for encoding
molecular graphs into fixed-dimensional embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    NNConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    Set2Set,
    MessagePassing,
)
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, List


class GCNDrugEncoder(nn.Module):
    """
    Graph Convolutional Network encoder for molecular graphs.

    Uses GCN layers with normalization and residual connections.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        pool_type: str = 'mean',
        residual: bool = True,
    ):
        """
        Initialize GCN encoder.

        Args:
            num_features: Input node feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            pool_type: Pooling type ('mean', 'max', 'add', 'set2set')
            residual: Use residual connections
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Pooling
        self.pool_type = pool_type
        if pool_type == 'set2set':
            self.pool = Set2Set(hidden_dim, processing_steps=3)
            self.pool_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.pool = self._get_pool_fn(pool_type)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def _get_pool_fn(self, pool_type: str):
        """Get pooling function."""
        if pool_type == 'mean':
            return global_mean_pool
        elif pool_type == 'max':
            return global_max_pool
        elif pool_type == 'add':
            return global_add_pool
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            edge_attr: Edge features (unused in GCN)

        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        # Input projection
        h = self.input_proj(x)

        # GCN layers
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index)
            h_new = self.batch_norms[i](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            if self.residual:
                h = h + h_new
            else:
                h = h_new

        # Pooling
        if self.pool_type == 'set2set':
            graph_emb = self.pool(h, batch)
            graph_emb = self.pool_proj(graph_emb)
        else:
            graph_emb = self.pool(h, batch)

        # Output projection
        return self.output_proj(graph_emb)


class GATDrugEncoder(nn.Module):
    """
    Graph Attention Network encoder for molecular graphs.

    Uses multi-head attention to weight neighbor contributions.
    Returns both graph embedding and atom-level attention weights.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        pool_type: str = 'mean',
        residual: bool = True,
        use_edge_features: bool = False,
        edge_dim: Optional[int] = None,
    ):
        """
        Initialize GAT encoder.

        Args:
            num_features: Input node feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            pool_type: Pooling type
            residual: Use residual connections
            use_edge_features: Whether to use edge features
            edge_dim: Edge feature dimension (if using edge features)
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.use_edge_features = use_edge_features

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers - 1):
            if use_edge_features and edge_dim is not None:
                self.convs.append(
                    GATConv(
                        hidden_dim, hidden_dim // num_heads,
                        heads=num_heads, dropout=dropout,
                        edge_dim=edge_dim, concat=True
                    )
                )
            else:
                self.convs.append(
                    GATConv(
                        hidden_dim, hidden_dim // num_heads,
                        heads=num_heads, dropout=dropout, concat=True
                    )
                )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Last layer with single head
        if use_edge_features and edge_dim is not None:
            self.convs.append(
                GATConv(
                    hidden_dim, hidden_dim,
                    heads=1, dropout=dropout,
                    edge_dim=edge_dim, concat=False
                )
            )
        else:
            self.convs.append(
                GATConv(
                    hidden_dim, hidden_dim,
                    heads=1, dropout=dropout, concat=False
                )
            )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Pooling
        self.pool_type = pool_type
        if pool_type == 'set2set':
            self.pool = Set2Set(hidden_dim, processing_steps=3)
            self.pool_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.pool = self._get_pool_fn(pool_type)

        # Attention readout
        self.attention_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _get_pool_fn(self, pool_type: str):
        if pool_type == 'mean':
            return global_mean_pool
        elif pool_type == 'max':
            return global_max_pool
        elif pool_type == 'add':
            return global_add_pool
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            edge_attr: Edge features [num_edges, edge_dim]
            return_attention: Return atom attention weights

        Returns:
            Graph embeddings [batch_size, hidden_dim]
            Attention weights [num_nodes] (optional)
        """
        # Input projection
        h = self.input_proj(x)

        # GAT layers
        attention_weights = None
        for i in range(self.num_layers):
            if self.use_edge_features and edge_attr is not None:
                h_new, attention = self.convs[i](
                    h, edge_index, edge_attr=edge_attr,
                    return_attention_weights=True
                )
            else:
                h_new, attention = self.convs[i](
                    h, edge_index,
                    return_attention_weights=True
                )

            h_new = self.batch_norms[i](h_new)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            if self.residual and h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new

            if i == self.num_layers - 1:
                attention_weights = attention

        # Compute atom importance scores
        atom_scores = self.attention_readout(h).squeeze(-1)
        atom_attention = torch.zeros_like(atom_scores)

        # Normalize within each graph
        for i in range(batch.max().item() + 1):
            mask = batch == i
            atom_attention[mask] = F.softmax(atom_scores[mask], dim=0)

        # Pooling
        if self.pool_type == 'set2set':
            graph_emb = self.pool(h, batch)
            graph_emb = self.pool_proj(graph_emb)
        else:
            # Use attention-weighted pooling
            h_weighted = h * atom_attention.unsqueeze(-1)
            graph_emb = global_add_pool(h_weighted, batch)

        if return_attention:
            return graph_emb, atom_attention
        return graph_emb, None


class MPNNEncoder(nn.Module):
    """
    Message Passing Neural Network encoder.

    Uses neural network to compute edge-dependent messages.
    Suitable for graphs with rich edge features (like bond features).
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        """
        Initialize MPNN encoder.

        Args:
            num_node_features: Input node feature dimension
            num_edge_features: Input edge feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of message passing layers
            dropout: Dropout rate
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Node embedding
        self.node_embed = nn.Linear(num_node_features, hidden_dim)

        # Edge network for each layer
        self.edge_networks = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.grus = nn.ModuleList()

        for _ in range(num_layers):
            edge_net = nn.Sequential(
                nn.Linear(num_edge_features, hidden_dim * hidden_dim),
            )
            self.edge_networks.append(edge_net)
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr='add'))
            self.grus.append(nn.GRU(hidden_dim, hidden_dim, batch_first=True))

        # Set2Set readout
        self.set2set = Set2Set(hidden_dim, processing_steps=6)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, num_edge_features]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        # Node embedding
        h = self.node_embed(x)

        # Message passing
        for i in range(self.num_layers):
            # Convolution
            m = self.convs[i](h, edge_index, edge_attr)
            m = F.relu(m)

            # GRU update
            h_in = h.unsqueeze(0)
            m_in = m.unsqueeze(0)
            _, h = self.grus[i](m_in, h_in)
            h = h.squeeze(0)

            h = F.dropout(h, p=self.dropout, training=self.training)

        # Readout
        graph_emb = self.set2set(h, batch)
        graph_emb = self.output_proj(graph_emb)

        return graph_emb


class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network encoder.

    Powerful for distinguishing graph structures.
    Uses MLPs for node updates.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 5,
        dropout: float = 0.2,
        train_eps: bool = True,
    ):
        """
        Initialize GIN encoder.

        Args:
            num_features: Input node feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GIN layers
            dropout: Dropout rate
            train_eps: Whether to train epsilon parameter
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Pooling
        self.pool = global_add_pool

        # Output projection (combines all layer outputs)
        self.output_proj = nn.Linear(hidden_dim * (num_layers + 1), hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            edge_attr: Edge features (unused)

        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        # Input projection
        h = self.input_proj(x)

        # Collect representations from all layers
        h_list = [self.pool(h, batch)]

        # GIN layers
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            h_list.append(self.pool(h, batch))

        # Concatenate all layer representations
        h_concat = torch.cat(h_list, dim=1)

        # Output projection
        return self.output_proj(h_concat)


def get_encoder(
    encoder_type: str,
    num_features: int,
    hidden_dim: int = 128,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get encoder by type.

    Args:
        encoder_type: Encoder type ('gcn', 'gat', 'mpnn', 'gin')
        num_features: Input feature dimension
        hidden_dim: Hidden dimension
        **kwargs: Additional arguments for encoder

    Returns:
        Encoder module
    """
    encoder_type = encoder_type.lower()

    if encoder_type == 'gcn':
        return GCNDrugEncoder(num_features, hidden_dim, **kwargs)
    elif encoder_type == 'gat':
        return GATDrugEncoder(num_features, hidden_dim, **kwargs)
    elif encoder_type == 'mpnn':
        num_edge_features = kwargs.pop('num_edge_features', 13)
        return MPNNEncoder(num_features, num_edge_features, hidden_dim, **kwargs)
    elif encoder_type == 'gin':
        return GINEncoder(num_features, hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
