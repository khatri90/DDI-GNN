"""
Knowledge Graph Module

Handles loading and encoding knowledge graph embeddings for drugs.
Supports Hetionet and other biomedical knowledge graphs.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import json
import pickle


class KnowledgeGraphEncoder(nn.Module):
    """
    Encoder for knowledge graph drug embeddings.

    Takes pre-computed KG embeddings and projects them to the model's
    hidden dimension.
    """

    def __init__(
        self,
        kg_embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Initialize KG encoder.

        Args:
            kg_embedding_dim: Dimension of input KG embeddings
            hidden_dim: Output hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(kg_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, kg_embedding: torch.Tensor) -> torch.Tensor:
        """
        Encode KG embedding.

        Args:
            kg_embedding: Knowledge graph embedding [batch_size, kg_dim]

        Returns:
            Encoded embedding [batch_size, hidden_dim]
        """
        return self.encoder(kg_embedding)


def load_hetionet_embeddings(
    file_path: str,
    embedding_dim: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Load pre-computed Hetionet drug embeddings.

    Args:
        file_path: Path to embeddings file
        embedding_dim: Expected embedding dimension

    Returns:
        Dictionary mapping drug IDs to embeddings
    """
    if not os.path.exists(file_path):
        print(f"Warning: Hetionet embeddings not found at {file_path}")
        print("Generating random embeddings for demonstration...")
        return {}

    # Try different file formats
    if file_path.endswith('.pt'):
        data = torch.load(file_path)
        return data
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return {k: torch.tensor(v, dtype=torch.float) for k, v in data.items()}
    elif file_path.endswith('.npz'):
        data = np.load(file_path)
        return {k: torch.tensor(v, dtype=torch.float) for k, v in data.items()}
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def compute_kg_embeddings_from_graph(
    kg_triples_path: str,
    method: str = 'TransE',
    embedding_dim: int = 128,
    output_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute knowledge graph embeddings from triples.

    Args:
        kg_triples_path: Path to KG triples file (CSV with head, relation, tail)
        method: Embedding method ('TransE', 'RotatE', 'ComplEx')
        embedding_dim: Embedding dimension
        output_path: Optional path to save embeddings

    Returns:
        Dictionary mapping entity IDs to embeddings
    """
    # Load triples
    df = pd.read_csv(kg_triples_path)

    # Get unique entities
    all_entities = set(df['head'].unique()) | set(df['tail'].unique())
    entity_to_idx = {e: i for i, e in enumerate(all_entities)}

    # Get unique relations
    all_relations = df['relation'].unique()
    relation_to_idx = {r: i for i, r in enumerate(all_relations)}

    num_entities = len(entity_to_idx)
    num_relations = len(relation_to_idx)

    print(f"Knowledge Graph Statistics:")
    print(f"  Entities: {num_entities}")
    print(f"  Relations: {num_relations}")
    print(f"  Triples: {len(df)}")

    # Initialize embeddings
    entity_embeddings = nn.Embedding(num_entities, embedding_dim)
    relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    nn.init.xavier_uniform_(entity_embeddings.weight)
    nn.init.xavier_uniform_(relation_embeddings.weight)

    # Prepare training data
    heads = torch.tensor([entity_to_idx[h] for h in df['head']])
    relations = torch.tensor([relation_to_idx[r] for r in df['relation']])
    tails = torch.tensor([entity_to_idx[t] for t in df['tail']])

    # Simple TransE training
    if method == 'TransE':
        embeddings = train_transe(
            entity_embeddings, relation_embeddings,
            heads, relations, tails,
            num_entities, embedding_dim,
        )
    else:
        # Default to random for other methods
        embeddings = {
            entity: entity_embeddings.weight[idx].detach()
            for entity, idx in entity_to_idx.items()
        }

    if output_path:
        torch.save(embeddings, output_path)
        print(f"Embeddings saved to {output_path}")

    return embeddings


def train_transe(
    entity_embeddings: nn.Embedding,
    relation_embeddings: nn.Embedding,
    heads: torch.Tensor,
    relations: torch.Tensor,
    tails: torch.Tensor,
    num_entities: int,
    embedding_dim: int,
    epochs: int = 100,
    lr: float = 0.01,
    margin: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Train TransE embeddings.

    Args:
        entity_embeddings: Entity embedding layer
        relation_embeddings: Relation embedding layer
        heads, relations, tails: Triple tensors
        num_entities: Number of entities
        embedding_dim: Embedding dimension
        epochs: Training epochs
        lr: Learning rate
        margin: Margin for loss

    Returns:
        Trained entity embeddings
    """
    optimizer = torch.optim.Adam(
        list(entity_embeddings.parameters()) + list(relation_embeddings.parameters()),
        lr=lr
    )

    from tqdm import tqdm
    for epoch in tqdm(range(epochs), desc="Training TransE"):
        optimizer.zero_grad()

        # Get embeddings
        h = entity_embeddings(heads)
        r = relation_embeddings(relations)
        t = entity_embeddings(tails)

        # Positive score
        pos_score = torch.norm(h + r - t, p=2, dim=1)

        # Generate negative samples (corrupt tail)
        neg_tails = torch.randint(0, num_entities, (len(tails),))
        t_neg = entity_embeddings(neg_tails)
        neg_score = torch.norm(h + r - t_neg, p=2, dim=1)

        # Margin-based loss
        loss = torch.mean(torch.relu(pos_score - neg_score + margin))

        loss.backward()
        optimizer.step()

        # Normalize embeddings
        with torch.no_grad():
            entity_embeddings.weight.data = nn.functional.normalize(
                entity_embeddings.weight.data, p=2, dim=1
            )

    # Return embeddings as dictionary
    idx_to_entity = {v: k for k, v in entity_to_idx.items()}
    return {
        idx_to_entity[i]: entity_embeddings.weight[i].detach()
        for i in range(num_entities)
    }


class DrugKGMatcher:
    """
    Matches drugs from DDI dataset to knowledge graph entities.

    Handles different drug identifiers (DrugBank IDs, names, SMILES).
    """

    def __init__(
        self,
        kg_embeddings: Dict[str, torch.Tensor],
        drugbank_mapping_path: Optional[str] = None,
    ):
        """
        Initialize matcher.

        Args:
            kg_embeddings: Pre-computed KG embeddings
            drugbank_mapping_path: Path to DrugBank ID mapping file
        """
        self.kg_embeddings = kg_embeddings
        self.embedding_dim = next(iter(kg_embeddings.values())).shape[0] if kg_embeddings else 128

        self.drug_to_kg = {}
        if drugbank_mapping_path and os.path.exists(drugbank_mapping_path):
            self._load_mapping(drugbank_mapping_path)

    def _load_mapping(self, path: str):
        """Load drug to KG entity mapping."""
        if path.endswith('.json'):
            with open(path, 'r') as f:
                self.drug_to_kg = json.load(f)
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
            self.drug_to_kg = dict(zip(df['drug_id'], df['kg_id']))

    def get_embedding(self, drug_id: str) -> torch.Tensor:
        """
        Get KG embedding for a drug.

        Args:
            drug_id: Drug identifier

        Returns:
            KG embedding tensor
        """
        # Try direct match
        if drug_id in self.kg_embeddings:
            return self.kg_embeddings[drug_id]

        # Try mapped ID
        if drug_id in self.drug_to_kg:
            kg_id = self.drug_to_kg[drug_id]
            if kg_id in self.kg_embeddings:
                return self.kg_embeddings[kg_id]

        # Return zero embedding for unknown drugs
        return torch.zeros(self.embedding_dim)

    def get_batch_embeddings(
        self,
        drug_ids: List[str],
    ) -> torch.Tensor:
        """
        Get KG embeddings for a batch of drugs.

        Args:
            drug_ids: List of drug identifiers

        Returns:
            Batch of KG embeddings [batch_size, embedding_dim]
        """
        embeddings = [self.get_embedding(drug_id) for drug_id in drug_ids]
        return torch.stack(embeddings)


def download_hetionet(save_dir: str = 'data/raw') -> str:
    """
    Download Hetionet knowledge graph data.

    Args:
        save_dir: Directory to save data

    Returns:
        Path to downloaded file
    """
    import urllib.request

    os.makedirs(save_dir, exist_ok=True)

    url = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2"
    save_path = os.path.join(save_dir, "hetionet-v1.0.json.bz2")

    if not os.path.exists(save_path):
        print(f"Downloading Hetionet from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded to {save_path}")

    return save_path


def process_hetionet(
    input_path: str,
    output_dir: str = 'data/processed',
    embedding_dim: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Process Hetionet and compute drug embeddings.

    Args:
        input_path: Path to Hetionet JSON file
        output_dir: Directory for processed files
        embedding_dim: Embedding dimension

    Returns:
        Drug embeddings dictionary
    """
    import bz2

    os.makedirs(output_dir, exist_ok=True)

    # Load Hetionet
    print("Loading Hetionet...")
    with bz2.open(input_path, 'rt') as f:
        data = json.load(f)

    # Extract drug nodes
    drugs = {}
    for node in data['nodes']:
        if node['kind'] == 'Compound':
            drugs[node['identifier']] = node.get('name', node['identifier'])

    print(f"Found {len(drugs)} compounds in Hetionet")

    # Extract edges involving drugs
    triples = []
    for edge in data['edges']:
        source = edge['source']
        target = edge['target']
        kind = edge['kind']

        # Check if either endpoint is a drug
        if source[1] in drugs or target[1] in drugs:
            triples.append({
                'head': f"{source[0]}:{source[1]}",
                'relation': kind,
                'tail': f"{target[0]}:{target[1]}",
            })

    print(f"Extracted {len(triples)} drug-related triples")

    # Save triples
    triples_df = pd.DataFrame(triples)
    triples_path = os.path.join(output_dir, 'hetionet_drug_triples.csv')
    triples_df.to_csv(triples_path, index=False)

    # Compute embeddings
    embeddings_path = os.path.join(output_dir, 'kg_embeddings.pt')
    embeddings = compute_kg_embeddings_from_graph(
        triples_path,
        method='TransE',
        embedding_dim=embedding_dim,
        output_path=embeddings_path,
    )

    # Filter to only drug embeddings
    drug_embeddings = {}
    for drug_id in drugs.keys():
        key = f"Compound:{drug_id}"
        if key in embeddings:
            drug_embeddings[drug_id] = embeddings[key]

    print(f"Generated embeddings for {len(drug_embeddings)} drugs")

    return drug_embeddings
