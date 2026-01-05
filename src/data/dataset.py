"""
DDI Dataset Module

PyTorch Geometric Dataset classes for Drug-Drug Interaction prediction.
Supports multiple data sources including DrugBank and custom datasets.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable, Union
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch.utils.data import Dataset as TorchDataset
import pickle
import hashlib

from .featurizers import MoleculeFeaturizer, smiles_to_graph


class DDIPairData:
    """
    Container for a drug-drug interaction pair.

    Attributes:
        drug1: Graph data for first drug
        drug2: Graph data for second drug
        label: Interaction label (class index)
        drug1_id: Identifier for first drug
        drug2_id: Identifier for second drug
    """

    def __init__(
        self,
        drug1: Data,
        drug2: Data,
        label: int,
        drug1_id: Optional[str] = None,
        drug2_id: Optional[str] = None,
        drug1_kg: Optional[torch.Tensor] = None,
        drug2_kg: Optional[torch.Tensor] = None,
    ):
        self.drug1 = drug1
        self.drug2 = drug2
        self.label = label
        self.drug1_id = drug1_id
        self.drug2_id = drug2_id
        self.drug1_kg = drug1_kg
        self.drug2_kg = drug2_kg

    def to(self, device):
        """Move data to device."""
        self.drug1 = self.drug1.to(device)
        self.drug2 = self.drug2.to(device)
        if self.drug1_kg is not None:
            self.drug1_kg = self.drug1_kg.to(device)
        if self.drug2_kg is not None:
            self.drug2_kg = self.drug2_kg.to(device)
        return self


class DDIDataset(TorchDataset):
    """
    PyTorch Dataset for Drug-Drug Interaction pairs.

    Loads data from TDC (Therapeutics Data Commons) or custom sources.
    Caches molecular graphs for efficiency.
    """

    def __init__(
        self,
        data_source: str = 'drugbank',
        split: str = 'train',
        root: str = 'data',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        use_cache: bool = True,
        featurizer: Optional[MoleculeFeaturizer] = None,
        kg_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize DDI Dataset.

        Args:
            data_source: Data source ('drugbank', 'twosides', or path to CSV)
            split: Data split ('train', 'valid', 'test')
            root: Root directory for data
            transform: Optional transform function
            pre_transform: Optional pre-transform function
            use_cache: Whether to cache molecular graphs
            featurizer: MoleculeFeaturizer instance
            kg_embeddings: Optional knowledge graph embeddings dict
        """
        super().__init__()

        self.data_source = data_source
        self.split = split
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.use_cache = use_cache
        self.featurizer = featurizer or MoleculeFeaturizer()
        self.kg_embeddings = kg_embeddings

        # Create directories
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(root, 'raw'), exist_ok=True)

        # Load data
        self.data_df = self._load_data()

        # Build graph cache
        self.graph_cache: Dict[str, Data] = {}
        if use_cache:
            self._build_cache()

        # Convert labels to 0-indexed if needed (before getting label info)
        if self.data_df['Y'].min() == 1:
            self.data_df['Y'] = self.data_df['Y'] - 1

        # Get label information
        self.num_classes = self.data_df['Y'].nunique()
        self.label_names = self._get_label_names()

    def _load_data(self) -> pd.DataFrame:
        """Load data from source."""
        cache_path = os.path.join(
            self.root, 'processed',
            f'{self.data_source}_{self.split}.csv'
        )

        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)

        # Load from TDC
        if self.data_source.lower() in ['drugbank', 'twosides']:
            try:
                from tdc.multi_pred import DDI
                data = DDI(name=self.data_source.capitalize())
                splits = data.get_split()
                df = splits[self.split]
                df.to_csv(cache_path, index=False)
                return df
            except ImportError:
                raise ImportError("PyTDC is required for loading datasets. Install with: pip install PyTDC")
        else:
            # Load from custom CSV
            if os.path.exists(self.data_source):
                return pd.read_csv(self.data_source)
            else:
                raise FileNotFoundError(f"Data source not found: {self.data_source}")

    def _build_cache(self):
        """Build cache of molecular graphs."""
        cache_file = os.path.join(
            self.root, 'processed',
            f'{self.data_source}_graphs.pkl'
        )
        failed_file = os.path.join(
            self.root, 'processed',
            f'{self.data_source}_failed_smiles.pkl'
        )

        # Load existing cache if available
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.graph_cache = pickle.load(f)
        if os.path.exists(failed_file):
            with open(failed_file, 'rb') as f:
                failed_smiles = pickle.load(f)
        else:
            failed_smiles = set()

        # Get unique SMILES from current dataset
        all_smiles = set(self.data_df['Drug1'].unique()) | set(self.data_df['Drug2'].unique())

        # Find SMILES not yet in cache or failed set
        missing_smiles = all_smiles - set(self.graph_cache.keys()) - failed_smiles

        if missing_smiles:
            print(f"Processing {len(missing_smiles)} new molecules for {self.split} split...")

            from tqdm import tqdm
            for smiles in tqdm(missing_smiles, desc="Converting SMILES"):
                graph = smiles_to_graph(smiles)
                if graph is not None:
                    self.graph_cache[smiles] = graph
                else:
                    failed_smiles.add(smiles)

            # Save updated cache and failed SMILES
            with open(cache_file, 'wb') as f:
                pickle.dump(self.graph_cache, f)
            with open(failed_file, 'wb') as f:
                pickle.dump(failed_smiles, f)

            print(f"Cache updated: {len(self.graph_cache)} graphs, {len(failed_smiles)} failed")

        # Filter out pairs with invalid SMILES
        self._filter_invalid_pairs(failed_smiles)

    def _filter_invalid_pairs(self, failed_smiles: set):
        """Remove DDI pairs that contain invalid SMILES."""
        if not failed_smiles:
            return

        original_len = len(self.data_df)
        mask = ~(self.data_df['Drug1'].isin(failed_smiles) | self.data_df['Drug2'].isin(failed_smiles))
        self.data_df = self.data_df[mask].reset_index(drop=True)
        filtered_count = original_len - len(self.data_df)

        if filtered_count > 0:
            print(f"Filtered out {filtered_count} pairs with invalid SMILES ({len(self.data_df)} remaining)")

    def _get_label_names(self) -> Dict[int, str]:
        """Get mapping from label indices to names."""
        if 'Y_Name' in self.data_df.columns:
            unique_labels = self.data_df[['Y', 'Y_Name']].drop_duplicates()
            return dict(zip(unique_labels['Y'], unique_labels['Y_Name']))
        return {i: f"Interaction_{i}" for i in range(self.num_classes)}

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> DDIPairData:
        """Get a drug-drug interaction pair."""
        row = self.data_df.iloc[idx]

        smiles1 = row['Drug1']
        smiles2 = row['Drug2']
        label = int(row['Y'])

        # Get drug IDs if available
        drug1_id = row.get('Drug1_ID', smiles1)
        drug2_id = row.get('Drug2_ID', smiles2)

        # Get molecular graphs from cache
        # Invalid SMILES should have been filtered out during dataset loading
        if smiles1 not in self.graph_cache:
            raise KeyError(f"SMILES not in cache (should have been filtered): {smiles1[:50]}...")
        if smiles2 not in self.graph_cache:
            raise KeyError(f"SMILES not in cache (should have been filtered): {smiles2[:50]}...")

        drug1_graph = self.graph_cache[smiles1].clone()
        drug2_graph = self.graph_cache[smiles2].clone()

        # Apply transforms
        if self.transform is not None:
            drug1_graph = self.transform(drug1_graph)
            drug2_graph = self.transform(drug2_graph)

        # Get KG embeddings if available
        drug1_kg = None
        drug2_kg = None
        if self.kg_embeddings is not None:
            drug1_kg = self.kg_embeddings.get(str(drug1_id), torch.zeros(128))
            drug2_kg = self.kg_embeddings.get(str(drug2_id), torch.zeros(128))

        return DDIPairData(
            drug1=drug1_graph,
            drug2=drug2_graph,
            label=label,
            drug1_id=str(drug1_id),
            drug2_id=str(drug2_id),
            drug1_kg=drug1_kg,
            drug2_kg=drug2_kg,
        )


class DDICollator:
    """
    Custom collator for DDI dataset that creates proper batches.
    """

    def __init__(self, follow_batch: Optional[List[str]] = None):
        self.follow_batch = follow_batch or []

    def __call__(self, batch: List[DDIPairData]) -> Dict:
        """Collate a batch of DDIPairData objects."""
        from torch_geometric.data import Batch

        drug1_list = [item.drug1 for item in batch]
        drug2_list = [item.drug2 for item in batch]
        labels = torch.tensor([item.label for item in batch], dtype=torch.long)

        # Batch the graphs
        drug1_batch = Batch.from_data_list(drug1_list)
        drug2_batch = Batch.from_data_list(drug2_list)

        result = {
            'drug1': drug1_batch,
            'drug2': drug2_batch,
            'labels': labels,
            'drug1_ids': [item.drug1_id for item in batch],
            'drug2_ids': [item.drug2_id for item in batch],
        }

        # Add KG embeddings if available
        if batch[0].drug1_kg is not None:
            result['drug1_kg'] = torch.stack([item.drug1_kg for item in batch])
            result['drug2_kg'] = torch.stack([item.drug2_kg for item in batch])

        return result


def create_data_splits(
    data_source: str = 'drugbank',
    split_type: str = 'random',
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    root: str = 'data',
) -> Tuple[DDIDataset, DDIDataset, DDIDataset]:
    """
    Create train/valid/test splits for DDI dataset.

    Args:
        data_source: Data source name or path
        split_type: Type of split ('random', 'cold_drug', 'cold_pair')
        train_ratio: Training set ratio
        valid_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        root: Root directory for data

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    np.random.seed(seed)

    # Load full data
    if data_source.lower() in ['drugbank', 'twosides']:
        from tdc.multi_pred import DDI
        data = DDI(name=data_source.capitalize())

        if split_type == 'random':
            splits = data.get_split(method='random', seed=seed)
        elif split_type == 'cold_drug':
            # Cold-start split: some drugs only in test
            splits = data.get_split(method='cold_split', column_name='Drug1_ID', seed=seed)
        else:
            splits = data.get_split(method='random', seed=seed)

        # Save splits
        processed_dir = os.path.join(root, 'processed')
        os.makedirs(processed_dir, exist_ok=True)

        for split_name, split_df in splits.items():
            split_df.to_csv(
                os.path.join(processed_dir, f'{data_source}_{split_name}.csv'),
                index=False
            )
    else:
        raise ValueError(f"Custom data source requires pre-split files")

    # Create datasets - each loads/updates the shared cache file independently
    train_dataset = DDIDataset(data_source, split='train', root=root)
    valid_dataset = DDIDataset(data_source, split='valid', root=root)
    test_dataset = DDIDataset(data_source, split='test', root=root)

    return train_dataset, valid_dataset, test_dataset


def get_dataloader(
    dataset: DDIDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for DDI dataset.

    Args:
        dataset: DDIDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory

    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=DDICollator(),
    )
