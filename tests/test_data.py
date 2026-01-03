"""Tests for data processing module."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFeaturizers:
    """Tests for molecular featurizers."""

    def test_smiles_to_graph_valid(self):
        """Test conversion of valid SMILES to graph."""
        from src.data.featurizers import smiles_to_graph

        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        graph = smiles_to_graph(smiles)

        assert graph is not None
        assert hasattr(graph, 'x')
        assert hasattr(graph, 'edge_index')
        assert graph.x.dim() == 2
        assert graph.edge_index.dim() == 2
        assert graph.edge_index.shape[0] == 2

    def test_smiles_to_graph_invalid(self):
        """Test conversion of invalid SMILES."""
        from src.data.featurizers import smiles_to_graph

        invalid_smiles = "INVALID_SMILES"
        graph = smiles_to_graph(invalid_smiles)

        assert graph is None

    def test_get_fingerprint(self):
        """Test fingerprint generation."""
        from src.data.featurizers import get_fingerprint

        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        fp = get_fingerprint(smiles, radius=2, nbits=1024)

        assert fp is not None
        assert len(fp) == 1024
        assert all(bit in [0, 1] for bit in fp)

    def test_molecule_featurizer(self):
        """Test MoleculeFeaturizer class."""
        from src.data.featurizers import MoleculeFeaturizer

        featurizer = MoleculeFeaturizer(
            use_chirality=True,
            use_stereo=True,
            fingerprint_type='morgan',
            fingerprint_radius=2,
            fingerprint_bits=1024,
        )

        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

        # Test graph conversion
        graph = featurizer.smiles_to_graph(smiles)
        assert graph is not None

        # Test fingerprint
        fp = featurizer.smiles_to_fingerprint(smiles)
        assert fp is not None
        assert len(fp) == 1024

        # Test descriptors
        desc = featurizer.smiles_to_descriptors(smiles)
        assert desc is not None
        assert 'molecular_weight' in desc
        assert 'logp' in desc

    def test_feature_dimensions(self):
        """Test feature dimension calculations."""
        from src.data.featurizers import MoleculeFeaturizer

        featurizer = MoleculeFeaturizer()

        atom_dim = featurizer.get_atom_feature_dim()
        bond_dim = featurizer.get_bond_feature_dim()

        assert atom_dim > 0
        assert bond_dim > 0


class TestDataset:
    """Tests for DDI dataset."""

    def test_ddi_pair_data(self):
        """Test DDIPairData container."""
        from src.data.dataset import DDIPairData
        from torch_geometric.data import Data

        drug1 = Data(
            x=torch.randn(5, 10),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        )
        drug2 = Data(
            x=torch.randn(7, 10),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        )

        pair = DDIPairData(
            drug1=drug1,
            drug2=drug2,
            label=5,
            drug1_id="D001",
            drug2_id="D002",
        )

        assert pair.label == 5
        assert pair.drug1_id == "D001"
        assert pair.drug2_id == "D002"

    def test_ddi_collator(self):
        """Test DDI data collator."""
        from src.data.dataset import DDICollator, DDIPairData
        from torch_geometric.data import Data

        # Create sample pairs
        pairs = []
        for i in range(4):
            drug1 = Data(
                x=torch.randn(5, 10),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            )
            drug2 = Data(
                x=torch.randn(7, 10),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
            )
            pairs.append(DDIPairData(drug1, drug2, i, f"D{i}", f"D{i+10}"))

        collator = DDICollator()
        batch = collator(pairs)

        assert 'drug1' in batch
        assert 'drug2' in batch
        assert 'labels' in batch
        assert len(batch['labels']) == 4


class TestModels:
    """Tests for model architectures."""

    def test_gcn_encoder(self):
        """Test GCN encoder."""
        from src.models.encoders import GCNDrugEncoder

        encoder = GCNDrugEncoder(
            num_features=10,
            hidden_dim=64,
            num_layers=2,
        )

        x = torch.randn(20, 10)
        edge_index = torch.randint(0, 20, (2, 40))
        batch = torch.zeros(20, dtype=torch.long)

        out = encoder(x, edge_index, batch)

        assert out.shape == (1, 64)

    def test_gat_encoder(self):
        """Test GAT encoder."""
        from src.models.encoders import GATDrugEncoder

        encoder = GATDrugEncoder(
            num_features=10,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )

        x = torch.randn(20, 10)
        edge_index = torch.randint(0, 20, (2, 40))
        batch = torch.zeros(20, dtype=torch.long)

        out, attention = encoder(x, edge_index, batch)

        assert out.shape == (1, 64)

    def test_ddi_predictor(self):
        """Test DDI predictor."""
        from src.models.predictors import DDIPredictor
        from torch_geometric.data import Data, Batch

        model = DDIPredictor(
            num_atom_features=10,
            hidden_dim=64,
            num_classes=86,
            encoder_type='gcn',
            num_layers=2,
        )

        # Create batch of drug pairs
        drug1_list = [
            Data(x=torch.randn(5, 10), edge_index=torch.randint(0, 5, (2, 8)))
            for _ in range(4)
        ]
        drug2_list = [
            Data(x=torch.randn(7, 10), edge_index=torch.randint(0, 7, (2, 12)))
            for _ in range(4)
        ]

        drug1_batch = Batch.from_data_list(drug1_list)
        drug2_batch = Batch.from_data_list(drug2_list)

        out = model(drug1_batch, drug2_batch)

        assert out.shape == (4, 86)

    def test_siamese_ddi(self):
        """Test Siamese DDI model."""
        from src.models.predictors import SiameseDDI
        from torch_geometric.data import Data, Batch

        model = SiameseDDI(
            num_atom_features=10,
            hidden_dim=64,
            num_classes=86,
            combination='concat',
        )

        drug1_list = [
            Data(x=torch.randn(5, 10), edge_index=torch.randint(0, 5, (2, 8)))
            for _ in range(4)
        ]
        drug2_list = [
            Data(x=torch.randn(7, 10), edge_index=torch.randint(0, 7, (2, 12)))
            for _ in range(4)
        ]

        drug1_batch = Batch.from_data_list(drug1_list)
        drug2_batch = Batch.from_data_list(drug2_list)

        out = model(drug1_batch, drug2_batch)

        assert out.shape == (4, 86)


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_compute_metrics(self):
        """Test metrics computation."""
        from src.evaluation.metrics import compute_metrics

        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 2, 1, 1, 2])
        y_prob = np.random.rand(9, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=3)

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1_macro <= 1
        assert metrics.confusion_matrix.shape == (3, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
