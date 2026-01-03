"""
DDI-GNN: Drug-Drug Interaction Prediction using Graph Neural Networks

A comprehensive framework for predicting drug-drug interactions using
state-of-the-art graph neural network architectures.
"""

__version__ = "1.0.0"
__author__ = "Bazil"

from .data import DDIDataset, smiles_to_graph
from .models import DDIPredictor, GATDrugEncoder, GCNDrugEncoder, MPNNEncoder
