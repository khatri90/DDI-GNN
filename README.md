# DDI-GNN: Drug-Drug Interaction Prediction Framework

A comprehensive, production-ready framework for predicting drug-drug interactions (DDIs) using state-of-the-art Graph Neural Networks (GNNs).

## Interface Preview

![Application Interface](screenshots/Screenshot%202026-01-06%20025009.png)

*The DDI-GNN Streamlit interface demonstrating interaction prediction with structural visualizations.*

## Overview

This project implements a robust system for predicting drug-drug interactions, a critical task in pharmacology and patient safety. Adverse drug reactions are a significant healthcare burden, responsible for approximately 125,000 deaths annually in the US alone. Our system leverages advanced graph representation learning to identify potential interactions between drug pairs with high accuracy.

### Key Capabilities

*   **Advanced GNN Architectures**: Supports GCN, GAT, MPNN, and GIN encoders.
*   **Siamese Network Design**: Utilizes a shared encoder architecture optimized for pair-wise inputs.
*   **Knowledge Graph Integration**: Enhances molecular features with biomedical knowledge embeddings.
*   **Explainable AI**: Provides attention visualization and substructure analysis for model interpretability.
*   **Uncertainty Quantification**: Implements Monte Carlo dropout for reliable confidence estimation.
*   **Deployment Ready**: Includes a FastAPI server, Streamlit dashboard, and Docker containerization.

## Performance Metrics

| Model | Accuracy | F1 Macro | F1 Weighted | AUC |
|-------|----------|----------|-------------|-----|
| Random Forest (Baseline) | ~85% | ~0.70 | ~0.82 | ~0.92 |
| GCN | ~89% | ~0.78 | ~0.87 | ~0.95 |
| GAT | ~91% | ~0.82 | ~0.89 | ~0.96 |
| **GAT + Knowledge Graph** | **87.8%** | **0.76** | **0.88** | **-** |

*Results based on the DrugBank dataset using standard random split validation.*

## Installation

### Prerequisites

*   Python 3.9+
*   CUDA 11.7+ (Recommended for GPU acceleration)
*   RDKit

### Quick Install

```bash
# Clone the repository
git clone https://github.com/username/ddi-gnn.git
cd ddi-gnn

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t ddi-gnn .

# Launch the API server
docker run -p 8000:8000 ddi-gnn

# Launch the Streamlit dashboard
docker run -p 8501:8501 ddi-gnn streamlit run src/api/streamlit_app.py
```

## Usage Guide

### Python API

```python
import torch
from torch_geometric.data import Batch
from src.data.featurizers import smiles_to_graph
from src.models.full_model import load_model

# Load the production model
model = load_model("models/production_model.pt", device="cpu")
model.eval()

# Define drug pair (SMILES strings)
aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
ibuprofen = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"

# Convert to graph representations
graph1 = smiles_to_graph(aspirin)
graph2 = smiles_to_graph(ibuprofen)

# Prepare batches
batch1 = Batch.from_data_list([graph1])
batch2 = Batch.from_data_list([graph2])

# Generate prediction
with torch.no_grad():
    logits = model(batch1, batch2)
    probs = torch.softmax(logits[0], dim=1)[0]
    prediction_id = probs.argmax().item()
    confidence = probs[prediction_id].item()

print(f"Predicted Interaction ID: {prediction_id}")
print(f"Confidence: {confidence:.2%}")
```

### Command Line Interface

```bash
python scripts/predict.py \
    --drug1 "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --drug2 "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \
    --checkpoint models/production_model.pt
```

### Streamlit Dashboard

Launch the interactive web interface:

```bash
streamlit run src/api/streamlit_app.py
```

## Project Logic

```
ddi-gnn/
├── configs/                  # Model configurations
├── data/
│   ├── raw/                  # Source datasets
│   ├── processed/            # Feature-engineered data
│   └── label_names.json      # Interaction type mappings
├── models/
│   ├── production_model.pt   # Active deployed model
│   └── production_config.yaml
├── src/
│   ├── data/                 # Data loading and featurization
│   ├── models/               # Network architectures
│   ├── training/             # Training loops and loss functions
│   └── api/                  # Serving infrastructure
├── scripts/                  # CLI utilities
└── screenshots/              # Application images
```

## Configuration

Configuration is managed via YAML files. Example configuration:

```yaml
model:
  name: "GAT_KG"
  encoder:
    type: "gat"
    num_layers: 4
    hidden_dim: 256
    num_heads: 8
    dropout: 0.3
  kg_encoder:
    enabled: true
    kg_embedding_dim: 256

training:
  epochs: 200
  batch_size: 32
  learning_rate: 0.0003
```

## Citation

Please cite this work if you utilize this codebase:

```bibtex
@software{ddi_gnn_2026,
  title={DDI-GNN: Drug-Drug Interaction Prediction using Graph Neural Networks},
  author={Prokopios Bazil},
  year={2026},
  url={https://github.com/username/ddi-gnn}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

*   Therapeutics Data Commons
*   PyTorch Geometric
*   RDKit
