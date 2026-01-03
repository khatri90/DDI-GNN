# DDI-GNN: Drug-Drug Interaction Prediction using Graph Neural Networks

A comprehensive framework for predicting drug-drug interactions using state-of-the-art graph neural network architectures.

## Overview

This project implements a production-ready system for predicting drug-drug interactions (DDIs) using Graph Neural Networks (GNNs). DDIs are a major cause of adverse drug reactions, responsible for approximately 125,000 deaths annually in the US alone. This system can help healthcare professionals and researchers identify potential interactions between drug pairs.

### Key Features

- **Multiple GNN Architectures**: GCN, GAT, MPNN, and GIN encoders
- **Siamese Network Design**: Shared encoder for drug pairs
- **Knowledge Graph Integration**: Combine molecular features with biomedical knowledge
- **Interpretability**: Attention visualization and substructure analysis
- **Uncertainty Quantification**: Monte Carlo dropout for confidence estimation
- **Production Ready**: FastAPI server, Streamlit demo, Docker containerization

## Results

| Model | Accuracy | F1 Macro | F1 Weighted | AUC |
|-------|----------|----------|-------------|-----|
| Random Forest (Baseline) | ~85% | ~0.70 | ~0.82 | ~0.92 |
| GCN | ~89% | ~0.78 | ~0.87 | ~0.95 |
| GAT | ~91% | ~0.82 | ~0.89 | ~0.96 |
| GAT + Knowledge Graph | ~93% | ~0.85 | ~0.91 | ~0.97 |

*Results on DrugBank dataset with standard random split*

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.7+ (optional, for GPU acceleration)
- RDKit

### Quick Install

```bash
# Clone repository
git clone https://github.com/username/ddi-gnn.git
cd ddi-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Docker Installation

```bash
# Build image
docker build -t ddi-gnn .

# Run API server
docker run -p 8000:8000 ddi-gnn

# Run Streamlit demo
docker run -p 8501:8501 ddi-gnn streamlit run src/api/streamlit_app.py
```

## Quick Start

### Single Prediction

```python
from src.data.featurizers import smiles_to_graph
from src.models.full_model import load_model
import torch
from torch_geometric.data import Batch

# Load model
model = load_model("models/best_model.pt", device="cpu")
model.eval()

# Prepare drug pair
aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
ibuprofen = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"

graph1 = smiles_to_graph(aspirin)
graph2 = smiles_to_graph(ibuprofen)

batch1 = Batch.from_data_list([graph1])
batch2 = Batch.from_data_list([graph2])

# Predict
with torch.no_grad():
    logits = model(batch1, batch2)
    probs = torch.softmax(logits[0], dim=1)[0]
    prediction = probs.argmax().item()
    confidence = probs[prediction].item()

print(f"Predicted interaction: {prediction}, Confidence: {confidence:.2%}")
```

### Command Line Prediction

```bash
python scripts/predict.py \
    --drug1 "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --drug2 "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \
    --checkpoint models/best_model.pt
```

### Training

```bash
# Train with default configuration
python scripts/train.py --config configs/gcn_baseline.yaml

# Train with custom settings
python scripts/train.py \
    --config configs/gat_kg.yaml \
    --epochs 100 \
    --batch-size 32 \
    --wandb
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/experiment/best_model.pt \
    --data-split test \
    --save-predictions
```

### API Server

```bash
# Start FastAPI server
uvicorn src.api.serve:app --host 0.0.0.0 --port 8000

# Or using Docker
docker-compose up api
```

### Streamlit Demo

```bash
# Start Streamlit app
streamlit run src/api/streamlit_app.py

# Or using Docker
docker-compose up demo
```

## Project Structure

```
ddi-gnn/
├── configs/                  # Configuration files
│   ├── gcn_baseline.yaml
│   ├── gat_kg.yaml
│   └── best_model.yaml
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Preprocessed data
│   └── scripts/
├── src/
│   ├── data/                 # Data processing
│   │   ├── featurizers.py    # SMILES to graph conversion
│   │   ├── dataset.py        # PyTorch datasets
│   │   └── knowledge_graph.py
│   ├── models/               # Model architectures
│   │   ├── encoders.py       # GNN encoders
│   │   ├── predictors.py     # DDI predictors
│   │   └── full_model.py
│   ├── training/             # Training utilities
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── evaluation/           # Evaluation utilities
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── api/                  # API and demo
│       ├── serve.py          # FastAPI server
│       └── streamlit_app.py  # Streamlit demo
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── notebooks/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
└── README.md
```

## Configuration

Example configuration file (`configs/best_model.yaml`):

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
  scheduler:
    type: "cosine_annealing"

data:
  dataset: "drugbank"
  split_type: "random"
```

## API Reference

### POST /predict

Predict drug-drug interaction.

**Request:**
```json
{
  "drug1": {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
  "drug2": {"smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"}
}
```

**Response:**
```json
{
  "predicted_interaction": "The metabolism of Drug1 can be decreased when combined with Drug2",
  "predicted_interaction_id": 42,
  "confidence": 0.87,
  "top_predictions": [...]
}
```

## Methodology

### Molecular Representation

Drugs are represented as molecular graphs where:
- **Nodes**: Atoms with features (atomic number, degree, charge, hybridization, etc.)
- **Edges**: Bonds with features (bond type, conjugation, ring membership)

### Model Architecture

1. **Encoder**: Graph Neural Network (GCN/GAT/MPNN/GIN) encodes each drug molecule
2. **Siamese Design**: Both drugs use shared encoder weights
3. **Knowledge Graph**: Optional integration of biomedical knowledge embeddings
4. **Classifier**: MLP predicts interaction type from concatenated drug embeddings

### Training Strategy

- Cross-entropy loss with label smoothing
- AdamW optimizer with cosine annealing scheduler
- Early stopping based on validation F1 score
- Gradient clipping for stability

## Datasets

### DrugBank (Default)
- 86 interaction types
- ~500K drug pairs
- Accessed via TDC (Therapeutics Data Commons)

### TWOSIDES
- Multi-label side effect prediction
- ~600 side effect types

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ddi_gnn_2026,
  title={DDI-GNN: Drug-Drug Interaction Prediction using Graph Neural Networks},
  author={Bazil},
  year={2026},
  url={https://github.com/username/ddi-gnn}
}
```

## References

- [Graph Convolutional Networks (Kipf & Welling, 2017)](https://arxiv.org/abs/1609.02907)
- [Graph Attention Networks (Velickovic et al., 2018)](https://arxiv.org/abs/1710.10903)
- [KGNN: Knowledge Graph Neural Network for DDI Prediction](https://arxiv.org/abs/1912.00935)
- [TDC: Therapeutics Data Commons](https://tdcommons.ai/)

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Therapeutics Data Commons for dataset access
- PyTorch Geometric team for GNN implementations
- RDKit community for molecular processing tools
