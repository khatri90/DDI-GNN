"""
Streamlit Demo Application for DDI-GNN

Interactive web interface for predicting drug-drug interactions.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import io

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.data.featurizers import smiles_to_graph, MoleculeFeaturizer
from src.models.full_model import load_model, DDIModel


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="DDI-GNN: Drug-Drug Interaction Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #000000; /* Force black text for visibility on light background */
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .molecule-image {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Model Loading (Cached)
# ============================================================================

def find_latest_model():
    """Find the most recent trained model in outputs directory."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return None, None

    # Find all experiment directories
    exp_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if not exp_dirs:
        return None, None

    # Sort by modification time (most recent first)
    exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Look for best_model.pt in each directory
    for exp_dir in exp_dirs:
        model_path = exp_dir / "best_model.pt"
        if model_path.exists():
            # Try to find corresponding config
            exp_name = exp_dir.name
            # Extract config name from experiment name (e.g., GAT_KG -> gat_kg.yaml)
            config_name = "_".join(exp_name.split("_")[:-2]).lower() + ".yaml"
            config_path = Path("configs") / config_name
            if not config_path.exists():
                config_path = None
            return str(model_path), str(config_path) if config_path else None

    return None, None


def get_model_cache_key():
    """Generate cache key based on latest model file."""
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        exp_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
        if exp_dirs:
            exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for exp_dir in exp_dirs:
                model_path = exp_dir / "best_model.pt"
                if model_path.exists():
                    return str(model_path.stat().st_mtime)
    return "default"


@st.cache_resource
def load_ddi_model(_cache_key=None):
    """Load the DDI prediction model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # First check environment variables
    model_path = os.environ.get("MODEL_PATH")
    config_path = os.environ.get("CONFIG_PATH")

    # If not set, try to find the production model first
    if not model_path:
        prod_model_path = Path("models/production_model.pt")
        prod_config_path = Path("models/production_config.yaml")
        
        if prod_model_path.exists():
            model_path = str(prod_model_path)
            if not config_path and prod_config_path.exists():
                config_path = str(prod_config_path)
        
        # Fallback to finding latest model in outputs
        if not model_path:
            model_path, auto_config_path = find_latest_model()
            if not config_path:
                config_path = auto_config_path

    try:
        if model_path and os.path.exists(model_path):
            # Load checkpoint to get num_atom_features from saved state
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            # Get state dict
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            # Infer model architecture from checkpoint keys
            num_atom_features = 169  # Default
            hidden_dim = 128
            num_layers = 3
            num_heads = 4

            # Look for encoder input_proj layer weight shape [hidden_dim, num_features]
            for key, value in state_dict.items():
                if 'input_proj.weight' in key and 'output' not in key:
                    num_atom_features = value.shape[1]  # Shape is [hidden_dim, num_features]
                    hidden_dim = value.shape[0]
                    st.sidebar.info(f"Detected: {num_atom_features} atom features, {hidden_dim} hidden dim")
                    break

            # Count number of conv layers to get num_layers
            conv_keys = [k for k in state_dict.keys() if 'convs.' in k and '.weight' in k]
            if conv_keys:
                layer_nums = set()
                for k in conv_keys:
                    parts = k.split('.')
                    for i, p in enumerate(parts):
                        if p == 'convs' and i + 1 < len(parts):
                            try:
                                layer_nums.add(int(parts[i + 1]))
                            except ValueError:
                                pass
                if layer_nums:
                    num_layers = max(layer_nums) + 1

            # Create model with correct dimensions
            model = DDIModel(
                model_type='siamese',
                num_atom_features=num_atom_features,
                hidden_dim=hidden_dim,
                num_classes=86,
                encoder_type='gat',
                num_layers=num_layers,
                dropout=0.2,
                num_heads=num_heads,
            ).to(device)

            # Load weights
            model.load_state_dict(state_dict)

            model.eval()
            st.sidebar.success(f"Model loaded from {model_path}")
            return model, device
        else:
            # Use default model for demo
            model = DDIModel(
                model_type='siamese',
                num_atom_features=169,
                hidden_dim=128,
                num_classes=86,
                encoder_type='gat',
            ).to(device)
            st.sidebar.warning("Using default model (not trained)")

        return model, device
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None, device


@st.cache_data
def load_label_names():
    """Load interaction type names."""
    # Try resolving relative to this file first
    project_root = Path(__file__).parent.parent.parent
    label_path = project_root / "data/processed/label_names.json"
    
    if label_path.exists():
        import json
        with open(label_path, 'r', encoding='utf-8') as f: # Specify encoding for safety
            return {int(k): v for k, v in json.load(f).items()}
    
    # Fallback to current directory
    elif os.path.exists("data/processed/label_names.json"):
        import json
        with open("data/processed/label_names.json", 'r', encoding='utf-8') as f:
            return {int(k): v for k, v in json.load(f).items()}
            
    return {i: f"Interaction Type {i}" for i in range(86)}


@st.cache_data
def get_example_drugs():
    """Get example drug SMILES."""
    return {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Acetaminophen": "CC(=O)NC1=CC=C(O)C=C1",
        "Warfarin": "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O",
        "Metformin": "CN(C)C(=N)NC(=N)N",
        "Simvastatin": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
        "Omeprazole": "COC1=CC2=NC(CS(=O)C3=NC4=CC=C(OC)C=C4N3)=NC2=CC1C",
        "Lisinopril": "NCCCC[C@H](N[C@@H](CCC1=CC=CC=C1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O",
    }


# ============================================================================
# Helper Functions
# ============================================================================

def smiles_to_image(smiles: str, size: Tuple[int, int] = (300, 300)) -> Optional[bytes]:
    """Convert SMILES to image bytes."""
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=size)

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"


def predict_interaction(
    model: DDIModel,
    smiles1: str,
    smiles2: str,
    device: str,
    label_names: Dict[int, str],
) -> Optional[Dict]:
    """Predict drug-drug interaction."""
    try:
        # Convert SMILES to graphs
        graph1 = smiles_to_graph(smiles1)
        graph2 = smiles_to_graph(smiles2)

        if graph1 is None or graph2 is None:
            return None

        # Create batch
        from torch_geometric.data import Batch
        batch1 = Batch.from_data_list([graph1]).to(device)
        batch2 = Batch.from_data_list([graph2]).to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(batch1, batch2)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            probs = torch.softmax(logits, dim=1)[0]
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()

            # Get top 5 predictions
            top_probs, top_indices = probs.topk(5)

            return {
                'prediction': pred_class,
                'prediction_name': label_names.get(pred_class, f"Type {pred_class}"),
                'confidence': confidence,
                'top_predictions': [
                    {
                        'class': idx.item(),
                        'name': label_names.get(idx.item(), f"Type {idx.item()}"),
                        'probability': prob.item(),
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ],
            }

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# ============================================================================
# Main App
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">Drug-Drug Interaction Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Graph Neural Networks</div>', unsafe_allow_html=True)

    # Load model (with cache key to detect model changes)
    model, device = load_ddi_model(_cache_key=get_model_cache_key())
    label_names = load_label_names()
    example_drugs = get_example_drugs()

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("""
    This application predicts potential drug-drug interactions using
    state-of-the-art Graph Neural Networks.

    **How it works:**
    1. Enter SMILES for two drugs
    2. The model encodes molecular structures as graphs
    3. A GNN processes these graphs to predict interactions

    **Model:** GAT-based Siamese Network
    """)

    st.sidebar.header("Device Info")
    st.sidebar.write(f"**Device:** {device}")
    st.sidebar.write(f"**RDKit Available:** {RDKIT_AVAILABLE}")

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drug 1")

        # Example selection
        drug1_example = st.selectbox(
            "Select example drug:",
            ["Custom"] + list(example_drugs.keys()),
            key="drug1_example"
        )

        if drug1_example != "Custom":
            drug1_smiles = example_drugs[drug1_example]
        else:
            drug1_smiles = st.text_input(
                "Enter SMILES:",
                value="CC(=O)OC1=CC=CC=C1C(=O)O",
                key="drug1_smiles"
            )

        st.text_input("SMILES:", value=drug1_smiles, disabled=True, key="drug1_display")

        # Show molecule
        if RDKIT_AVAILABLE and drug1_smiles:
            img_bytes = smiles_to_image(drug1_smiles)
            if img_bytes:
                st.image(img_bytes, caption="Drug 1 Structure", width="stretch")
            else:
                st.warning("Invalid SMILES for Drug 1")

    with col2:
        st.subheader("Drug 2")

        # Example selection
        drug2_example = st.selectbox(
            "Select example drug:",
            ["Custom"] + list(example_drugs.keys()),
            key="drug2_example"
        )

        if drug2_example != "Custom":
            drug2_smiles = example_drugs[drug2_example]
        else:
            drug2_smiles = st.text_input(
                "Enter SMILES:",
                value="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                key="drug2_smiles"
            )

        st.text_input("SMILES:", value=drug2_smiles, disabled=True, key="drug2_display")

        # Show molecule
        if RDKIT_AVAILABLE and drug2_smiles:
            img_bytes = smiles_to_image(drug2_smiles)
            if img_bytes:
                st.image(img_bytes, caption="Drug 2 Structure", width="stretch")
            else:
                st.warning("Invalid SMILES for Drug 2")

    # Prediction button
    st.markdown("---")

    if st.button("Predict Interaction", type="primary", width="stretch"):
        if model is None:
            st.error("Model not loaded. Please check the model path.")
        elif not drug1_smiles or not drug2_smiles:
            st.warning("Please enter SMILES for both drugs.")
        else:
            with st.spinner("Analyzing molecular structures..."):
                result = predict_interaction(
                    model, drug1_smiles, drug2_smiles,
                    device, label_names
                )

            if result:
                st.success("Prediction complete!")

                # Results
                st.markdown("### Prediction Results")

                # Main prediction
                confidence_class = get_confidence_class(result['confidence'])
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Predicted Interaction Type</h3>
                    <h2>{result['prediction_name']}</h2>
                    <p>Confidence: <span class="{confidence_class}">{result['confidence']:.1%}</span></p>
                </div>
                """, unsafe_allow_html=True)

                # Top predictions
                st.markdown("### Top 5 Predictions")

                pred_df = pd.DataFrame(result['top_predictions'])
                pred_df.columns = ['Class ID', 'Interaction Type', 'Probability']
                pred_df['Probability'] = pred_df['Probability'].apply(lambda x: f"{x:.2%}")

                st.dataframe(pred_df, width="stretch")

                # Visualization
                st.markdown("### Probability Distribution")

                chart_data = pd.DataFrame({
                    'Class ID': [f"Class {p['class']}" for p in result['top_predictions']],
                    'Probability': [p['probability'] for p in result['top_predictions']],
                })
                st.bar_chart(chart_data.set_index('Class ID'))

            else:
                st.error("Could not make prediction. Please check the SMILES inputs.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        DDI-GNN: Drug-Drug Interaction Prediction using Graph Neural Networks<br>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
