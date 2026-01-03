"""
FastAPI Server for DDI Prediction

Provides REST API endpoints for drug-drug interaction prediction.
"""

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.featurizers import smiles_to_graph, MoleculeFeaturizer
from src.models.full_model import load_model, DDIModel


# ============================================================================
# Pydantic Models
# ============================================================================

class DrugInfo(BaseModel):
    """Drug information."""
    smiles: str = Field(..., description="SMILES representation of the drug")
    name: Optional[str] = Field(None, description="Drug name (optional)")


class DDIPredictionRequest(BaseModel):
    """Request for DDI prediction."""
    drug1: DrugInfo = Field(..., description="First drug")
    drug2: DrugInfo = Field(..., description="Second drug")
    return_attention: bool = Field(False, description="Return attention weights")
    return_uncertainty: bool = Field(False, description="Return uncertainty estimate")


class PredictionResult(BaseModel):
    """Single prediction result."""
    interaction_type: str
    interaction_id: int
    probability: float


class DDIPredictionResponse(BaseModel):
    """Response for DDI prediction."""
    drug1_smiles: str
    drug2_smiles: str
    predicted_interaction: str
    predicted_interaction_id: int
    confidence: float
    top_predictions: List[PredictionResult]
    uncertainty: Optional[float] = None
    drug1_attention: Optional[List[float]] = None
    drug2_attention: Optional[List[float]] = None


class BatchPredictionRequest(BaseModel):
    """Request for batch prediction."""
    pairs: List[DDIPredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[DDIPredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_type: str
    num_classes: int
    device: str
    config: Dict


# ============================================================================
# Global State
# ============================================================================

class ModelState:
    """Global model state."""
    model: Optional[DDIModel] = None
    featurizer: Optional[MoleculeFeaturizer] = None
    device: str = "cpu"
    config: Dict = {}
    label_names: Dict[int, str] = {}


state = ModelState()


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="DDI-GNN Prediction API",
    description="Drug-Drug Interaction Prediction using Graph Neural Networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Startup Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = os.environ.get("MODEL_PATH", "models/best_model.pt")
    config_path = os.environ.get("CONFIG_PATH", "configs/best_model.yaml")

    state.device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if os.path.exists(model_path):
            state.model = load_model(
                model_path,
                config_path if os.path.exists(config_path) else None,
                device=state.device,
            )
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}")
            # Create a default model for demo
            state.model = DDIModel(
                model_type='siamese',
                num_atom_features=157,
                hidden_dim=128,
                num_classes=86,
                encoder_type='gat',
            ).to(state.device)
            print("Using default model (not trained)")

        state.featurizer = MoleculeFeaturizer()

        # Load label names
        label_names_path = os.environ.get("LABEL_NAMES_PATH", "data/processed/label_names.json")
        if os.path.exists(label_names_path):
            import json
            with open(label_names_path, 'r') as f:
                state.label_names = {int(k): v for k, v in json.load(f).items()}
        else:
            state.label_names = {i: f"Interaction Type {i}" for i in range(86)}

    except Exception as e:
        print(f"Error loading model: {e}")
        state.model = None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.model is not None,
        device=state.device,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name="DDI-GNN",
        model_type=state.model.model_type,
        num_classes=86,
        device=state.device,
        config=state.config,
    )


@app.post("/predict", response_model=DDIPredictionResponse)
async def predict_ddi(request: DDIPredictionRequest):
    """
    Predict drug-drug interaction.

    Args:
        request: Prediction request with two drugs

    Returns:
        Prediction response with interaction type and confidence
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert SMILES to graphs
        drug1_graph = smiles_to_graph(request.drug1.smiles)
        drug2_graph = smiles_to_graph(request.drug2.smiles)

        if drug1_graph is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid SMILES for drug 1: {request.drug1.smiles}"
            )
        if drug2_graph is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid SMILES for drug 2: {request.drug2.smiles}"
            )

        # Add batch dimension
        from torch_geometric.data import Batch
        drug1_batch = Batch.from_data_list([drug1_graph]).to(state.device)
        drug2_batch = Batch.from_data_list([drug2_graph]).to(state.device)

        # Predict
        state.model.eval()
        with torch.no_grad():
            if request.return_attention:
                outputs = state.model(
                    drug1_batch, drug2_batch,
                    return_attention=True
                )
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, attention = outputs
                else:
                    logits = outputs
                    attention = None
            else:
                outputs = state.model(drug1_batch, drug2_batch)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                attention = None

            probs = torch.softmax(logits, dim=1)[0]
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()

            # Get top 5 predictions
            top_k = 5
            top_probs, top_indices = probs.topk(top_k)

            top_predictions = [
                PredictionResult(
                    interaction_type=state.label_names.get(idx.item(), f"Type {idx.item()}"),
                    interaction_id=idx.item(),
                    probability=prob.item(),
                )
                for prob, idx in zip(top_probs, top_indices)
            ]

        response = DDIPredictionResponse(
            drug1_smiles=request.drug1.smiles,
            drug2_smiles=request.drug2.smiles,
            predicted_interaction=state.label_names.get(pred_class, f"Type {pred_class}"),
            predicted_interaction_id=pred_class,
            confidence=confidence,
            top_predictions=top_predictions,
        )

        # Add attention if available
        if attention is not None and isinstance(attention, dict):
            response.drug1_attention = attention.get('drug1_attention', [])
            response.drug2_attention = attention.get('drug2_attention', [])

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict drug-drug interactions for multiple pairs.

    Args:
        request: Batch prediction request

    Returns:
        Batch prediction response
    """
    predictions = []

    for pair in request.pairs:
        try:
            pred = await predict_ddi(pair)
            predictions.append(pred)
        except HTTPException as e:
            # Create error response for failed prediction
            predictions.append(DDIPredictionResponse(
                drug1_smiles=pair.drug1.smiles,
                drug2_smiles=pair.drug2.smiles,
                predicted_interaction="Error",
                predicted_interaction_id=-1,
                confidence=0.0,
                top_predictions=[],
            ))

    return BatchPredictionResponse(predictions=predictions)


@app.get("/interaction-types")
async def get_interaction_types():
    """Get list of all interaction types."""
    return {
        "interaction_types": [
            {"id": k, "name": v}
            for k, v in sorted(state.label_names.items())
        ]
    }


@app.post("/validate-smiles")
async def validate_smiles(smiles: str):
    """Validate a SMILES string."""
    graph = smiles_to_graph(smiles)
    if graph is None:
        return {
            "valid": False,
            "smiles": smiles,
            "error": "Could not parse SMILES",
        }

    return {
        "valid": True,
        "smiles": smiles,
        "num_atoms": graph.num_nodes,
        "num_bonds": graph.num_edges // 2,  # Edges are bidirectional
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the API server."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))

    print(f"Starting DDI-GNN API server on {host}:{port}")

    uvicorn.run(
        "src.api.serve:app",
        host=host,
        port=port,
        reload=os.environ.get("DEBUG", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
