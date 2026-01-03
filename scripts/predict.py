#!/usr/bin/env python
"""
Prediction Script for DDI-GNN

Usage:
    python scripts/predict.py --drug1 "CC(=O)OC1=CC=CC=C1C(=O)O" --drug2 "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    python scripts/predict.py --input pairs.csv --output predictions.csv
"""

import os
import sys
import argparse
import json
import torch
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.featurizers import smiles_to_graph
from src.models.full_model import load_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict drug-drug interactions")

    # Single prediction
    parser.add_argument(
        "--drug1",
        type=str,
        default=None,
        help="SMILES for first drug",
    )
    parser.add_argument(
        "--drug2",
        type=str,
        default=None,
        help="SMILES for second drug",
    )

    # Batch prediction
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV file with drug pairs (columns: drug1_smiles, drug2_smiles)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output file for predictions",
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use",
    )

    # Output options
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to return",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Include uncertainty estimation",
    )

    return parser.parse_args()


def load_label_names(path: str = "data/processed/label_names.json") -> dict:
    """Load label names."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {i: f"Interaction Type {i}" for i in range(86)}


def predict_single(
    model,
    smiles1: str,
    smiles2: str,
    device: str,
    label_names: dict,
    top_k: int = 5,
) -> dict:
    """Predict interaction for a single drug pair."""
    # Convert SMILES to graphs
    graph1 = smiles_to_graph(smiles1)
    graph2 = smiles_to_graph(smiles2)

    if graph1 is None:
        return {'error': f"Invalid SMILES for drug 1: {smiles1}"}
    if graph2 is None:
        return {'error': f"Invalid SMILES for drug 2: {smiles2}"}

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

        # Get top K predictions
        top_probs, top_indices = probs.topk(top_k)

        return {
            'drug1_smiles': smiles1,
            'drug2_smiles': smiles2,
            'predicted_class': pred_class,
            'predicted_interaction': label_names.get(pred_class, f"Type {pred_class}"),
            'confidence': float(confidence),
            'top_predictions': [
                {
                    'class': int(idx),
                    'interaction': label_names.get(int(idx), f"Type {idx}"),
                    'probability': float(prob),
                }
                for prob, idx in zip(top_probs, top_indices)
            ],
        }


def predict_batch(
    model,
    df: pd.DataFrame,
    device: str,
    label_names: dict,
    smiles1_col: str = 'drug1_smiles',
    smiles2_col: str = 'drug2_smiles',
) -> pd.DataFrame:
    """Predict interactions for multiple drug pairs."""
    from tqdm import tqdm
    from torch_geometric.data import Batch

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        smiles1 = row[smiles1_col]
        smiles2 = row[smiles2_col]

        # Convert to graphs
        graph1 = smiles_to_graph(smiles1)
        graph2 = smiles_to_graph(smiles2)

        if graph1 is None or graph2 is None:
            results.append({
                'drug1_smiles': smiles1,
                'drug2_smiles': smiles2,
                'predicted_class': -1,
                'predicted_interaction': 'Error',
                'confidence': 0.0,
                'valid': False,
            })
            continue

        # Create batch
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

        results.append({
            'drug1_smiles': smiles1,
            'drug2_smiles': smiles2,
            'predicted_class': pred_class,
            'predicted_interaction': label_names.get(pred_class, f"Type {pred_class}"),
            'confidence': confidence,
            'valid': True,
        })

    return pd.DataFrame(results)


def main():
    """Main prediction function."""
    args = parse_args()

    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load label names
    label_names = load_label_names()

    # Check if model exists
    if not os.path.exists(args.checkpoint):
        print(f"Warning: Model not found at {args.checkpoint}")
        print("Using default model (not trained)")

        from src.models.full_model import DDIModel
        model = DDIModel(
            model_type='siamese',
            num_atom_features=157,
            hidden_dim=128,
            num_classes=86,
            encoder_type='gat',
        ).to(device)
    else:
        print(f"Loading model from {args.checkpoint}")
        model = load_model(args.checkpoint, args.config, device=device)

    # Single prediction
    if args.drug1 and args.drug2:
        print(f"\nPredicting interaction between:")
        print(f"  Drug 1: {args.drug1}")
        print(f"  Drug 2: {args.drug2}")

        result = predict_single(
            model, args.drug1, args.drug2,
            device, label_names, args.top_k
        )

        if 'error' in result:
            print(f"\nError: {result['error']}")
            return

        print(f"\nPrediction Results:")
        print(f"  Interaction Type: {result['predicted_interaction']}")
        print(f"  Confidence: {result['confidence']:.2%}")

        print(f"\n  Top {args.top_k} Predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"    {i}. {pred['interaction']} ({pred['probability']:.2%})")

        if args.json:
            print(f"\nJSON Output:")
            print(json.dumps(result, indent=2))

    # Batch prediction
    elif args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        print(f"\nLoading drug pairs from {args.input}")
        df = pd.read_csv(args.input)

        # Detect column names
        smiles1_col = None
        smiles2_col = None

        for col in ['drug1_smiles', 'Drug1', 'smiles1', 'SMILES1']:
            if col in df.columns:
                smiles1_col = col
                break

        for col in ['drug2_smiles', 'Drug2', 'smiles2', 'SMILES2']:
            if col in df.columns:
                smiles2_col = col
                break

        if smiles1_col is None or smiles2_col is None:
            print("Error: Could not find SMILES columns in input file")
            print(f"  Available columns: {list(df.columns)}")
            return

        print(f"  Using columns: {smiles1_col}, {smiles2_col}")
        print(f"  Total pairs: {len(df)}")

        # Predict
        results_df = predict_batch(
            model, df, device, label_names,
            smiles1_col, smiles2_col
        )

        # Save results
        results_df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")

        # Summary
        valid_preds = results_df[results_df['valid']]
        print(f"\nSummary:")
        print(f"  Valid predictions: {len(valid_preds)} / {len(results_df)}")
        print(f"  Average confidence: {valid_preds['confidence'].mean():.2%}")

        # Top interactions
        top_interactions = valid_preds['predicted_interaction'].value_counts().head(5)
        print(f"\n  Most predicted interactions:")
        for interaction, count in top_interactions.items():
            print(f"    {interaction}: {count}")

    else:
        print("Error: Please provide either --drug1 and --drug2 for single prediction,")
        print("       or --input for batch prediction.")
        print("\nExamples:")
        print("  Single: python scripts/predict.py --drug1 'CC(=O)OC1=CC=CC=C1C(=O)O' --drug2 'CCO'")
        print("  Batch:  python scripts/predict.py --input pairs.csv --output predictions.csv")


if __name__ == "__main__":
    main()
