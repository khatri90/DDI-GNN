"""
Visualization Module for DDI-GNN

Provides visualization tools for model interpretability and results analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Tuple
import torch
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    top_k: int = 20,
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix [num_classes, num_classes]
        class_names: Class names for labels
        normalize: Whether to normalize
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        top_k: Show only top K classes

    Returns:
        Matplotlib figure
    """
    # Get top K classes by support
    if cm.shape[0] > top_k:
        class_support = cm.sum(axis=1)
        top_indices = np.argsort(class_support)[-top_k:]
        cm = cm[np.ix_(top_indices, top_indices)]
        if class_names:
            class_names = [class_names[i] for i in top_indices]

    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cm_normalized = cm

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_normalized,
        annot=True if cm.shape[0] <= 20 else False,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names if class_names else range(cm.shape[0]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
        ax=ax,
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    top_k: int = 10,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curves for top K classes.

    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        class_names: Class name mapping
        top_k: Number of classes to plot
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Get unique classes
    classes = np.unique(y_true)
    n_classes = len(classes)

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=classes)

    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    # Compute ROC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i, cls in enumerate(classes):
        fpr[cls], tpr[cls], _ = roc_curve(y_true_bin[:, i], y_prob[:, cls])
        roc_auc[cls] = auc(fpr[cls], tpr[cls])

    # Sort by AUC and get top K
    sorted_classes = sorted(roc_auc.keys(), key=lambda x: roc_auc[x], reverse=True)[:top_k]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, top_k))

    for cls, color in zip(sorted_classes, colors):
        class_name = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
        ax.plot(
            fpr[cls], tpr[cls],
            color=color,
            label=f'{class_name} (AUC = {roc_auc[cls]:.3f})'
        )

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (Top Classes)')
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_attention(
    attention_weights: torch.Tensor,
    batch_idx: int = 0,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize attention weights between drug atoms.

    Args:
        attention_weights: Attention matrix
        batch_idx: Batch index to visualize
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        attention_weights,
        cmap='Reds',
        ax=ax,
    )

    ax.set_xlabel('Drug 2 Atoms')
    ax.set_ylabel('Drug 1 Atoms')
    ax.set_title('Atom-Atom Attention Weights')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_molecule_attention(
    smiles: str,
    attention_weights: np.ndarray,
    title: str = 'Atom Attention',
    figsize: Tuple[int, int] = (400, 400),
    save_path: Optional[str] = None,
) -> Optional[object]:
    """
    Visualize attention weights on molecular structure.

    Args:
        smiles: SMILES string
        attention_weights: Attention weights per atom
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save figure

    Returns:
        PIL Image or None if RDKit unavailable
    """
    if not RDKIT_AVAILABLE:
        print("RDKit required for molecular visualization")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Normalize weights to [0, 1]
    weights = np.array(attention_weights)
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    # Create atom color map (blue to red)
    atom_colors = {}
    for i, w in enumerate(weights):
        if i < mol.GetNumAtoms():
            # Interpolate between blue (low) and red (high)
            atom_colors[i] = (w, 0.2, 1 - w)

    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)

    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(figsize[0], figsize[1])

    # Set drawing options
    opts = drawer.drawOptions()
    opts.addAtomIndices = True

    # Draw molecule
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(min(len(weights), mol.GetNumAtoms()))),
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()

    # Get PNG
    png_data = drawer.GetDrawingText()

    if save_path:
        with open(save_path, 'wb') as f:
            f.write(png_data)

    # Convert to PIL Image
    try:
        from PIL import Image
        import io
        return Image.open(io.BytesIO(png_data))
    except ImportError:
        return png_data


def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training curves from training history.

    Args:
        history: Training history dict
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Loss
    if 'train_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train', color='blue')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Validation', color='orange')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    if 'val_accuracy' in history:
        axes[0, 1].plot(history['val_accuracy'], color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].grid(True, alpha=0.3)

    # F1 Score
    if 'val_f1_macro' in history:
        axes[1, 0].plot(history['val_f1_macro'], label='Macro', color='red')
    if 'val_f1_weighted' in history:
        axes[1, 0].plot(history['val_f1_weighted'], label='Weighted', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Scores')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'], color='brown')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_class_distribution(
    labels: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    top_k: int = 20,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution of interaction classes.

    Args:
        labels: Array of class labels
        class_names: Class name mapping
        top_k: Show only top K classes
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    unique, counts = np.unique(labels, return_counts=True)

    # Sort by count
    sorted_indices = np.argsort(counts)[::-1][:top_k]
    unique = unique[sorted_indices]
    counts = counts[sorted_indices]

    # Get names
    if class_names:
        names = [class_names.get(int(c), f"Class {c}") for c in unique]
    else:
        names = [f"Class {c}" for c in unique]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(range(len(counts)), counts, color='steelblue')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_xlabel('Interaction Type')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Top {top_k} Interaction Types')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha='center',
            va='bottom',
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    top_k_classes: int = 10,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot t-SNE visualization of drug embeddings.

    Args:
        embeddings: Drug embeddings [N, dim]
        labels: Class labels [N]
        class_names: Class name mapping
        top_k_classes: Number of classes to show
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.manifold import TSNE

    # Get top K classes
    unique, counts = np.unique(labels, return_counts=True)
    top_classes = unique[np.argsort(counts)[-top_k_classes:]]

    # Filter to top classes
    mask = np.isin(labels, top_classes)
    embeddings_filtered = embeddings[mask]
    labels_filtered = labels[mask]

    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_filtered)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    for cls in top_classes:
        mask = labels_filtered == cls
        class_name = class_names.get(int(cls), f"Class {cls}") if class_names else f"Class {cls}"
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name,
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Drug Pair Embedding Visualization')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_prediction_report(
    drug1_smiles: str,
    drug2_smiles: str,
    prediction: int,
    confidence: float,
    top_predictions: List[Tuple[int, float]],
    attention_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    class_names: Optional[Dict[int, str]] = None,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Create a comprehensive prediction report.

    Args:
        drug1_smiles: First drug SMILES
        drug2_smiles: Second drug SMILES
        prediction: Predicted class
        confidence: Prediction confidence
        top_predictions: Top K predictions with probabilities
        attention_weights: Tuple of attention weights for each drug
        class_names: Class name mapping
        save_dir: Directory to save visualizations

    Returns:
        Report dictionary
    """
    report = {
        'drug1_smiles': drug1_smiles,
        'drug2_smiles': drug2_smiles,
        'prediction': {
            'class': int(prediction),
            'name': class_names.get(prediction, f"Class {prediction}") if class_names else str(prediction),
            'confidence': float(confidence),
        },
        'top_predictions': [
            {
                'class': int(cls),
                'name': class_names.get(cls, f"Class {cls}") if class_names else str(cls),
                'probability': float(prob),
            }
            for cls, prob in top_predictions
        ],
    }

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save molecule visualizations with attention
        if attention_weights is not None and RDKIT_AVAILABLE:
            attn1, attn2 = attention_weights

            img1 = visualize_molecule_attention(
                drug1_smiles, attn1,
                title='Drug 1 Attention',
                save_path=str(save_dir / 'drug1_attention.png')
            )
            img2 = visualize_molecule_attention(
                drug2_smiles, attn2,
                title='Drug 2 Attention',
                save_path=str(save_dir / 'drug2_attention.png')
            )

            report['visualizations'] = {
                'drug1_attention': str(save_dir / 'drug1_attention.png'),
                'drug2_attention': str(save_dir / 'drug2_attention.png'),
            }

    return report
