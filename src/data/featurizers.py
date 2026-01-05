"""
Molecular Featurization Module

Converts SMILES strings to PyTorch Geometric graph representations
with comprehensive atom and bond features.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from torch_geometric.data import Data

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Some features will be disabled.")


# Atom feature specifications
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # Periodic table elements
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'num_hs': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ] if RDKIT_AVAILABLE else [],
    'is_aromatic': [False, True],
    'chiral_tag': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ] if RDKIT_AVAILABLE else [],
}

# Bond feature specifications
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ] if RDKIT_AVAILABLE else [],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ] if RDKIT_AVAILABLE else [],
    'is_conjugated': [False, True],
    'is_in_ring': [False, True],
}


def one_hot_encoding(value, allowable_set: List, include_unknown: bool = True) -> List[int]:
    """
    One-hot encode a value given an allowable set.

    Args:
        value: Value to encode
        allowable_set: List of allowable values
        include_unknown: Whether to include an unknown category

    Returns:
        One-hot encoded list
    """
    if include_unknown:
        allowable_set = list(allowable_set) + ['UNK']

    encoding = [0] * len(allowable_set)

    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    elif include_unknown:
        encoding[-1] = 1

    return encoding


def get_atom_features(atom, use_chirality: bool = True) -> List[float]:
    """
    Extract comprehensive features from an RDKit atom.

    Args:
        atom: RDKit atom object
        use_chirality: Whether to include chirality features

    Returns:
        List of atom features
    """
    features = []

    # Atomic number (one-hot)
    features.extend(one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))

    # Degree (one-hot)
    features.extend(one_hot_encoding(atom.GetDegree(), ATOM_FEATURES['degree']))

    # Formal charge (one-hot)
    features.extend(one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))

    # Number of hydrogens (one-hot)
    features.extend(one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']))

    # Hybridization (one-hot)
    features.extend(one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization']))

    # Is aromatic (binary)
    features.append(float(atom.GetIsAromatic()))

    # Additional features
    features.append(float(atom.GetMass()) / 100.0)  # Normalized mass
    features.append(float(atom.GetNumRadicalElectrons()))
    features.append(float(atom.IsInRing()))

    if use_chirality:
        features.extend(one_hot_encoding(atom.GetChiralTag(), ATOM_FEATURES['chiral_tag']))

    return features


def get_bond_features(bond, use_stereo: bool = True) -> List[float]:
    """
    Extract comprehensive features from an RDKit bond.

    Args:
        bond: RDKit bond object
        use_stereo: Whether to include stereochemistry features

    Returns:
        List of bond features
    """
    features = []

    # Bond type (one-hot)
    features.extend(one_hot_encoding(bond.GetBondType(), BOND_FEATURES['bond_type']))

    # Is conjugated (binary)
    features.append(float(bond.GetIsConjugated()))

    # Is in ring (binary)
    features.append(float(bond.IsInRing()))

    if use_stereo:
        features.extend(one_hot_encoding(bond.GetStereo(), BOND_FEATURES['stereo']))

    return features


def smiles_to_graph(
    smiles: str,
    use_chirality: bool = True,
    use_stereo: bool = True,
    add_self_loops: bool = False,
) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles: SMILES string representation of molecule
        use_chirality: Include chirality in atom features
        use_stereo: Include stereochemistry in bond features
        add_self_loops: Add self-loop edges

    Returns:
        PyTorch Geometric Data object or None if conversion fails
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES to graph conversion")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add hydrogens for more complete representation
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates if needed (optional, for geometric features)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass

    # Remove hydrogens for standard graph representation
    mol = Chem.RemoveHs(mol)

    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom, use_chirality))

    if len(atom_features) == 0:
        return None

    # Extract bond features and edge indices
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feat = get_bond_features(bond, use_stereo)

        # Add both directions (undirected graph)
        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([bond_feat, bond_feat])

    # Add self-loops if requested
    if add_self_loops:
        num_atoms = len(atom_features)
        for i in range(num_atoms):
            edge_indices.append([i, i])
            # Self-loop features (zeros)
            edge_features.append([0.0] * len(edge_features[0]) if edge_features else [])

    # Convert to tensors
    x = torch.tensor(atom_features, dtype=torch.float)

    # Calculate bond feature dimension
    bond_feat_dim = len(BOND_FEATURES['bond_type']) + 1  # +1 for unknown
    bond_feat_dim += 2  # is_conjugated, is_in_ring
    if use_stereo:
        bond_feat_dim += len(BOND_FEATURES['stereo']) + 1  # +1 for unknown

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        # Handle molecules with no bonds (single atoms)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, bond_feat_dim), dtype=torch.float)

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
        num_nodes=x.size(0),
    )

    return data


class MoleculeFeaturizer:
    """
    A comprehensive molecular featurizer that handles various representations.

    Supports:
    - Graph representation (atoms as nodes, bonds as edges)
    - Fingerprint representation (Morgan, MACCS, etc.)
    - Global molecular descriptors
    """

    def __init__(
        self,
        use_chirality: bool = True,
        use_stereo: bool = True,
        fingerprint_type: str = 'morgan',
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 1024,
    ):
        """
        Initialize the featurizer.

        Args:
            use_chirality: Include chirality in atom features
            use_stereo: Include stereochemistry in bond features
            fingerprint_type: Type of fingerprint ('morgan', 'maccs', 'rdkit')
            fingerprint_radius: Radius for Morgan fingerprints
            fingerprint_bits: Number of bits for fingerprints
        """
        self.use_chirality = use_chirality
        self.use_stereo = use_stereo
        self.fingerprint_type = fingerprint_type
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES to graph representation."""
        return smiles_to_graph(
            smiles,
            use_chirality=self.use_chirality,
            use_stereo=self.use_stereo,
        )

    def smiles_to_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert SMILES to molecular fingerprint.

        Args:
            smiles: SMILES string

        Returns:
            Numpy array of fingerprint bits
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for fingerprint generation")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if self.fingerprint_type == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.fingerprint_radius,
                nBits=self.fingerprint_bits
            )
        elif self.fingerprint_type == 'maccs':
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        elif self.fingerprint_type == 'rdkit':
            fp = Chem.RDKFingerprint(mol, fpSize=self.fingerprint_bits)
        else:
            raise ValueError(f"Unknown fingerprint type: {self.fingerprint_type}")

        return np.array(fp)

    def smiles_to_descriptors(self, smiles: str) -> Optional[Dict[str, float]]:
        """
        Calculate global molecular descriptors.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of descriptor names and values
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for descriptor calculation")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        descriptors = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_hbd': Descriptors.NumHDonors(mol),
            'num_hba': Descriptors.NumHAcceptors(mol),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'fraction_sp3': rdMolDescriptors.CalcFractionCSP3(mol),
        }

        return descriptors

    def get_atom_feature_dim(self) -> int:
        """Get the dimension of atom features."""
        # Calculate based on one-hot encoding sizes
        dim = 0
        dim += len(ATOM_FEATURES['atomic_num']) + 1  # +1 for unknown
        dim += len(ATOM_FEATURES['degree']) + 1
        dim += len(ATOM_FEATURES['formal_charge']) + 1
        dim += len(ATOM_FEATURES['num_hs']) + 1
        dim += len(ATOM_FEATURES['hybridization']) + 1
        dim += 1  # is_aromatic
        dim += 3  # mass, radical electrons, is_in_ring

        if self.use_chirality:
            dim += len(ATOM_FEATURES['chiral_tag']) + 1

        return dim

    def get_bond_feature_dim(self) -> int:
        """Get the dimension of bond features."""
        dim = 0
        dim += len(BOND_FEATURES['bond_type']) + 1  # +1 for unknown
        dim += 2  # is_conjugated, is_in_ring

        if self.use_stereo:
            dim += len(BOND_FEATURES['stereo']) + 1

        return dim


def batch_smiles_to_graphs(
    smiles_list: List[str],
    featurizer: Optional[MoleculeFeaturizer] = None,
    show_progress: bool = True,
) -> Tuple[List[Data], List[int]]:
    """
    Convert a batch of SMILES strings to graphs.

    Args:
        smiles_list: List of SMILES strings
        featurizer: MoleculeFeaturizer instance (creates default if None)
        show_progress: Show progress bar

    Returns:
        Tuple of (list of Data objects, list of failed indices)
    """
    if featurizer is None:
        featurizer = MoleculeFeaturizer()

    graphs = []
    failed_indices = []

    iterator = smiles_list
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(smiles_list, desc="Converting SMILES to graphs")
        except ImportError:
            pass

    for idx, smiles in enumerate(iterator):
        graph = featurizer.smiles_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
        else:
            failed_indices.append(idx)

    return graphs, failed_indices


# For backward compatibility
def get_fingerprint(smiles: str, radius: int = 2, nbits: int = 1024) -> Optional[List[int]]:
    """
    Get Morgan fingerprint for a SMILES string.

    Args:
        smiles: SMILES string
        radius: Fingerprint radius
        nbits: Number of bits

    Returns:
        List of fingerprint bits
    """
    featurizer = MoleculeFeaturizer(
        fingerprint_type='morgan',
        fingerprint_radius=radius,
        fingerprint_bits=nbits,
    )
    fp = featurizer.smiles_to_fingerprint(smiles)
    return list(fp) if fp is not None else None
