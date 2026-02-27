# hergat/preprocess.py
# RDKit-based preprocessing for hERGAT:
#  - Graph inputs (atom/bond features + neighbor indices)
#  - Physicochemical vector = Morgan(1024, radius=3) + scaled QED props (MW, ALOGP, HBA, HBD, PSA)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, QED

from .featurizer import atom_features, bond_features


@dataclass
class ScalerParams:
    mean: np.ndarray  # shape (5,)
    scale: np.ndarray  # shape (5,)

    @classmethod
    def from_checkpoint(cls, d: Dict) -> "ScalerParams":
        mean = np.asarray(d["mean"], dtype=float)
        scale = np.asarray(d["scale"], dtype=float)
        if mean.shape != (5,) or scale.shape != (5,):
            raise ValueError("ScalerParams must have mean/scale of shape (5,).")
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


PHYS_ORDER = ["MW", "ALOGP", "HBA", "HBD", "PSA"]


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


# -----------------------
# Graph preprocessing
# -----------------------
DEGREES = [0, 1, 2, 3, 4, 5]


class MolGraph:
    def __init__(self):
        self.nodes: Dict = {}

    def new_node(self, ntype, features=None, rdkit_ix=None):
        node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(node)
        return node

    def add_subgraph(self, subgraph):
        for ntype in set(self.nodes.keys()) | set(subgraph.nodes.keys()):
            self.nodes.setdefault(ntype, []).extend(subgraph.nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i: [] for i in DEGREES}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in DEGREES:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes["atom"]])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        neighbor_idxs = {n: i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor] for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]


class Node:
    __slots__ = ["ntype", "features", "_neighbors", "rdkit_ix"]

    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]


def graph_from_smiles(smiles: str) -> MolGraph:
    graph = MolGraph()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse SMILES")
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node("atom", features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

    for bond in mol.GetBonds():
        atom1 = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2 = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node("bond", features=bond_features(bond))
        new_bond_node.add_neighbors((atom1, atom2))
        atom1.add_neighbors((atom2,))

    mol_node = graph.new_node("molecule")
    mol_node.add_neighbors(graph.nodes["atom"])
    return graph


def array_rep_from_smiles(molgraph: MolGraph) -> Dict:
    arrayrep = {
        "atom_features": molgraph.feature_array("atom"),
        "bond_features": molgraph.feature_array("bond"),
        "atom_list": molgraph.neighbor_list("molecule", "atom"),
        "rdkit_ix": molgraph.rdkit_ix_array(),
    }

    for degree in DEGREES:
        arrayrep[("atom_neighbors", degree)] = np.array(molgraph.neighbor_list(("atom", degree), "atom"), dtype=int)
        arrayrep[("bond_neighbors", degree)] = np.array(molgraph.neighbor_list(("atom", degree), "bond"), dtype=int)
    return arrayrep


def smiles_to_graph_arrays(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x_atom:   (L, A)
      x_bond:   (M, B)
      x_atom_i: (L, K)
      x_bond_i: (L, K)
      x_mask:   (L,)
      rdkit_ix: (L-1,)  # excludes pad row
    """
    molgraph = graph_from_smiles(smiles)
    molgraph.sort_nodes_by_degree("atom")
    arrayrep = array_rep_from_smiles(molgraph)

    atom_features_arr = arrayrep["atom_features"]
    bond_features_arr = arrayrep["bond_features"]
    rdkit_ix = arrayrep["rdkit_ix"]

    atom_len = atom_features_arr.shape[0]
    bond_len = bond_features_arr.shape[0]

    # padding row at the end (same as training)
    max_atom_len = atom_len + 1
    max_bond_len = bond_len + 1
    pad_atom_idx = atom_len
    pad_bond_idx = bond_len

    num_atom_feat = atom_features_arr.shape[1]
    num_bond_feat = bond_features_arr.shape[1]

    mask = np.zeros((max_atom_len,), dtype=np.float32)
    atoms = np.zeros((max_atom_len, num_atom_feat), dtype=np.float32)
    bonds = np.zeros((max_bond_len, num_bond_feat), dtype=np.float32)
    atom_neighbors = np.zeros((max_atom_len, len(DEGREES)), dtype=np.int64)
    bond_neighbors = np.zeros((max_atom_len, len(DEGREES)), dtype=np.int64)
    atom_neighbors.fill(pad_atom_idx)
    bond_neighbors.fill(pad_bond_idx)

    # fill atoms/bonds
    for i, feat in enumerate(atom_features_arr):
        mask[i] = 1.0
        atoms[i] = feat

    for j, feat in enumerate(bond_features_arr):
        bonds[j] = feat

    atom_neighbor_count = 0
    bond_neighbor_count = 0

    for degree in DEGREES:
        atom_neighbors_list = arrayrep[("atom_neighbors", degree)]
        bond_neighbors_list = arrayrep[("bond_neighbors", degree)]

        if len(atom_neighbors_list) > 0:
            for degree_array in atom_neighbors_list:
                for jj, value in enumerate(degree_array):
                    atom_neighbors[atom_neighbor_count, jj] = int(value)
                atom_neighbor_count += 1

        if len(bond_neighbors_list) > 0:
            for degree_array in bond_neighbors_list:
                for jj, value in enumerate(degree_array):
                    bond_neighbors[bond_neighbor_count, jj] = int(value)
                bond_neighbor_count += 1

    return atoms, bonds, atom_neighbors, bond_neighbors, mask, rdkit_ix


# -----------------------
# Physicochemical vector
# -----------------------
def morgan_bits(mol: Chem.Mol, radius: int = 3, n_bits: int = 1024) -> np.ndarray:
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
    arr = np.zeros((int(n_bits),), dtype=np.float32)
    # RDKit's ConvertToNumpyArray is fastest, but keep simple:
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def qed_phys_props(mol: Chem.Mol) -> np.ndarray:
    props = QED.properties(mol)  # namedtuple
    # match training: MW, ALOGP, HBA, HBD, PSA (5)
    x = np.array([props.MW, props.ALOGP, props.HBA, props.HBD, props.PSA], dtype=np.float32)
    return x


def smiles_to_physchem_vector(smiles: str, scaler: ScalerParams, morgan_radius: int = 3, morgan_bits_n: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    Chem.SanitizeMol(mol)

    fp = morgan_bits(mol, radius=morgan_radius, n_bits=morgan_bits_n)
    props = qed_phys_props(mol)
    props_scaled = scaler.transform(props.astype(float)).astype(np.float32)
    vec = np.concatenate([fp, props_scaled], axis=0).astype(np.float32)
    return vec
