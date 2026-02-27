# hergat/visualize.py
from __future__ import annotations

from typing import List, Tuple, Dict
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def mol_to_svg_highlight(smiles: str, atom_weights: np.ndarray, out_svg: str, mol_size=(450, 320)) -> None:
    """
    Create an SVG with atoms highlighted by weights (0..1).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    Chem.SanitizeMol(mol)

    # normalize
    w = atom_weights.astype(float)
    if w.size == 0:
        w = np.array([0.0])
    w = w - np.min(w)
    if np.max(w) > 1e-12:
        w = w / np.max(w)

    # RDKit expects dict: atom_idx -> (r,g,b)
    # Use simple red intensity; keep deterministic.
    atom_cols = {int(i): (float(wi), 0.0, 0.0) for i, wi in enumerate(w)}

    drawer = rdMolDraw2D.MolDraw2DSVG(mol_size[0], mol_size[1])
    opts = drawer.drawOptions()
    opts.addAtomIndices = True  # helpful for debugging
    drawer.DrawMolecule(mol, highlightAtoms=list(atom_cols.keys()), highlightAtomColors=atom_cols)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)


def save_attention_heatmap(att_mat: np.ndarray, out_png: str, title: str = "Molecule attention over T steps") -> None:
    """
    att_mat: (T, N_atoms)
    """
    if att_mat.ndim != 2:
        raise ValueError("att_mat must be 2D (T, N_atoms)")
    plt.figure(figsize=(max(6, att_mat.shape[1] * 0.35), 3.5))
    plt.imshow(att_mat, aspect="auto")
    plt.yticks(range(att_mat.shape[0]), [f"t={i+1}" for i in range(att_mat.shape[0])])
    plt.xticks(range(att_mat.shape[1]), [str(i) for i in range(att_mat.shape[1])], rotation=90)
    plt.title(title)
    plt.xlabel("Atom index (RDKit order)")
    plt.ylabel("Step")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def mol_to_png_highlight(smiles: str, atom_weights: np.ndarray, out_png: str, mol_size=(500, 500)) -> None:
    """
    Create a PNG with atoms highlighted by weights (0..1) using RDKit MolDraw2DCairo.
    This avoids SVG->PNG conversion dependencies.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    Chem.SanitizeMol(mol)

    w = np.asarray(atom_weights, dtype=float)
    n_atoms = mol.GetNumAtoms()
    if w.ndim != 1:
        w = w.reshape(-1)
    if w.size < n_atoms:
        w = np.pad(w, (0, n_atoms - w.size))
    w = w[:n_atoms]

    # normalize 0..1
    w = w - float(np.min(w)) if w.size else w
    mx = float(np.max(w)) if w.size else 0.0
    if mx > 1e-12:
        w = w / mx
    else:
        w = np.zeros_like(w)

    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    atom_radii: Dict[int, float] = {}
    highlight_atoms: List[int] = []

    for i in range(n_atoms):
        val = float(w[i])
        if val <= 0:
            continue
        highlight_atoms.append(i)
        # red intensity
        atom_colors[i] = (1.0, 1.0 - 0.75 * val, 1.0 - 0.75 * val)
        atom_radii[i] = 0.35 + 0.35 * val

    drawer = rdMolDraw2D.MolDraw2DCairo(int(mol_size[0]), int(mol_size[1]))
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    with open(out_png, "wb") as f:
        f.write(png)
