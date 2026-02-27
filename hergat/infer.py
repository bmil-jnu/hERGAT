# hergat/infer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import os
import numpy as np
import torch

from .checkpoint import load_checkpoint, CheckpointBundle
from .preprocess import canonicalize_smiles, smiles_to_graph_arrays, smiles_to_physchem_vector
from .visualize import mol_to_svg_highlight, mol_to_png_highlight, save_attention_heatmap


@dataclass
class InferenceResult:
    canon_smiles: str
    prob_non_blocker: float
    prob_blocker: float
    pred_label: str
    pred_class: int
    # attention (optional)
    atom_attention: Optional[List[np.ndarray]] = None  # list of (L,K)
    mol_attention: Optional[np.ndarray] = None         # (T, N_atoms)
    images: Optional[Dict[str, str]] = None            # {"svg": "...", "heatmap": "..."}


def predict_one(
    smiles: str,
    checkpoint_path: str,
    device: torch.device,
    return_attention: bool = False,
    save_images: bool = False,
    image_prefix: Optional[str] = None,
    image_root: Optional[str] = None,
    image_host: Optional[str] = None,
) -> InferenceResult:
    """
    Single-SMILES inference.
    If save_images=True, image_root must be provided.
    image_prefix is used for filenames (no extension).
    """
    canon = canonicalize_smiles(smiles)

    model, bundle = load_checkpoint(checkpoint_path, device=device)

    atoms, bonds, atom_neighbors, bond_neighbors, mask, rdkit_ix = smiles_to_graph_arrays(canon)
    phys_vec = smiles_to_physchem_vector(canon, scaler=bundle.scaler, morgan_radius=bundle.morgan["radius"], morgan_bits_n=bundle.morgan["nBits"])

    # add batch dim
    atom_t = torch.from_numpy(atoms).unsqueeze(0).float()
    bond_t = torch.from_numpy(bonds).unsqueeze(0).float()
    atom_i = torch.from_numpy(atom_neighbors).unsqueeze(0).long()
    bond_i = torch.from_numpy(bond_neighbors).unsqueeze(0).long()
    mask_t = torch.from_numpy(mask).unsqueeze(0).float()
    phys_t = torch.from_numpy(phys_vec).unsqueeze(0).float()

    with torch.no_grad():
        (
            _atom_feature,
            _atom_feature_viz,
            atom_attention_weight_viz,
            _mol_feature_viz,
            _mol_feature_unbounded_viz,
            mol_attention_weight_viz,
            probs,
        ) = model(atom_t, bond_t, atom_i, bond_i, mask_t, phys_t)

    probs_np = probs.detach().cpu().numpy().reshape(-1)
    if probs_np.shape[0] != 2:
        raise ValueError("Model output must be 2-class softmax.")

    prob0 = float(probs_np[0])
    prob1 = float(probs_np[1])
    pred_class = int(np.argmax(probs_np))
    pred_label = bundle.label_map.get(str(pred_class), str(pred_class))

    result = InferenceResult(
        canon_smiles=canon,
        prob_non_blocker=prob0,
        prob_blocker=prob1,
        pred_label=pred_label,
        pred_class=pred_class,
    )

    if return_attention:
        # atom attention list: each element (B,L,K,1) -> (L,K)
        atom_att = [a.detach().cpu().numpy()[0, :, :, 0] for a in atom_attention_weight_viz]
        result.atom_attention = atom_att

        # mol attention list: each element (B,L,1) -> (L,)
        mol_att = np.stack([m.detach().cpu().numpy()[0, :, 0] for m in mol_attention_weight_viz], axis=0)  # (T, L)
        # remove pad row using mask
        valid = mask.astype(bool)
        mol_att = mol_att[:, valid]
        result.mol_attention = mol_att

    if save_images:
        if image_root is None or image_prefix is None:
            raise ValueError("image_root and image_prefix are required when save_images=True")

        os.makedirs(image_root, exist_ok=True)
        images = {}

        if result.mol_attention is not None:
            # aggregate attention over T (mean) and map to atoms
            atom_weights = result.mol_attention.mean(axis=0)  # (N_atoms_valid,)
            # RDKit order in mol_attention corresponds to sorted-atom order used by model, not original RDKit atom indices.
            # For human readability, we keep this order in the drawing with atom indices displayed.
            mol_png_path = os.path.join(image_root, f"{image_prefix}_01.png")
            mol_to_png_highlight(result.canon_smiles, atom_weights, mol_png_path)
            images["mol_png_path"] = mol_png_path

            heat_path = os.path.join(image_root, f"{image_prefix}_02_heatmap.png")
            save_attention_heatmap(result.mol_attention, heat_path)
            images["heatmap_path"] = heat_path

            if image_host:
                images["mol_png_url"] = f"{image_host.rstrip('/')}/{os.path.basename(mol_png_path)}"
                images["heatmap_url"] = f"{image_host.rstrip('/')}/{os.path.basename(heat_path)}"

        result.images = images

    # cleanup (match guide)
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return result
