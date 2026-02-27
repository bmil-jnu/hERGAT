#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate paper figures for TOP-K molecules by predicted blocker probability.

- Loads a trained checkpoint (.pt)
- Predicts probabilities for a CSV (SMILES + optional label)
- Selects TOP-K (default: from test split indices if provided)
- Saves:
  - molecule-level attention highlight PNG + attention heatmap
  - atom-level attention heatmaps + per-layer highlight PNGs
  - a JSON summary (includes paths + probabilities)

Typical usage (after training):
python scripts/make_topk_figures.py \
  --csv data/hERGAT_final_dataset.csv \
  --ckpt outputs/train_run/ckpt/hERGAT_best.pt \
  --out_dir outputs/train_run/figures \
  --smiles_col SMILES \
  --label_col Class \
  --top_k 20 \
  --device cuda:0 \
  --split_indices outputs/train_run/split_indices.json \
  --top_from test
"""
from __future__ import annotations

import os
import json
import argparse
import sys
from typing import Dict, Any, List, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch

from rdkit import Chem  # noqa: F401

from hergat.checkpoint import load_checkpoint
from hergat.preprocess import canonicalize_smiles, smiles_to_graph_arrays, smiles_to_physchem_vector
from hergat.visualize import mol_to_png_highlight, mol_to_svg_highlight, save_attention_heatmap


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# =========================
# ✅ NEW: robust split loader
# =========================
def _pick_first_key(d: dict, candidates: List[str]) -> str | None:
    for k in candidates:
        if k in d:
            return k
    return None


def _unwrap_split_container(splits: dict) -> dict:
    """
    Some files might be saved as {"split": {...}} or {"indices": {...}}.
    If so, unwrap once.
    """
    if not isinstance(splits, dict):
        return splits
    for container in ("split", "splits", "indices", "index", "idx"):
        if container in splits and isinstance(splits[container], dict):
            return splits[container]
    return splits


def _get_indices_from_splits(splits: dict, which: str) -> List[int]:
    """
    Return indices list for 'train'/'val'/'test' allowing multiple key names.
    """
    splits = _unwrap_split_container(splits)

    which = which.lower()
    if which == "train":
        candidates = [
            "train_indices", "train_idx", "train", "idx_train", "train_index", "train_idxs", "train_ids", "train_ind"
        ]
    elif which in ("val", "valid", "validation"):
        candidates = [
            "val_indices", "valid_indices", "val_idx", "valid_idx", "val", "valid",
            "idx_val", "val_index", "val_idxs", "val_ids", "val_ind", "validation_indices"
        ]
    elif which == "test":
        candidates = [
            "test_indices", "test_idx", "test", "idx_test", "test_index", "test_idxs", "test_ids", "test_ind"
        ]
    else:
        raise ValueError(f"Unknown split name: {which}")

    key = _pick_first_key(splits, candidates)
    if key is None:
        raise ValueError(
            f"split_indices.json must contain '{which}' indices. "
            f"Tried keys={candidates}. Found keys={list(splits.keys())}"
        )

    idx = splits[key]

    # sometimes indices may be saved as dict {"0": 12, "1": 99}
    if isinstance(idx, dict):
        idx = list(idx.values())

    if not isinstance(idx, (list, tuple)):
        raise ValueError(f"'{key}' must be list/tuple/dict, got {type(idx)}")

    out = []
    for x in idx:
        try:
            out.append(int(x))
        except Exception as e:
            raise ValueError(f"Non-integer index in '{key}': {x!r}") from e
    return out


def _resolve_smiles_split_to_indices(splits: dict, which: str, smiles_all: Sequence[str]) -> List[int]:
    """Resolve split smiles (e.g., test_smiles) to row indices in the input CSV."""
    which = which.lower()
    if which == "train":
        candidates = ["train_smiles", "smiles_train", "train_smile", "train_smi"]
    elif which in ("val", "valid", "validation"):
        candidates = ["val_smiles", "valid_smiles", "validation_smiles", "smiles_val", "val_smi"]
    elif which == "test":
        candidates = ["test_smiles", "smiles_test", "test_smile", "test_smi"]
    else:
        raise ValueError(f"Unknown split name: {which}")

    key = _pick_first_key(splits, candidates)
    if key is None:
        raise ValueError(
            f"split file must contain '{which}' indices or smiles. "
            f"Tried smiles keys={candidates}. Found keys={list(splits.keys())}"
        )

    split_smiles = splits[key]
    if not isinstance(split_smiles, (list, tuple)):
        raise ValueError(f"'{key}' must be list/tuple, got {type(split_smiles)}")

    canon_to_indices: Dict[str, List[int]] = {}
    raw_to_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(smiles_all):
        s_raw = str(s)
        raw_to_indices.setdefault(s_raw, []).append(i)
        try:
            c = canonicalize_smiles(s_raw)
            canon_to_indices.setdefault(c, []).append(i)
        except Exception:
            pass

    out: List[int] = []
    seen = set()
    for s in split_smiles:
        s_raw = str(s)
        matched = False

        if s_raw in raw_to_indices:
            for i in raw_to_indices[s_raw]:
                if i not in seen:
                    out.append(i)
                    seen.add(i)
                    matched = True
                    break

        if not matched:
            try:
                c = canonicalize_smiles(s_raw)
                if c in canon_to_indices:
                    for i in canon_to_indices[c]:
                        if i not in seen:
                            out.append(i)
                            seen.add(i)
                            matched = True
                            break
            except Exception:
                pass

    if not out:
        raise RuntimeError(
            f"Failed to match any rows for split smiles key='{key}'. "
            "Check CSV smiles column / canonicalization consistency."
        )

    return out


def _get_subset_indices(splits: dict, which: str, smiles_all: Sequence[str]) -> List[int]:
    splits = _unwrap_split_container(splits)
    try:
        return _get_indices_from_splits(splits, which)
    except ValueError:
        return _resolve_smiles_split_to_indices(splits, which, smiles_all)


@torch.no_grad()
def _predict_with_attention(
    model: torch.nn.Module,
    bundle,
    smiles: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Run one forward pass and return probs + attention arrays."""
    canon = canonicalize_smiles(smiles)

    atoms, bonds, atom_neighbors, bond_neighbors, mask, rdkit_ix = smiles_to_graph_arrays(canon)
    phys_vec = smiles_to_physchem_vector(
        canon,
        scaler=bundle.scaler,
        morgan_radius=bundle.morgan["radius"],
        morgan_bits_n=bundle.morgan["nBits"],
    )

    # add batch dim
    atom_t = torch.from_numpy(atoms).unsqueeze(0).float().to(device)
    bond_t = torch.from_numpy(bonds).unsqueeze(0).float().to(device)
    atom_i = torch.from_numpy(atom_neighbors).unsqueeze(0).long().to(device)
    bond_i = torch.from_numpy(bond_neighbors).unsqueeze(0).long().to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).float().to(device)
    phys_t = torch.from_numpy(phys_vec).unsqueeze(0).float().to(device)

    out = model(atom_t, bond_t, atom_i, bond_i, mask_t, phys_t)
    probs = out[-1]
    probs_np = probs.detach().cpu().numpy().reshape(-1)

    # attention
    atom_attention_weight_viz = out[2]  # list of tensors (B,L,K,1)
    mol_attention_weight_viz = out[5]   # list of tensors (B,L,1)

    atom_att = [a.detach().cpu().numpy()[0, :, :, 0] for a in atom_attention_weight_viz]  # list (L,K)
    mol_att = np.stack([m.detach().cpu().numpy()[0, :, 0] for m in mol_attention_weight_viz], axis=0)  # (T, L)
    valid = mask.astype(bool)
    mol_att = mol_att[:, valid]  # (T, N_atoms_valid)

    return {
        "canon_smiles": canon,
        "prob_non_blocker": float(probs_np[0]),
        "prob_blocker": float(probs_np[1]),
        "pred_class": int(np.argmax(probs_np)),
        "atom_attention": atom_att,
        "mol_attention": mol_att,
    }


def _save_matrix_heatmap(mat: np.ndarray, out_png: str, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(max(6, mat.shape[1] * 0.35), max(3.5, mat.shape[0] * 0.25)))
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--label_col", default="Class")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--top_from", choices=["test", "all"], default="test")
    ap.add_argument("--split_indices", default="", help="Optional split_indices.json from training to select test subset")
    ap.add_argument("--save_svg", action="store_true", help="Also save SVG highlights")
    ap.add_argument("--max_items", type=int, default=0, help="Debug: limit number of rows processed (0=all)")
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)
    mol_dir = os.path.join(args.out_dir, "molecule_level")
    atom_dir = os.path.join(args.out_dir, "atom_level")
    _safe_mkdir(mol_dir)
    _safe_mkdir(atom_dir)

    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"CSV missing smiles_col='{args.smiles_col}'")
    has_label = args.label_col in df.columns

    smiles_all = df[args.smiles_col].astype(str).tolist()
    y_all = df[args.label_col].astype(int).to_numpy() if has_label else None

    if args.max_items and args.max_items > 0:
        smiles_all = smiles_all[: args.max_items]
        if y_all is not None:
            y_all = y_all[: args.max_items]

    # =========================
    # ✅ UPDATED: choose subset (test or all) robustly
    # =========================
    subset_ix = list(range(len(smiles_all)))

    if args.top_from == "test":
        if not args.split_indices:
            raise ValueError("--split_indices is required when --top_from test")
        with open(args.split_indices, "r", encoding="utf-8") as f:
            splits = json.load(f)

        # robust key search: supports index-based and smiles-based split files
        subset_ix = _get_subset_indices(splits, "test", smiles_all)
        subset_ix = [i for i in subset_ix if 0 <= i < len(smiles_all)]

        if not subset_ix:
            raise RuntimeError(
                "Test indices are empty after filtering bounds. "
                "Check split_indices.json and the CSV row count."
            )

    # load model once
    device = torch.device(args.device)
    model, bundle = load_checkpoint(args.ckpt, device=device)
    model.eval()

    # predict probs (no images yet)
    probs = []
    valid_rows = []
    dropped = 0
    for i in subset_ix:
        s = smiles_all[i]
        try:
            canon = canonicalize_smiles(s)
            atoms, bonds, atom_neighbors, bond_neighbors, mask, rdkit_ix = smiles_to_graph_arrays(canon)
            phys_vec = smiles_to_physchem_vector(
                canon, scaler=bundle.scaler,
                morgan_radius=bundle.morgan["radius"],
                morgan_bits_n=bundle.morgan["nBits"]
            )

            atom_t = torch.from_numpy(atoms).unsqueeze(0).float().to(device)
            bond_t = torch.from_numpy(bonds).unsqueeze(0).float().to(device)
            atom_i = torch.from_numpy(atom_neighbors).unsqueeze(0).long().to(device)
            bond_i = torch.from_numpy(bond_neighbors).unsqueeze(0).long().to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).float().to(device)
            phys_t = torch.from_numpy(phys_vec).unsqueeze(0).float().to(device)

            out = model(atom_t, bond_t, atom_i, bond_i, mask_t, phys_t)
            p = out[-1].detach().cpu().numpy().reshape(-1)
            probs.append((float(p[1]), float(p[0])))
            valid_rows.append(i)
        except Exception:
            dropped += 1
            continue

    if not probs:
        raise RuntimeError("No valid molecules for prediction. Check SMILES column / RDKit install.")

    # select top-k by prob_blocker
    prob_blocker = np.array([p[0] for p in probs], dtype=float)
    order = np.argsort(-prob_blocker)
    k = min(args.top_k, order.size)
    top_idx_local = order[:k]
    top_global_ix = [valid_rows[j] for j in top_idx_local]

    # save topk.csv
    pb_map = {valid_rows[j]: probs[j][0] for j in range(len(valid_rows))}
    pn_map = {valid_rows[j]: probs[j][1] for j in range(len(valid_rows))}

    top_rows = []
    for rank, gi in enumerate(top_global_ix, start=1):
        row = {
            "rank": rank,
            "row_index": gi,
            "smiles": smiles_all[gi],
            "prob_blocker": float(pb_map[gi]),
            "prob_non_blocker": float(pn_map[gi]),
        }
        if y_all is not None:
            row["true_label"] = int(y_all[gi])
        top_rows.append(row)

    topk_csv = os.path.join(args.out_dir, "topk_predictions.csv")
    pd.DataFrame(top_rows).to_csv(topk_csv, index=False, encoding="utf-8-sig")

    # generate figures for each topk item (with attention)
    entries: List[Dict[str, Any]] = []
    for row in top_rows:
        rank = int(row["rank"])
        s = row["smiles"]
        prefix = f"top{rank:03d}"
        try:
            pred = _predict_with_attention(model, bundle, s, device=device)
        except Exception as e:
            entries.append({**row, "error": str(e)})
            continue

        # molecule-level
        mol_att = pred["mol_attention"]  # (T, N_atoms)
        atom_weights = mol_att.mean(axis=0)
        mol_png = os.path.join(mol_dir, f"{prefix}_mol.png")
        mol_heat = os.path.join(mol_dir, f"{prefix}_mol_heatmap.png")
        mol_to_png_highlight(pred["canon_smiles"], atom_weights, mol_png)
        save_attention_heatmap(mol_att, mol_heat, title=f"Molecule attention (rank {rank})")

        mol_svg = ""
        if args.save_svg:
            mol_svg = os.path.join(mol_dir, f"{prefix}_mol.svg")
            mol_to_svg_highlight(pred["canon_smiles"], atom_weights, mol_svg)

        # atom-level per layer
        atom_files = []
        atom_att_list = pred["atom_attention"]  # list of (L,K)
        agg_weights_all = []
        for li, att in enumerate(atom_att_list, start=1):
            # per-atom weight = mean over neighbor dimension
            w = att.mean(axis=1)  # (L,)
            agg_weights_all.append(w)

            layer_png = os.path.join(atom_dir, f"{prefix}_atom_layer{li:02d}.png")
            layer_heat = os.path.join(atom_dir, f"{prefix}_atom_layer{li:02d}_heatmap.png")
            mol_to_png_highlight(pred["canon_smiles"], w, layer_png)
            _save_matrix_heatmap(att, layer_heat, title=f"Atom attention layer {li} (rank {rank})",
                                 xlabel="Neighbor slot", ylabel="Atom index")
            atom_files.append({"layer": li, "highlight_png": layer_png, "heatmap_png": layer_heat})

        # aggregated atom attention across layers
        if agg_weights_all:
            w_agg = np.mean(np.stack(agg_weights_all, axis=0), axis=0)
            agg_png = os.path.join(atom_dir, f"{prefix}_atom_agg.png")
            mol_to_png_highlight(pred["canon_smiles"], w_agg, agg_png)
        else:
            agg_png = ""

        entries.append({
            **row,
            "canon_smiles": pred["canon_smiles"],
            "pred_class": pred["pred_class"],
            "molecule_level": {
                "highlight_png": mol_png,
                "heatmap_png": mol_heat,
                "highlight_svg": mol_svg,
            },
            "atom_level": {
                "per_layer": atom_files,
                "aggregate_png": agg_png,
            },
        })

    summary = {
        "csv": os.path.abspath(args.csv),
        "ckpt": os.path.abspath(args.ckpt),
        "smiles_col": args.smiles_col,
        "label_col": args.label_col if has_label else "",
        "top_k": int(args.top_k),
        "top_from": args.top_from,
        "n_rows_in_subset": int(len(subset_ix)),
        "n_predicted_valid": int(len(valid_rows)),
        "dropped_invalid_for_prediction": int(dropped),
        "topk_csv": os.path.abspath(topk_csv),
        "entries": entries,
    }

    out_json = os.path.join(args.out_dir, "topk_figures_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[saved] {topk_csv}")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()