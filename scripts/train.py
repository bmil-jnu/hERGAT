#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train hERGAT (ExtendedFingerprint) from a single CSV and export outputs/.../ckpt/hERGAT_best.pt.

- Input: one CSV containing SMILES + binary label.
- Split: stratified train/val/test (by default 80/10/10).
- Outputs:
    - out_ckpt: torch checkpoint with model_state_dict + hparams + scaler
    - metrics.json, roc_curve.json, pr_curve.json, split_indices.json
        (contains both *_indices and *_smiles keys)

This script is designed to be runnable from terminal with a clean conda environment.
"""
from __future__ import annotations

import os
import json
import argparse
import sys
from dataclasses import asdict
from typing import List, Tuple, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from hergat.model import HergatHParams, ExtendedFingerprint
from hergat.preprocess import canonicalize_smiles, smiles_to_graph_arrays, ScalerParams, qed_phys_props, morgan_bits, smiles_to_physchem_vector
from hergat.checkpoint import save_checkpoint


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int = 2024) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Dataset + Collate
# -------------------------
class HergatDataset(Dataset):
    def __init__(self, smiles: List[str], labels: np.ndarray, scaler: ScalerParams):
        self.smiles = smiles
        self.labels = labels.astype(np.int64)
        self.scaler = scaler

        self.graphs = []
        self.phys = []

        for s in self.smiles:
            # graph arrays (variable length)
            g = smiles_to_graph_arrays(s)
            self.graphs.append(g)
            # physchem vector (fixed 1024 + 5)
            v = smiles_to_physchem_vector(s, scaler)
            self.phys.append(v)

        self.phys = np.asarray(self.phys, dtype=np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx: int):
        return self.graphs[idx], self.phys[idx], int(self.labels[idx])


def collate_batch(batch):
    # batch: list of (graph_tuple, phys_vec, label)
    graphs, phys, labels = zip(*batch)
    phys = np.stack(phys, axis=0).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    # unpack graph tuples
    x_atoms, x_bonds, x_atom_i, x_bond_i, x_masks, rdkit_ixs = zip(*graphs)

    max_L = max(x.shape[0] for x in x_atoms)
    max_M = max(x.shape[0] for x in x_bonds)
    A = x_atoms[0].shape[1]
    B = x_bonds[0].shape[1]
    K = x_atom_i[0].shape[1]

    bat_atoms = np.zeros((len(batch), max_L, A), dtype=np.float32)
    bat_bonds = np.zeros((len(batch), max_M, B), dtype=np.float32)
    bat_atom_i = np.zeros((len(batch), max_L, K), dtype=np.int64)
    bat_bond_i = np.zeros((len(batch), max_L, K), dtype=np.int64)
    bat_mask = np.zeros((len(batch), max_L), dtype=np.float32)

    for i in range(len(batch)):
        L = x_atoms[i].shape[0]
        M = x_bonds[i].shape[0]
        bat_atoms[i, :L, :] = x_atoms[i]
        bat_bonds[i, :M, :] = x_bonds[i]
        bat_atom_i[i, :L, :] = x_atom_i[i]
        bat_bond_i[i, :L, :] = x_bond_i[i]
        bat_mask[i, :L] = x_masks[i]

    return (
        torch.from_numpy(bat_atoms),
        torch.from_numpy(bat_bonds),
        torch.from_numpy(bat_atom_i),
        torch.from_numpy(bat_bond_i),
        torch.from_numpy(bat_mask),
        torch.from_numpy(phys),
        torch.from_numpy(labels),
    )


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_y = []
    all_p = []
    losses = []

    for x_atom, x_bond, x_atom_i, x_bond_i, x_mask, x_phys, y in loader:
        x_atom = x_atom.to(device)
        x_bond = x_bond.to(device)
        x_atom_i = x_atom_i.to(device)
        x_bond_i = x_bond_i.to(device)
        x_mask = x_mask.to(device)
        x_phys = x_phys.to(device)
        y = y.to(device)

        out = model(x_atom, x_bond, x_atom_i, x_bond_i, x_mask, x_phys)
        probs = out[-1] if isinstance(out, (tuple, list)) else out  # (B,2) probs

        # NLL from probs
        eps = 1e-9
        logp = torch.log(probs + eps)
        loss = F.nll_loss(logp, y)

        losses.append(float(loss.detach().cpu().item()))
        all_y.append(y.detach().cpu().numpy())
        all_p.append(probs[:, 1].detach().cpu().numpy())

    y_true = np.concatenate(all_y, axis=0) if all_y else np.array([], dtype=np.int64)
    p_pos = np.concatenate(all_p, axis=0) if all_p else np.array([], dtype=np.float32)

    res = {"loss": float(np.mean(losses)) if losses else float("nan")}
    if y_true.size > 0 and len(np.unique(y_true)) == 2:
        res["auroc"] = float(roc_auc_score(y_true, p_pos))
        res["aupr"] = float(average_precision_score(y_true, p_pos))
        fpr, tpr, thr = roc_curve(y_true, p_pos)
        pr, rc, thr2 = precision_recall_curve(y_true, p_pos)
        res["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thr.tolist()}
        res["pr_curve"] = {"precision": pr.tolist(), "recall": rc.tolist(), "thr": thr2.tolist()}
    else:
        res["auroc"] = float("nan")
        res["aupr"] = float("nan")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with smiles + label columns")
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--label_col", default="Class")
    ap.add_argument("--out_ckpt", default="outputs/train_run/ckpt/hERGAT_best.pt")
    ap.add_argument("--out_dir", default="outputs/train_run")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=218)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=10**-2.13)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--train_size", type=float, default=0.8)
    ap.add_argument("--val_size", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "ckpt"), exist_ok=True)

    seed_everything(args.seed)
    device = torch.device(args.device)

    df = pd.read_csv(args.csv)
    if args.smiles_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {args.smiles_col}, {args.label_col}")

    raw_smiles = df[args.smiles_col].astype(str).tolist()
    raw_y = df[args.label_col].astype(int).to_numpy()

    # canonicalize + drop invalid
    smiles = []
    y = []
    kept_row_indices = []
    dropped = 0
    for ridx, (s, yy) in enumerate(zip(raw_smiles, raw_y)):
        try:
            cs = canonicalize_smiles(s)
            smiles.append(cs)
            y.append(int(yy))
            kept_row_indices.append(int(ridx))
        except Exception:
            dropped += 1
    y = np.asarray(y, dtype=np.int64)
    if len(smiles) == 0:
        raise RuntimeError("No valid SMILES after canonicalization.")

    # stratified split: train / temp
    test_size = 1.0 - args.train_size
    smiles_tr, smiles_tmp, y_tr, y_tmp, idx_tr, idx_tmp = train_test_split(
        smiles, y, kept_row_indices, test_size=test_size, random_state=args.seed, stratify=y
    )
    # split temp into val/test
    val_ratio = args.val_size / (args.val_size + (1.0 - args.train_size - args.val_size))
    smiles_va, smiles_te, y_va, y_te, idx_va, idx_te = train_test_split(
        smiles_tmp, y_tmp, idx_tmp, test_size=(1.0 - val_ratio), random_state=args.seed, stratify=y_tmp
    )

    # scaler from train set props
    props_tr = []
    from rdkit import Chem
    for s in smiles_tr:
        mol = Chem.MolFromSmiles(s)
        Chem.SanitizeMol(mol)
        props_tr.append(qed_phys_props(mol).astype(float))
    props_tr = np.vstack(props_tr)
    mean = props_tr.mean(axis=0)
    scale = props_tr.std(axis=0)
    scale[scale < 1e-12] = 1.0
    scaler = ScalerParams(mean=mean.astype(float), scale=scale.astype(float))

    ds_tr = HergatDataset(smiles_tr, y_tr, scaler)
    ds_va = HergatDataset(smiles_va, y_va, scaler)
    ds_te = HergatDataset(smiles_te, y_te, scaler)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    # hparams (match preprocess: atom/bond feature dim inferred from first sample)
    g0 = ds_tr.graphs[0]
    input_feature_dim = g0[0].shape[1]
    input_bond_dim = g0[1].shape[1]

    hp = HergatHParams(
        radius=2,
        T=2,
        input_feature_dim=input_feature_dim,
        input_bond_dim=input_bond_dim,
        fingerprint_dim=1024,
        output_units_num=2,
        p_dropout=0.2,
        physicochemical_feature_dim=1029,
        physicochemical_feature_dim_1=512,
        physicochemical_feature_dim_2=1024,
        final1_fc1=256,
        final1_fc2=64,
    )

    model = ExtendedFingerprint(hp, device=device).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_aupr = -1.0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for x_atom, x_bond, x_atom_i, x_bond_i, x_mask, x_phys, yb in dl_tr:
            x_atom = x_atom.to(device); x_bond = x_bond.to(device)
            x_atom_i = x_atom_i.to(device); x_bond_i = x_bond_i.to(device)
            x_mask = x_mask.to(device); x_phys = x_phys.to(device); yb = yb.to(device)

            out = model(x_atom, x_bond, x_atom_i, x_bond_i, x_mask, x_phys)
            probs = out[-1]  # (B,2)
            logp = torch.log(probs + 1e-9)
            loss = F.nll_loss(logp, yb)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            tr_losses.append(float(loss.detach().cpu().item()))

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_res = evaluate(model, dl_va, device)
        va_aupr = float(va_res.get("aupr", float("nan")))
        va_auroc = float(va_res.get("auroc", float("nan")))

        print(f"[epoch {epoch:03d}] train_loss={tr_loss:.4f}  val_aupr={va_aupr:.4f}  val_auroc={va_auroc:.4f}")

        history.append({
            "epoch": int(epoch),
            "train_loss": tr_loss,
            "val_loss": float(va_res.get("loss", float("nan"))),
            "val_aupr": va_aupr,
            "val_auroc": va_auroc,
        })

        improved = np.isfinite(va_aupr) and (va_aupr > best_aupr + 1e-6)
        if improved:
            best_aupr = va_aupr
            bad_epochs = 0
            # save best
            save_checkpoint(
                path=args.out_ckpt,
                model=model,
                hp=hp,
                scaler_mean=scaler.mean,
                scaler_scale=scaler.scale,
                extra={"best_val_aupr": best_aupr, "seed": args.seed},
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[early_stop] no improvement for {args.patience} epochs.")
                break

    # load best for final test metrics (if available)
    te_res = evaluate(model, dl_te, device)

    # write outputs
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dropped_invalid_smiles": int(dropped),
            "n_train": int(len(ds_tr)),
            "n_val": int(len(ds_va)),
            "n_test": int(len(ds_te)),
            "best_val_aupr": float(best_aupr),
            "test_loss": float(te_res.get("loss", float("nan"))),
            "test_auroc": float(te_res.get("auroc", float("nan"))),
            "test_aupr": float(te_res.get("aupr", float("nan"))),
            "hparams": hp.__dict__,
        }, f, indent=2, ensure_ascii=False)

    # curves
    if "roc_curve" in te_res:
        with open(os.path.join(args.out_dir, "roc_curve.json"), "w", encoding="utf-8") as f:
            json.dump(te_res["roc_curve"], f, indent=2, ensure_ascii=False)
    if "pr_curve" in te_res:
        with open(os.path.join(args.out_dir, "pr_curve.json"), "w", encoding="utf-8") as f:
            json.dump(te_res["pr_curve"], f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "split_indices.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_indices": [int(i) for i in idx_tr],
            "val_indices": [int(i) for i in idx_va],
            "test_indices": [int(i) for i in idx_te],
            "train_smiles": smiles_tr,
            "val_smiles": smiles_va,
            "test_smiles": smiles_te,
        }, f, indent=2, ensure_ascii=False)

    print(f"[done] checkpoint: {args.out_ckpt}")
    print(f"[done] outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
