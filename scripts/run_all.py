#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One-command paper pipeline:
1) Train model from CSV  -> checkpoint (.pt)
2) Evaluate + save metrics json (from training script)
3) Generate TOP-K molecule/atom attention figures
4) Write a single 'paper_run_summary.json' that points to all outputs

Example (Windows CMD):
python scripts/run_all.py ^
  --data_csv data/hERGAT_final_dataset.csv ^
  --out_dir outputs/paper_run1 ^
  --smiles_col SMILES ^
  --label_col Class ^
  --device cuda:0 ^
  --top_k 20

Notes:
- This script calls scripts/train.py then scripts/make_topk_figures.py using the SAME python executable.
- No serving dependencies.
"""
from __future__ import annotations

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("[cmd]", " ".join(cmd))
    r = subprocess.run(cmd, check=False, cwd=str(cwd), env=env)
    if r.returncode != 0:
        raise SystemExit(f"Command failed with code {r.returncode}: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--out_dir", default="outputs/paper_run")
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--label_col", default="Class")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--train_size", type=float, default=0.8)
    ap.add_argument("--val_size", type=float, default=0.1)

    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--top_from", choices=["test", "all"], default="test")
    ap.add_argument("--save_svg", action="store_true")

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = project_root / "scripts"

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "ckpt"
    fig_dir = out_dir / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(ckpt_dir / "hERGAT_best.pt")

    env = os.environ.copy()
    prev_py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) + ((os.pathsep + prev_py_path) if prev_py_path else "")

    # 1) train
    run([
        sys.executable, str(scripts_dir / "train.py"),
        "--csv", args.data_csv,
        "--smiles_col", args.smiles_col,
        "--label_col", args.label_col,
        "--out_ckpt", ckpt_path,
        "--out_dir", str(out_dir),
        "--device", args.device,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--train_size", str(args.train_size),
        "--val_size", str(args.val_size),
    ], cwd=project_root, env=env)

    metrics_json = out_dir / "metrics.json"
    split_json = out_dir / "split_indices.json"
    if not metrics_json.exists():
        print("[warn] metrics.json not found at", metrics_json)
    if not split_json.exists():
        print("[warn] split_indices.json not found at", split_json)

    # 2) top-k figures
    run([
        sys.executable, str(scripts_dir / "make_topk_figures.py"),
        "--csv", args.data_csv,
        "--ckpt", ckpt_path,
        "--out_dir", str(fig_dir),
        "--smiles_col", args.smiles_col,
        "--label_col", args.label_col,
        "--device", args.device,
        "--top_k", str(args.top_k),
        "--top_from", args.top_from,
        "--split_indices", str(split_json) if split_json.exists() else "",
    ] + (["--save_svg"] if args.save_svg else []), cwd=project_root, env=env)

    topk_summary = fig_dir / "topk_figures_summary.json"

    # 3) one run summary
    summary = {
        "python_executable": sys.executable,
        "data_csv": os.path.abspath(args.data_csv),
        "out_dir": str(out_dir.resolve()),
        "checkpoint": os.path.abspath(ckpt_path),
        "metrics_json": str(metrics_json.resolve()) if metrics_json.exists() else "",
        "split_indices_json": str(split_json.resolve()) if split_json.exists() else "",
        "topk_figures_summary_json": str(topk_summary.resolve()) if topk_summary.exists() else "",
    }
    out_summary = out_dir / "paper_run_summary.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()
