#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict hERG blocker probability for a single SMILES and save attention visualization images.

Example:
python scripts/predict_with_attention.py \
  --ckpt outputs/run1/ckpt/hERGAT_best.pt \
  --smiles "CCO" \
  --device cuda:0 \
  --image_root ./images \
  --image_host http://localhost/images \
  --save_json
"""
from __future__ import annotations

import os
import json
import argparse
import torch

from hergat.infer import predict_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to hERGAT checkpoint (.pt)")
    ap.add_argument("--smiles", required=True, help="Single SMILES")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image_root", default="./images")
    ap.add_argument("--image_host", default="")
    ap.add_argument("--prefix", default="predict")
    ap.add_argument("--save_json", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.image_root, exist_ok=True)
    device = torch.device(args.device)

    pred = predict_one(
        smiles=args.smiles,
        checkpoint_path=args.ckpt,
        device=device,
        return_attention=True,
        save_images=True,
        image_prefix=args.prefix,
        image_root=args.image_root,
        image_host=args.image_host,
    )

    out = {
        "smiles": args.smiles,
        "canon_smiles": pred.canon_smiles,
        "herg_prediction": pred.pred_label,
        "prob_blocker": float(pred.prob_blocker),
        "prob_non_blocker": float(pred.prob_non_blocker),
        "images": pred.images,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.save_json:
        out_path = os.path.join(args.image_root, f"{args.prefix}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
