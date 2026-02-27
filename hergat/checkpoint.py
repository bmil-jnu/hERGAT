# hergat/checkpoint.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
import numpy as np

from .model import HergatHParams, ExtendedFingerprint
from .preprocess import ScalerParams


@dataclass
class CheckpointBundle:
    hp: HergatHParams
    scaler: ScalerParams
    label_map: Dict[str, str]
    morgan: Dict[str, int]
    state_dict: Dict[str, Any]


def load_checkpoint(path: str, device: torch.device) -> Tuple[ExtendedFingerprint, CheckpointBundle]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must be a dict with key 'model_state_dict'.")

    hp_d = ckpt.get("hparams", {})
    required = [
        "radius","T","input_feature_dim","input_bond_dim","fingerprint_dim","output_units_num","p_dropout",
        "physicochemical_feature_dim","physicochemical_feature_dim_1","physicochemical_feature_dim_2","final1_fc1","final1_fc2"
    ]
    missing = [k for k in required if k not in hp_d]
    if missing:
        raise ValueError(f"Missing hparams in checkpoint: {missing}")

    hp = HergatHParams(**{k: hp_d[k] for k in required})
    scaler = ScalerParams.from_checkpoint(ckpt["scaler"])
    label_map = ckpt.get("label_map", {"0":"non_blocker","1":"blocker"})
    morgan = ckpt.get("morgan", {"radius": 3, "nBits": 1024})

    model = ExtendedFingerprint(hp, device=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    bundle = CheckpointBundle(hp=hp, scaler=scaler, label_map=label_map, morgan=morgan, state_dict=ckpt["model_state_dict"])
    return model, bundle


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    hp: HergatHParams,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    label_map: Dict[str, str] | None = None,
    morgan: Dict[str, int] | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    ckpt = {
        "format_version": 1,
        "model_name": "hERGAT",
        "model_class": "ExtendedFingerprint",
        "hparams": hp.__dict__,
        "model_state_dict": model.state_dict(),
        "scaler": {"mean": scaler_mean.tolist(), "scale": scaler_scale.tolist()},
        "label_map": label_map or {"0":"non_blocker","1":"blocker"},
        "morgan": morgan or {"radius": 3, "nBits": 1024},
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
