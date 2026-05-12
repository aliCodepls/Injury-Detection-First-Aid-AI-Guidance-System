"""
CNN inference for FIRSTSIGHT using best_wound_model.pth (EfficientNet-B0 + metadata MLP + classifier).

Weights: set WOUND_MODEL_PATH, or place best_wound_model.pth next to this file.
At inference, missing metadata is filled with zeros (13 floats), matching deployment without side-channel features.

Vision / triage text comes from woundwatch.py (separate GGUF-backed server), not this module.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

SCRIPT_DIR = Path(__file__).resolve().parent

METADATA_DIM = 13

_DEFAULT_CLASS_MAPPING: dict[str, int] = {
    "Burns": 0,
    "Cuts_lacerations": 1,
    "Abrasions": 2,
    "Insect_bites": 3,
    "Bruises": 4,
}

_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class EfficientNetMetadataWoundClassifier(nn.Module):
    """Architecture matching best_wound_model.pth (efficientnet.features + metadata_mlp + classifier)."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        b0 = models.efficientnet_b0(weights=None)
        self.features = b0.features
        self.avgpool = b0.avgpool

        self.metadata_mlp = nn.Sequential(
            nn.Linear(METADATA_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
        )

        fused = 1280 + 16
        self.classifier = nn.Sequential(
            nn.Linear(fused, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        z = self.features(x)
        z = self.avgpool(z)
        z = torch.flatten(z, 1)
        if meta is None:
            meta = torch.zeros(z.size(0), METADATA_DIM, device=z.device, dtype=z.dtype)
        m = self.metadata_mlp(meta)
        z = torch.cat([z, m], dim=1)
        return self.classifier(z)


def _remap_checkpoint_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Checkpoint uses efficientnet.features.*; module uses features.*."""
    out: dict[str, Any] = {}
    prefix = "efficientnet.features."
    for k, v in raw.items():
        if k.startswith(prefix):
            out["features." + k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _load_class_mapping() -> dict[str, int]:
    path = SCRIPT_DIR / "class_mapping.json"
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): int(v) for k, v in data.items()}
    return dict(_DEFAULT_CLASS_MAPPING)


def _resolve_weights_path() -> Path:
    env = os.environ.get("WOUND_MODEL_PATH")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"WOUND_MODEL_PATH is not a file: {p}")
    p = SCRIPT_DIR / "best_wound_model.pth"
    if p.is_file():
        return p
    raise FileNotFoundError(
        f"Missing best_wound_model.pth in {SCRIPT_DIR} "
        "(or set WOUND_MODEL_PATH to your .pth file)."
    )


def _torch_load_checkpoint(path: Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=map_location)


_model: EfficientNetMetadataWoundClassifier | None = None
_device: torch.device | None = None
_class_mapping: dict[str, int] | None = None


def _get_model_and_device() -> tuple[EfficientNetMetadataWoundClassifier, torch.device, dict[str, int]]:
    global _model, _device, _class_mapping
    if _model is not None and _device is not None and _class_mapping is not None:
        return _model, _device, _class_mapping

    mapping = _load_class_mapping()
    ncls = len(set(mapping.values()))
    if ncls < 2:
        raise ValueError("class_mapping must define at least 2 classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = _resolve_weights_path()
    raw = _torch_load_checkpoint(weights_path, map_location=device)
    state = _remap_checkpoint_keys(raw)

    model = EfficientNetMetadataWoundClassifier(num_classes=ncls).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load {weights_path} into EfficientNetMetadataWoundClassifier. "
            "Expected best_wound_model.pth (efficientnet.features + metadata_mlp + classifier)."
        ) from e
    model.eval()

    _model, _device, _class_mapping = model, device, mapping
    return model, device, mapping


def predict_wound_class_from_bytes(image_bytes: bytes) -> dict[str, Any]:
    """Returns injury_type, confidence, class_probabilities for final_pipeline."""
    model, device, class_mapping = _get_model_and_device()
    inv_idx = {v: k for k, v in class_mapping.items()}

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _VAL_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor, meta=None)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
        pred_idx_int = int(pred_idx.item())
        injury_type = inv_idx.get(pred_idx_int, str(pred_idx_int))

    class_probabilities = {inv_idx[i]: float(probs[i].item()) for i in range(len(probs))}

    return {
        "injury_type": injury_type,
        "confidence": float(conf.item()),
        "class_probabilities": class_probabilities,
    }
