"""Inference checkpoints: tensors plus JSON-compatible metadata, no pickled models."""
import json
from pathlib import Path

import torch

from .toy import TimeMLP


def save_checkpoint(path: Path, model: TimeMLP, method: str, metadata: dict) -> None:
    if method not in ("ddpm", "flow"):
        raise ValueError("Unsupported checkpoint method")
    # Round-trip strips string subclasses and rejects arbitrary Python objects.
    plain_metadata = json.loads(json.dumps(metadata, allow_nan=False))
    payload = {
        "schema_version": 1,
        "method": method,
        "architecture": {"name": "TimeMLP-v1", "width": model.net[0].out_features},
        "model_state_dict": model.state_dict(),
        "metadata": plain_metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> tuple[TimeMLP, dict]:
    # CVE-2025-32434 affects weights_only=True before PyTorch 2.6.
    major_minor = tuple(int(part) for part in str(torch.__version__).split(".")[:2])
    if major_minor < (2, 6):
        raise RuntimeError("Checkpoint loading requires torch>=2.6 (CVE-2025-32434); rerun setup")
    # Explicit even on versions where weights_only is not the default.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    if payload.get("method") not in ("ddpm", "flow"):
        raise ValueError("Unsupported checkpoint method")
    architecture = payload.get("architecture", {})
    if not isinstance(architecture, dict) or architecture.get("name") != "TimeMLP-v1":
        raise ValueError("Unsupported checkpoint architecture")
    width = architecture.get("width")
    if not isinstance(width, int) or width < 1:
        raise ValueError("Invalid checkpoint width")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    json.dumps(payload["metadata"], allow_nan=False)
    model = TimeMLP(width=width)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload
