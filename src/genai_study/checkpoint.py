"""Schema-1 inference checkpoints: finite float32 tensors and plain metadata."""
from collections.abc import Mapping
import math
from pathlib import Path

import torch

from .toy import TimeMLP


def plain_metadata(metadata: dict) -> dict:
    """Copy JSON values, normalizing string subclasses and rejecting nonfinite data."""
    def convert(value):
        if value is None or type(value) in (bool, int):
            return value
        if isinstance(value, str):
            return str(value)
        if type(value) is float and math.isfinite(value):
            return value
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return {str(key): convert(item) for key, item in value.items()}
        raise ValueError("Checkpoint metadata requires finite JSON values and string keys")

    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    return convert(metadata)


def validate_state_dict(state: object) -> None:
    """Require schema-1 numeric state before PyTorch can silently cast tensors."""
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint state must be a nonempty mapping of tensor names")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Checkpoint state requires string names and tensor values")
        if tensor.dtype != torch.float32:
            raise ValueError(f"Checkpoint state {name!r} must have dtype float32")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Checkpoint state {name!r} must contain only finite values")


def save_payload(path: Path, payload: dict) -> None:
    """Replace a completed tensor checkpoint atomically within its directory."""
    validate_state_dict(payload.get("model_state_dict"))
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_payload(path: Path) -> dict:
    """Apply the shared safe-loader version gate before any checkpoint is read."""
    # CVE-2025-32434 affects weights_only=True before PyTorch 2.6.
    major_minor = tuple(int(part) for part in str(torch.__version__).split(".")[:2])
    if major_minor < (2, 6):
        raise RuntimeError("Checkpoint loading requires torch>=2.6 (CVE-2025-32434); rerun setup")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint schema")
    payload["metadata"] = plain_metadata(payload.get("metadata"))
    validate_state_dict(payload.get("model_state_dict"))
    return payload


def save_checkpoint(path: Path, model: TimeMLP, method: str, metadata: dict) -> None:
    if method not in ("ddpm", "flow"):
        raise ValueError("Unsupported checkpoint method")
    payload = {
        "schema_version": 1,
        "method": method,
        "architecture": {"name": "TimeMLP-v1", "width": model.net[0].out_features},
        "model_state_dict": model.state_dict(),
        "metadata": plain_metadata(metadata),
    }
    save_payload(path, payload)


def load_checkpoint(path: Path) -> tuple[TimeMLP, dict]:
    payload = load_payload(path)
    if payload.get("method") not in ("ddpm", "flow"):
        raise ValueError("Unsupported checkpoint method")
    architecture = payload.get("architecture", {})
    if not isinstance(architecture, dict) or architecture.get("name") != "TimeMLP-v1":
        raise ValueError("Unsupported checkpoint architecture")
    width = architecture.get("width")
    if type(width) is not int or width < 1:
        raise ValueError("Invalid checkpoint width")
    with torch.device("cpu"):
        model = TimeMLP(width=width).to(dtype=torch.float32)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload
