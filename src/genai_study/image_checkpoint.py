"""Versioned inference weights for the 16x16 conditional convolutional flow."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .checkpoint import load_payload, plain_metadata, save_payload

if TYPE_CHECKING:
    from .image_flow import ConditionalConvFlow


def architecture(width: int) -> dict:
    return {"name": "ConditionalConvFlow-v1", "width": width,
            "image_shape": [1, 16, 16], "num_classes": 2}


def save_image_checkpoint(path: Path, model: ConditionalConvFlow, metadata: dict) -> None:
    """Save final inference parameters, architecture and plain training metadata."""
    save_payload(path, {"schema_version": 1, "method": "image_flow",
                        "architecture": architecture(model.enc1[0].out_channels),
                        "model_state_dict": model.state_dict(),
                        "metadata": plain_metadata(metadata)})


def load_image_checkpoint(path: Path) -> tuple[ConditionalConvFlow, dict]:
    from .image_flow import ConditionalConvFlow

    payload = load_payload(path)
    if payload.get("method") != "image_flow":
        raise ValueError("Image solver requires an image_flow checkpoint")
    spec = payload.get("architecture")
    if not isinstance(spec, dict):
        raise ValueError("Unsupported image checkpoint architecture")
    width = spec.get("width")
    if type(width) is not int or width < 4 or width % 4:
        raise ValueError("Image checkpoint width must be a positive multiple of 4")
    if spec != architecture(width) or type(spec.get("num_classes")) is not int or any(type(n) is not int for n in spec.get("image_shape", [])):
        raise ValueError("Unsupported image checkpoint architecture")
    model = ConditionalConvFlow(width=width)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload
