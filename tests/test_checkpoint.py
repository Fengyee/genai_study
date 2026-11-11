"""Frozen checkpoint round trips and paired solver studies through the real CLI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from genai_study.checkpoint import load_checkpoint, save_checkpoint
from genai_study.toy import TimeMLP, integrate_flow


def test_checkpoint_roundtrip_exact_and_safe_load(tmp_path, monkeypatch):
    torch.manual_seed(31)
    model = TimeMLP()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, "flow", {"seed": 31, "torch": torch.__version__})
    original_load = torch.load
    calls = []

    def observe_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observe_load)
    loaded, payload = load_checkpoint(path)
    assert calls[0]["weights_only"] is True
    assert calls[0]["map_location"] == "cpu"
    assert type(payload["metadata"]["torch"]) is str
    initial = torch.randn(32, 2, generator=torch.Generator().manual_seed(8))
    untouched = initial.clone()
    for steps in (1, 10):
        torch.testing.assert_close(integrate_flow(loaded, initial, steps),
                                   integrate_flow(model, initial, steps), rtol=0, atol=0)
    torch.testing.assert_close(initial, untouched, rtol=0, atol=0)


def test_solver_study_frozen_paired_samples(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(15)
    model = TimeMLP()
    checkpoint = tmp_path / "flow.pt"
    save_checkpoint(checkpoint, model, "flow", {"training_steps": 0})
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = tmp_path / "study"
    subprocess.run([sys.executable, "-m", "genai_study.solver_study", "--checkpoint", str(checkpoint),
                    "--steps", "10", "2", "10", "--samples", "64", "--seed", "123",
                    "--output", str(output)], check=True, capture_output=True, text=True,
                   timeout=60, env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")})
    result = json.loads((output / "metrics.json").read_text())
    assert result["checkpoint"]["sha256"] == original_hash
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_hash
    assert result["config"]["steps"] == [2, 10]
    assert [run["nfe"] for run in result["runs"]] == [2, 10]
    assert result["runs"][0]["previous_steps"] is None
    assert result["runs"][1]["previous_steps"] == 2
    with np.load(output / "samples.npz") as samples:
        expected_initial = torch.randn(64, 2, generator=torch.Generator().manual_seed(123))
        np.testing.assert_array_equal(samples["initial"], expected_initial.numpy())
        for steps in (2, 10):
            expected = integrate_flow(model, expected_initial, steps)
            np.testing.assert_array_equal(samples[f"steps_{steps}"], expected.numpy())
        measured = np.sqrt(np.mean(np.sum((samples["steps_10"] - samples["steps_2"]) ** 2, axis=1)))
        assert result["runs"][1]["paired_rms_l2_difference_from_previous"] == pytest.approx(float(measured))
    assert (output / "samples.png").stat().st_size > 1000


def test_solver_rejects_ddpm_checkpoint(tmp_path):
    checkpoint = tmp_path / "ddpm.pt"
    save_checkpoint(checkpoint, TimeMLP(), "ddpm", {})
    output = tmp_path / "study"
    process = subprocess.run([sys.executable, "-m", "genai_study.solver_study",
                              "--checkpoint", str(checkpoint), "--output", str(output)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 2
    assert "requires a flow checkpoint" in process.stderr
    assert not output.exists()


def test_checkpoint_rejects_unknown_schema(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"schema_version": 99}, path)
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(path)


def test_checkpoint_load_rejects_known_vulnerable_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.5.1+cpu")
    # Fail before reading even a nonexistent file, so the unsafe loader cannot run.
    with pytest.raises(RuntimeError, match="CVE-2025-32434"):
        load_checkpoint(tmp_path / "does-not-exist.pt")
