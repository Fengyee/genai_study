"""Exercise the installed reader CLI, including its artifact contract."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from genai_study.checkpoint import load_checkpoint
from genai_study.toy import make_schedule, sample_ddpm, sample_flow


@pytest.mark.parametrize("method", ["ddpm", "flow"])
def test_cli_artifacts_and_repeatability(tmp_path: Path, method: str):
    # Four updates check wiring and repeatability, not distribution quality.
    results = []
    for run in ("first", "repeat"):
        output = tmp_path / run
        subprocess.run([sys.executable, "-m", "genai_study.experiment", "--method", method,
                        "--steps", "4", "--samples", "64", "--batch-size", "32",
                        "--output", str(output)], check=True, capture_output=True, text=True,
                       timeout=60, env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")})
        metrics = json.loads((output / "metrics.json").read_text())
        assert metrics["method"] == method
        assert metrics["checkpoint"] == "checkpoint.pt"
        loaded, checkpoint = load_checkpoint(output / "checkpoint.pt")
        assert checkpoint["method"] == method
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(7 + 3)
        replayed = (sample_ddpm(loaded, 64, make_schedule(), generator) if method == "ddpm"
                    else sample_flow(loaded, 64, 100, generator))
        assert np.isfinite(metrics["final_eval_loss"])
        assert metrics["diagnostics"]["energy_sample_count"] == 64
        assert metrics["baseline_diagnostics"]["energy_sample_count"] == 64
        assert metrics["baseline_diagnostics"]["fraction_within_3sigma"] > 0.9
        assert metrics["data_theory"]["own_center_3sigma_mass_2d"] == pytest.approx(0.98889100346)
        assert (output / "samples.png").stat().st_size > 1000
        assert (output / "loss.png").stat().st_size > 1000
        with np.load(output / "samples.npz") as artifact:
            assert artifact["generated"].shape == (64, 2)
            np.testing.assert_array_equal(replayed.numpy(), artifact["generated"])
            assert np.isfinite(artifact["generated"]).all()
            results.append((metrics, artifact["generated"].copy()))
    assert results[0][0]["final_eval_loss"] == results[1][0]["final_eval_loss"]
    assert results[0][0]["baseline_diagnostics"] == results[1][0]["baseline_diagnostics"]
    np.testing.assert_array_equal(results[0][1], results[1][1])
    existing = tmp_path / "repeat"
    refused = subprocess.run([sys.executable, "-m", "genai_study.experiment", "--method", method,
                              "--output", str(existing)], capture_output=True, text=True, timeout=30)
    assert refused.returncode == 2
    assert "nonempty" in refused.stderr
