"""Counterexamples distinguish temporal correctness from frame marginals."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from genai_study.video_diagnostics import centroids, make_videos, video_metrics


def test_shuffle_preserves_frames_but_breaks_temporal_task():
    videos, permutation = make_videos()
    reference, shuffled = videos["reference"], videos["shuffled"]
    np.testing.assert_array_equal(np.sort(permutation), np.arange(16))
    np.testing.assert_array_equal(shuffled[np.argsort(permutation)], reference)
    np.testing.assert_allclose(shuffled.mean(0), reference.mean(0), atol=1e-15, rtol=1e-14)
    measured = video_metrics(shuffled, reference, 8)
    assert measured["symmetric_frame_set_mse"] == 0
    assert measured["time_mean_image_mse"] < 1e-30
    assert measured["ordered_mse_to_reference"] > 0.001
    assert measured["centroid_rmse_px"] > 5
    assert measured["wrong_direction_steps"] > 0


def test_static_low_variation_is_not_success():
    videos, _ = make_videos()
    measured = video_metrics(videos["static_first_frame"], videos["reference"], 8)
    assert measured["mean_centroid_speed_px_per_second"] == 0
    assert measured["endpoint_displacement_px"] == 0
    assert measured["wrong_direction_steps"] == 0  # also fails to detect lack of motion
    assert measured["centroid_rmse_px"] > 10
    assert measured["ordered_mse_to_reference"] > 0.001


def test_reference_motion_and_fps_units():
    videos, _ = make_videos()
    reference = videos["reference"]
    slow = video_metrics(reference, reference, 8)
    fast = video_metrics(reference, reference, 16)
    assert slow["ordered_mse_to_reference"] == 0
    assert slow["centroid_rmse_px"] == 0
    assert slow["wrong_direction_steps"] == 0
    assert slow["endpoint_displacement_px"] == pytest.approx(31, abs=0.001)
    assert slow["mean_centroid_speed_px_per_second"] == pytest.approx(31 / 15 * 8, abs=0.001)
    assert fast["mean_centroid_speed_px_per_second"] == 2 * slow["mean_centroid_speed_px_per_second"]
    assert fast["endpoint_displacement_px"] == slow["endpoint_displacement_px"]
    np.testing.assert_allclose(centroids(reference)[:, 1], 15.5, atol=1e-12)


def test_smallest_video_still_shuffles():
    videos, permutation = make_videos(frames=2)
    assert not np.array_equal(permutation, np.arange(2))
    assert video_metrics(videos["shuffled"], videos["reference"], 8)["ordered_mse_to_reference"] > 0


def test_video_cli_artifacts(tmp_path):
    output = tmp_path / "video"
    subprocess.run([sys.executable, "-m", "genai_study.video_diagnostics", "--output", str(output)],
                   check=True, capture_output=True, text=True, timeout=60,
                   env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")})
    result = json.loads((output / "metrics.json").read_text())
    assert result["config"]["shape"] == [16, 32, 48]
    assert result["config"]["last_frame_time_seconds"] == 15 / 8
    assert result["config"]["nominal_playback_duration_seconds"] == 2
    assert result["metrics"]["shuffled"]["symmetric_frame_set_mse"] == 0
    with np.load(output / "frames.npz") as saved:
        np.testing.assert_array_equal(saved["shuffled"], saved["reference"][saved["shuffled_source_indices"]])
    assert (output / "contact_sheet.png").stat().st_size > 1000


@pytest.mark.parametrize("fps", ["1e308", "1e-320", "1e-308"])
def test_unrepresentable_fps_fails_before_creating_artifacts(tmp_path, fps):
    # At 1e-308 and two frames the last timestamp fits, but playback duration
    # overflows. Both derived units must be checked in addition to finite fps.
    output = tmp_path / "video"
    result = subprocess.run(
        [sys.executable, "-m", "genai_study.video_diagnostics", "--frames", "2",
         "--fps", fps, "--output", str(output)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "--fps must keep" in result.stderr
    assert "finite" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_invalid_fps_preserves_existing_artifacts_with_overwrite(tmp_path):
    output = tmp_path / "video"
    output.mkdir()
    saved = output / "frames.npz"
    np.savez_compressed(saved, timestamps_seconds=[0.0, 0.125])
    before = saved.read_bytes()
    result = subprocess.run(
        [sys.executable, "-m", "genai_study.video_diagnostics", "--frames", "2",
         "--fps", "1e308", "--overwrite", "--output", str(output)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert saved.read_bytes() == before
    assert sorted(p.name for p in output.iterdir()) == ["frames.npz"]
