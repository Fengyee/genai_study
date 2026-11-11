"""Independent GroupNorm arithmetic; standard library only, no tensors/models."""

from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import hashlib
import json


EPS = Fraction(1, 100_000)


def stats(values):
    values = tuple(map(Fraction, values))
    mean = sum(values) / len(values)
    centered_variance = sum((value - mean) ** 2 for value in values) / len(values)
    second_moment_variance = sum(value**2 for value in values) / len(values) - mean**2
    assert centered_variance == second_moment_variance
    return mean, centered_variance


def decimal_fraction(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def norm(value, moments):
    mean, variance = moments
    with localcontext() as context:
        context.prec = 70
        return decimal_fraction(Fraction(value) - mean) / decimal_fraction(variance + EPS).sqrt()


def report_stats(values):
    mean, variance = stats(values)
    return {"n": len(values), "mean": str(mean), "population_variance": str(variance)}


def main():
    # Shape [N,C,T,H,W] = [1,2,8,1,1]; contiguous channels, one group.
    channels = [list(range(0, 8)), list(range(8, 16))]
    full = [value for channel in channels for value in channel]
    chunk4 = [value for channel in channels for value in channel[:4]]
    halo5 = [value for channel in channels for value in channel[:5]]
    windows = {"full_8_frames": full, "first_4_frames": chunk4, "first_5_frames": halo5}
    expected = {
        "full_8_frames": (Fraction(15, 2), Fraction(85, 4)),
        "first_4_frames": (Fraction(11, 2), Fraction(69, 4)),
        "first_5_frames": (Fraction(6), Fraction(18)),
    }
    for name, values in windows.items():
        assert stats(values) == expected[name]

    # Frame numbering in prose is one-based; this is t=1, c=0, value=1.
    outputs = {name: norm(1, stats(values)) for name, values in windows.items()}
    assert len(set(outputs.values())) == 3

    changed_channels = [channel.copy() for channel in channels]
    changed_channels[1][7] = 115
    changed_full = [value for channel in changed_channels for value in channel]
    changed_halo5 = [value for channel in changed_channels for value in channel[:5]]
    assert halo5 == changed_halo5
    assert stats(changed_full) == (Fraction(55, 4), Fraction(11215, 16))
    original_local_outputs = [norm(value, stats(full)) for value in halo5]
    changed_local_outputs = [norm(value, stats(changed_full)) for value in changed_halo5]
    assert all(a != b for a, b in zip(original_local_outputs, changed_local_outputs))
    assert all(norm(value, stats(halo5)) == norm(value, stats(changed_halo5)) for value in halo5)

    # C=4,G=2: group zero is channels 0/1; group one is channels 2/3.
    group0 = list(range(16))
    group1 = list(range(100, 116))
    changed_group1 = group1.copy()
    changed_group1[-1] = 215
    before_group_moments = [stats(group0), stats(group1)]
    after_group_moments = [stats(group0), stats(changed_group1)]
    assert before_group_moments[0] == after_group_moments[0]
    assert before_group_moments[1] != after_group_moments[1]
    assert [norm(value, before_group_moments[0]) for value in group0] == [
        norm(value, after_group_moments[0]) for value in group0
    ]
    assert norm(101, before_group_moments[1]) != norm(101, after_group_moments[1])

    result = {
        "method": "Exact Fraction means and population variances, independently checked with centered squares and second moments; 70-digit Decimal square roots for outputs.",
        "configuration": {"shape_NCTHW": [1, 2, 8, 1, 1], "groups": 1, "affine": False, "epsilon": str(EPS), "teaching_simplifications": "C=2, G=1, affine=False; epsilon=1e-5 matches the parent review's VAE temporal block source check."},
        "statistics": {name: report_stats(values) for name, values in windows.items()},
        "channel_0_frame_2_value_1": {name: str(value) for name, value in outputs.items()},
        "unseen_context_change": {
            "change": "channel 1, frame 8: 15 -> 115 (one-based frame numbering)",
            "first_5_frames_inputs_identical": halo5 == changed_halo5,
            "changed_full_statistics": report_stats(changed_full),
            "all_10_first_5_frames_full_groupnorm_outputs_changed": all(a != b for a, b in zip(original_local_outputs, changed_local_outputs)),
            "channel_0_frame_2_changed_full_output": str(norm(1, stats(changed_full))),
            "first_5_frames_halo_groupnorm_outputs_unchanged": True,
        },
        "two_group_C4_check": {
            "shape_NCTHW": [1, 4, 8, 1, 1],
            "groups": 2,
            "group0_input": "channels 0/1 contain 0..15",
            "group1_input": "channels 2/3 contain 100..115",
            "change": "channel 3, frame 8: 115 -> 215",
            "group0_statistics_before_and_after": report_stats(group0),
            "group1_statistics_before": report_stats(group1),
            "group1_statistics_after": report_stats(changed_group1),
            "group0_outputs_identical": True,
            "group1_channel2_frame2_output_changed": True,
        },
        "scope": "Scalar GroupNorm counterexample only. Any deterministic reconstruction based solely on a fixed observed local window must return the same result for identical local inputs; these full-group outputs differ. This excludes universal exact local-only reconstruction when unobserved same-group values are unrestricted, not reconstruction with externally supplied global statistics, restricted inputs, or a halo covering the entire group. No claim about causal attribution in a real SVD full network.",
        "errors": [],
    }
    target = Path(__file__).with_name("results.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(target.read_text(), end="")
    for path in (Path(__file__), target):
        print(f"SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")


if __name__ == "__main__":
    main()
