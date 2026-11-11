"""Derived seed boundaries follow real PyTorch and fail before CLI side effects."""
import argparse
import importlib
import sys
from unittest.mock import Mock

import pytest
import torch

from genai_study.cli import seed_with_offset


@pytest.mark.parametrize("max_offset", [2, 5, 6])
def test_valid_seed_boundaries_preserve_real_torch_streams(max_offset):
    parse = seed_with_offset(max_offset)
    for seed in [-(2**63), -max_offset, -1, 0, 7, 123, 2**64 - 1 - max_offset]:
        parsed = parse(str(seed))
        assert parsed == seed  # Keep the signed value until each stream is seeded.
        for offset in range(max_offset + 1):
            actual = torch.Generator().manual_seed(parsed + offset)
            expected = torch.Generator().manual_seed(seed + offset)
            assert actual.initial_seed() == expected.initial_seed()
            torch.testing.assert_close(torch.rand(4, generator=actual),
                                       torch.rand(4, generator=expected), rtol=0, atol=0)


@pytest.mark.parametrize("max_offset", [2, 5, 6])
def test_seed_bounds_reject_actual_torch_overflow(max_offset):
    parse = seed_with_offset(max_offset)
    lower_invalid = -(2**63) - 1
    upper_invalid = 2**64 - max_offset
    # The upper base is legal by itself; its highest derived stream is not.
    torch.Generator().manual_seed(upper_invalid)
    for value in [lower_invalid, upper_invalid + max_offset]:
        with pytest.raises(RuntimeError, match="Overflow"):
            torch.Generator().manual_seed(value)
    for value in [str(lower_invalid), str(upper_invalid), "1.5"]:
        with pytest.raises(argparse.ArgumentTypeError, match="reserving"):
            parse(value)


CLI_CASES = [
    ("experiment", 5, 7, ["--method", "flow"], None),
    ("solver_study", 2, 123, [], "load_checkpoint"),
    ("image_flow", 6, 7, [], None),
    ("image_solver_study", 2, 123, [], "load_image_checkpoint"),
    ("image_guidance_study", 2, 123, [], "load_image_checkpoint"),
]


def cli_arguments(name, extra, loader, tmp_path):
    arguments = [name, *extra, "--output", str(tmp_path / "output")]
    if loader:
        arguments += ["--checkpoint", str(tmp_path / "missing-checkpoint.pt")]
    return arguments


@pytest.mark.parametrize("name,max_offset,default,extra,loader", CLI_CASES)
@pytest.mark.parametrize("invalid", ["lower", "upper", "noninteger"])
def test_cli_rejects_seed_before_output_or_checkpoint(
        name, max_offset, default, extra, loader, invalid, tmp_path, monkeypatch, capsys):
    module = importlib.import_module(f"genai_study.{name}")
    seed = {"lower": str(-(2**63) - 1), "upper": str(2**64 - max_offset),
            "noninteger": "1.5"}[invalid]
    arguments = cli_arguments(name, extra, loader, tmp_path) + [f"--seed={seed}"]
    monkeypatch.setattr(sys, "argv", arguments)
    checkpoint_reader = Mock(side_effect=AssertionError("checkpoint read before seed rejection"))
    if loader:
        monkeypatch.setattr(module, loader, checkpoint_reader)
    initializer = Mock(side_effect=AssertionError("RNG initialized before seed rejection"))
    monkeypatch.setattr(torch, "manual_seed", initializer)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "--seed" in stderr
    assert f"[{-(2**63)}, {2**64 - 1 - max_offset}]" in stderr
    assert f"reserving +{max_offset}" in stderr
    assert not (tmp_path / "output").exists()
    checkpoint_reader.assert_not_called()
    initializer.assert_not_called()


@pytest.mark.parametrize("name,max_offset,default,extra,loader", CLI_CASES)
def test_cli_real_parser_preserves_endpoints_negative_and_default_seeds(
        name, max_offset, default, extra, loader, tmp_path, monkeypatch):
    module = importlib.import_module(f"genai_study.{name}")
    real_parse = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        """Stop just after the genuine parser, before any experiment work."""

    def inspect_parsed_seed(parser, *args, **kwargs):
        parsed = real_parse(parser, *args, **kwargs)
        assert parsed.seed == expected
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", inspect_parsed_seed)
    for seed in [None, -(2**63), -1, 0, 2**64 - 1 - max_offset]:
        expected = default if seed is None else seed
        arguments = cli_arguments(name, extra, loader, tmp_path)
        if seed is not None:
            arguments.append(f"--seed={seed}")
        monkeypatch.setattr(sys, "argv", arguments)
        with pytest.raises(Parsed):
            module.main()
    assert not (tmp_path / "output").exists()
