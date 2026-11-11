"""Shared argument validation for the CPU teaching CLIs."""
import argparse
from collections.abc import Callable


TORCH_SEED_MIN = -(2**63)
TORCH_SEED_MAX = 2**64 - 1


def seed_with_offset(max_offset: int) -> Callable[[str], int]:
    """Accept base seeds whose nonnegative derived offsets all fit PyTorch.

    Keep the signed input unchanged: a negative seed plus an offset must be
    computed before PyTorch remaps that individual stream to an unsigned seed.
    """
    maximum = TORCH_SEED_MAX - max_offset
    message = (f"must be an integer in [{TORCH_SEED_MIN}, {maximum}] "
               f"(reserving +{max_offset} for derived seeds)")

    def parse_seed(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(message) from None
        if not TORCH_SEED_MIN <= number <= maximum:
            raise argparse.ArgumentTypeError(message)
        return number

    return parse_seed
