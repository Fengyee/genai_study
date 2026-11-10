#!/usr/bin/env python3
"""Run the reviewed, inexpensive CPU examples embedded in the lessons.

The allowlist below and adjacent ``cpu-check`` markers identify each block.
Only Python blocks or a single recognized Python heredoc are supported; shell
commands, training examples, downloads and artifact-dependent labs are excluded.
This runner executes reviewed repository code with the current interpreter and
repository permissions. Each example runs in its own process.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


EXAMPLES = {
    "diffusion-flow-bridge": "docs/readings/diffusion-flow-bridge.md",
    "autoregressive-causality": "docs/chapters/01-generative-models.md",
    "coverage-counterexample": "docs/chapters/05-evaluation.md",
    "is-kid-counterexamples": "docs/chapters/05-evaluation.md",
    "generative-identities": "docs/generative-source-audit.md",
    "kontext-packing-time": "docs/readings/flux-kontext-2025.md",
    "video-vp-shapes": "docs/readings/video-training-svd-cogvideox.md",
    "restok-group-factorization": "docs/readings/restok-image-ar-2026.md",
    "ltx2-joint-factorization": "docs/readings/ltx2-video-audio-2026.md",
    "ltx2-token-shapes": "docs/readings/ltx2-video-audio-2026.md",
    "ltx2-sampling-grid": "docs/readings/ltx2-video-audio-2026.md",
}
MARKER = re.compile(r"^<!-- cpu-check: ([a-z][a-z0-9-]*) -->$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
HEREDOC = re.compile(r"(?:\.venv/bin/)?python - <<'PY'")


@dataclass
class Block:
    path: str
    marker_line: int
    source: str = ""
    error: str = ""


def extract_block(lines: list[str], marker_index: int, path: str) -> Block:
    block = Block(path, marker_index + 1)
    opening = marker_index + 1
    while opening < len(lines) and not lines[opening].strip():
        opening += 1
    if opening == len(lines) or lines[opening].strip() not in ("```python", "```bash"):
        block.error = "marker must be followed directly by a python or bash code fence"
        return block
    closing = opening + 1
    while closing < len(lines) and lines[closing].strip() != "```":
        closing += 1
    if closing == len(lines):
        block.error = "code fence is not closed"
        return block
    content = lines[opening + 1:closing]
    first_python_line = opening + 2
    if lines[opening].strip() == "```bash":
        if (len(content) < 3 or HEREDOC.fullmatch(content[0]) is None
                or content[-1] != "PY"):
            block.error = "bash block must contain only a recognized python - <<'PY' heredoc"
            return block
        content = content[1:-1]
        first_python_line += 1
    if not any(line.strip() for line in content):
        block.error = "example is empty"
        return block
    # Preserve document line numbers in Python tracebacks, including heredocs.
    block.source = "\n" * (first_python_line - 1) + "\n".join(content) + "\n"
    return block


def collect_blocks(root: Path) -> tuple[dict[str, list[Block]], list[str]]:
    found: dict[str, list[Block]] = {}
    errors = []
    for path in sorted((root / "docs").rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            errors.append(f"{relative}: {error}")
            continue
        fence = None
        for index, line in enumerate(lines):
            boundary = FENCE.fullmatch(line)
            if fence is not None:
                if (boundary and boundary[1][0] == fence[0]
                        and len(boundary[1]) >= len(fence) and not boundary[2].strip()):
                    fence = None
                continue
            if boundary:
                fence = boundary[1]
                continue
            marker = MARKER.fullmatch(line)
            if marker:
                found.setdefault(marker[1], []).append(extract_block(lines, index, relative))
            elif line.lstrip().startswith("<!-- cpu-check:"):
                errors.append(f"{relative}:{index + 1}: malformed cpu-check marker")
    return found, errors


def print_captured(output: str | bytes | None) -> None:
    if output:
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        print(output, end="" if output.endswith("\n") else "\n", flush=True)


def run_examples(root: Path, timeout_seconds: float) -> int:
    found, errors = collect_blocks(root)
    for name in sorted(found.keys() - EXAMPLES.keys()):
        locations = ", ".join(f"{b.path}:{b.marker_line}" for b in found[name])
        errors.append(f"unknown cpu-check id {name!r} at {locations}")
    for error in errors:
        print(f"FAIL markers: {error}", flush=True)
    failures, passed = len(errors), 0
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for name, expected_path in EXAMPLES.items():
        blocks = found.get(name, [])
        error = ""
        if not blocks:
            error = f"missing marker in {expected_path}"
        elif len(blocks) != 1:
            error = "duplicate markers at " + ", ".join(f"{b.path}:{b.marker_line}" for b in blocks)
        elif blocks[0].path != expected_path:
            error = f"marker belongs in {expected_path}, found {blocks[0].path}"
        elif blocks[0].error:
            error = f"{expected_path}:{blocks[0].marker_line}: {blocks[0].error}"
        if error:
            print(f"FAIL {name}: {error}", flush=True)
            failures += 1
            continue
        block = blocks[0]
        print(f"RUN {name} ({block.path}:{block.marker_line})", flush=True)
        payload = f"exec(compile({block.source!r}, {block.path!r}, 'exec'))"
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, "-u", "-c", payload], cwd=root, env=environment,
                capture_output=True, text=True, timeout=timeout_seconds, check=False,
            )
            print_captured(result.stdout)
            print_captured(result.stderr)
            status = result.returncode
            detail = f"exit {status}"
        except subprocess.TimeoutExpired as error:
            print_captured(error.stdout)
            print_captured(error.stderr)
            status, detail = 124, f"timeout after {timeout_seconds:g}s"
        except OSError as error:
            status, detail = 1, str(error)
        elapsed = time.perf_counter() - started
        if status == 0:
            passed += 1
            print(f"PASS {name} ({elapsed:.2f}s)", flush=True)
        else:
            failures += 1
            print(f"FAIL {name}: {detail} ({elapsed:.2f}s)", flush=True)
    print(f"Lesson examples: {passed}/{len(EXAMPLES)} passed; {failures} failures", flush=True)
    return int(failures != 0)


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be a finite positive number")
    return seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root; defaults to this script's parent repository.")
    parser.add_argument("--timeout-seconds", type=positive_seconds, default=30.0,
                        help="Per-example process timeout (default: 30 seconds).")
    args = parser.parse_args()
    return run_examples(args.root.resolve(), args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
