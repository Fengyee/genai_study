#!/usr/bin/env python3
"""Check repository Markdown links without fetching external URLs.

Handles inline/image links, reference definitions, headings and HTML id anchors.
This is a conservative check for this repository's Markdown subset, not a full
CommonMark parser. Code fences and inline code are excluded from link scanning.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

REQUIRED = (
    "README.md", "CONTRIBUTING.md", "LICENSE", "docs/learning-path.md",
    "docs/notation.md", "docs/chapters/00-foundations.md",
    "docs/chapters/01-generative-models.md", "docs/chapters/02-diffusion.md",
    "docs/chapters/03-flow-matching.md", "docs/chapters/04-image-video-roadmap.md",
    "docs/chapters/05-evaluation.md", "docs/chapters/06-video-temporal-lab.md",
)
EXCLUDED = {".git", ".venv", "artifacts", "node_modules", "__pycache__", ".pytest_cache"}


def prose(text: str) -> str:
    """Blank fenced code while preserving line numbers."""
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if match:
            token = match.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            lines.append("")
        else:
            lines.append(line if fence is None else "")
    return "\n".join(lines)


def anchors(text: str) -> set[str]:
    text = prose(text)
    result = set(re.findall(r'\bid=["\']([^"\']+)["\']', text))
    seen: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        title = html.unescape(re.sub(r"<[^>]*>", "", match.group(1))).lower()
        title = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", title)
        slug = re.sub(r"[^\w\- ]", "", title).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        result.add(slug if count == 0 else f"{slug}-{count}")
    return result


def destinations(text: str):
    for number, line in enumerate(prose(text).splitlines(), 1):
        line = re.sub(r"(`+).*?\1", "", line)
        # Inline links and images; optional title after the destination.
        for match in re.finditer(r"!?\[[^\]]*\]\(\s*(<[^>]*>|[^\s)]+)(?:\s+[\"'][^\"']*[\"'])?\s*\)", line):
            yield number, match.group(1).strip("<>")
        reference = re.match(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)", line)
        if reference:
            yield number, reference.group(1).strip("<>")


def check(root: Path, require_paths: bool = True) -> tuple[list[str], int]:
    errors = []
    if require_paths:
        errors.extend(f"missing required file: {name}" for name in REQUIRED if not (root / name).is_file())
    files = sorted(p for p in root.rglob("*.md") if not any(part in EXCLUDED for part in p.relative_to(root).parts))
    anchor_cache: dict[Path, set[str]] = {}
    count = 0
    for source in files:
        text = source.read_text(encoding="utf-8")
        for line, destination in destinations(text):
            url = urlsplit(destination)
            if url.scheme or url.netloc:
                continue
            count += 1
            target = (source.parent / unquote(url.path)).resolve() if url.path else source
            label = f"{source.relative_to(root)}:{line}: {destination}"
            if not target.is_relative_to(root):
                errors.append(f"{label}: link escapes repository")
            elif not target.exists():
                errors.append(f"{label}: target does not exist")
            elif url.fragment and target.suffix.lower() == ".md":
                if target not in anchor_cache:
                    anchor_cache[target] = anchors(target.read_text(encoding="utf-8"))
                if unquote(url.fragment) not in anchor_cache[target]:
                    errors.append(f"{label}: heading/id anchor does not exist")
    return errors, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--no-required-paths", action="store_true", help="Only check links (for a small fixture or partial checkout).")
    args = parser.parse_args()
    errors, count = check(args.root.resolve(), not args.no_required_paths)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Markdown: checked {count} local links; {len(errors)} errors")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
