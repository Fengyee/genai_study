#!/usr/bin/env python3
"""Check repository Markdown links and math formatting without fetching URLs.

Handles inline/image links, reference definitions, headings and HTML id anchors.
This is a conservative check for this repository's Markdown subset, not a full
CommonMark parser. Code fences and inline code are excluded from link scanning.
Math checks enforce this repository's GitHub delimiter and transport rules; they do
not validate TeX, browser typesetting, or every possible Markdown construction.
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
    "docs/chapters/07-image-flow-lab.md",
    "docs/chapters/08-image-guidance-lab.md",
    "docs/chapters/09-pretrained-image-lab.md",
    "docs/chapters/10-pretrained-video-lab.md",
)
EXCLUDED = {".git", ".venv", ".venv-pretrained", "artifacts", "node_modules", "__pycache__", ".pytest_cache"}


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


def math_block_issues(text: str) -> list[tuple[int, str]]:
    """Check math-fence placement and line endings used by this repository.

    GitHub can leave display math unrendered when it shares a details section
    with inline math. We use short inline paragraphs or an external math block
    there. A TeX row separator at a physical line ending can also gain an extra
    backslash during GitHub transport, so the next formula stays on that line.
    """
    tokens = re.compile(
        r"(?P<fence>^ {0,3}(?P<marker>`{3,}|~{3,})(?P<info>[^\n]*))"
        r"|(?P<code>(?<!`)(?P<ticks>`+)(?!`)[\s\S]*?(?<!`)(?P=ticks)(?!`))"
        r"|(?P<comment><!--[\s\S]*?(?:-->|\Z))"
        r"|(?P<tag></?[A-Za-z][A-Za-z0-9:-]*(?=[\s/>])(?:[^>\"']|\"[^\"]*\"|'[^']*')*>)"
        r"|(?P<escape>\\.)",
        re.MULTILINE,
    )
    depth = 0
    position = 0
    issues = []
    while match := tokens.search(text, position):
        position = match.end()
        if match.group("fence") is not None:
            marker = match.group("marker")
            language = match.group("info").strip().split()
            if depth and language and language[0] == "math":
                line = text.count("\n", 0, match.start()) + 1
                issues.append((line, "math inside details: use a short standalone $`TeX`$ paragraph or move the math block outside details"))
            closing = re.compile(
                rf"^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*(?:\n|$)",
                re.MULTILINE,
            ).search(text, position)
            if language and language[0] == "math":
                body_start = position + int(text[position:position + 1] == "\n")
                body_end = closing.start() if closing else len(text)
                first_line = text.count("\n", 0, body_start) + 1
                for offset, line in enumerate(text[body_start:body_end].splitlines()):
                    if line.rstrip().endswith("\\\\"):
                        issues.append((first_line + offset, r"math block: keep TeX \\ and the following formula on the same physical line"))
            position = closing.end() if closing else len(text)
        elif match.group("tag") is not None:
            tag = match.group("tag")
            if re.match(r"</?details(?=[\s/>])", tag, re.IGNORECASE):
                if tag.startswith("</"):
                    depth = max(0, depth - 1)
                else:
                    depth += 1
    return issues


def math_format_issues(text: str) -> list[tuple[int, str]]:
    """Check math delimiters and the repository's GitHub transport rules.

    Explicit inline math, code spans, HTML tags/attributes and comments are
    blanked before scanning. Fenced blocks are already removed by ``prose``.
    Unpaired single dollars are left alone, including an isolated currency
    symbol. This is a format guard for the repository's Markdown subset.
    """
    ignored = re.compile(
        r"\$`[^`\n]+`\$"
        r"|(?<!`)(?P<ticks>`+)(?!`)[\s\S]*?(?<!`)(?P=ticks)(?!`)"
        r"|<!--[\s\S]*?(?:-->|\Z)"
        r"|</?[A-Za-z][A-Za-z0-9:-]*(?=[\s/>])(?:[^>\"']|\"[^\"]*\"|'[^']*')*>"
        r"|\\."
    )
    outside_fences = prose(text)
    issues = math_block_issues(text)
    table_lines = set()
    for match in ignored.finditer(outside_fences):
        token = match.group()
        if not token.startswith("$`") or r"\|" not in token[2:-2]:
            continue
        start = outside_fences.rfind("\n", 0, match.start()) + 1
        if outside_fences[start:match.start()].lstrip().startswith("|"):
            table_lines.add(outside_fences.count("\n", 0, match.start()) + 1)
    issues.extend((line, r"table math: use \Vert instead of \|; Markdown tables consume the backslash")
                  for line in sorted(table_lines))
    visible = ignored.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), outside_fences)
    in_display = False
    for number, line in enumerate(visible.splitlines(), 1):
        # Consume escapes in pairs: an escaped backslash leaves a subsequent
        # dollar active, while a backslash directly escaping the dollar does not.
        dollars = []
        index = 0
        while index < len(line):
            if line[index] == "\\":
                index += 2
                continue
            if line[index] == "$":
                dollars.append(index)
            index += 1
        doubles = []
        singles = []
        index = 0
        while index < len(dollars):
            if index + 1 < len(dollars) and dollars[index + 1] == dollars[index] + 1:
                doubles.append(dollars[index])
                index += 2
            else:
                singles.append(dollars[index])
                index += 1
        if in_display:
            if len(doubles) % 2:
                in_display = False
            continue
        if doubles:
            issues.append((number, "display math: replace $$ delimiters with a fenced ```math block"))
            in_display = len(doubles) % 2 == 1
        elif len(singles) >= 2:
            issues.append((number, "inline math: use explicit $`TeX`$ delimiters"))
    return sorted(issues, key=lambda issue: issue[0])


def check(root: Path, require_paths: bool = True) -> tuple[list[str], int]:
    errors = []
    if require_paths:
        errors.extend(f"missing required file: {name}" for name in REQUIRED if not (root / name).is_file())
    files = sorted(p for p in root.rglob("*.md") if not any(part in EXCLUDED for part in p.relative_to(root).parts))
    anchor_cache: dict[Path, set[str]] = {}
    count = 0
    for source in files:
        text = source.read_text(encoding="utf-8")
        errors.extend(f"{source.relative_to(root)}:{line}: {message}"
                      for line, message in math_format_issues(text))
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
    parser.add_argument("--no-required-paths", action="store_true", help="Skip the required-file list (for a small fixture or partial checkout).")
    args = parser.parse_args()
    errors, count = check(args.root.resolve(), not args.no_required_paths)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Markdown: checked {count} local links; {len(errors)} errors")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
