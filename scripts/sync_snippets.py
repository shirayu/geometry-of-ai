#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {"node_modules", ".git", "tmp"}
PY_SNIPPET_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.py$")
DEFAULT_TARGET_PATHS = ["series"]


@dataclass(frozen=True)
class SnippetBlock:
    name: str
    open_line_no: int
    code_start_idx: int
    code_end_idx: int


def iter_markdown_files(paths: list[str]) -> list[Path]:
    files: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".md":
            files.add(path)
            continue
        if not path.is_dir():
            continue
        for md in path.rglob("*.md"):
            if any(part in SKIP_DIRS for part in md.parts):
                continue
            if md.is_file():
                files.add(md)
    return sorted(files)


def is_closing_fence(line: str, marker_char: str, marker_len: int) -> bool:
    stripped = line.lstrip()
    i = 0
    while i < len(stripped) and stripped[i] == marker_char:
        i += 1
    if i < marker_len:
        return False
    return stripped[i:].strip() == ""


def parse_snippet_blocks(lines: list[str]) -> list[SnippetBlock]:
    blocks: list[SnippetBlock] = []
    open_re = re.compile(r"^[ \t]*(`{3,}|~{3,})([^\n]*)$")

    in_fence = False
    marker_char = ""
    marker_len = 0
    current_name: str | None = None
    code_start_idx = -1
    open_line_no = -1

    for idx, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n")
        if not in_fence:
            m = open_re.match(line)
            if not m:
                continue
            marker = m.group(1)
            info = m.group(2).strip()
            first_token = info.split()[0] if info else ""
            current_name = first_token if PY_SNIPPET_NAME_RE.match(first_token) else None
            in_fence = True
            marker_char = marker[0]
            marker_len = len(marker)
            code_start_idx = idx + 1
            open_line_no = idx + 1
            continue

        if is_closing_fence(line, marker_char, marker_len):
            if current_name is not None:
                blocks.append(
                    SnippetBlock(
                        name=current_name,
                        open_line_no=open_line_no,
                        code_start_idx=code_start_idx,
                        code_end_idx=idx,
                    )
                )
            in_fence = False
            marker_char = ""
            marker_len = 0
            current_name = None
            code_start_idx = -1
            open_line_no = -1

    return blocks


def snippet_scope(path: Path) -> str | None:
    if "series" in path.parts:
        return "series"
    return None


def scoped_snippets_dir(snippets_root: Path, path: Path) -> Path:
    scope = snippet_scope(path)
    return snippets_root / scope if scope is not None else snippets_root


def extract_snippets(files: list[Path], snippets_root: Path) -> int:
    seen_by_dir: dict[Path, dict[str, str]] = {}
    has_error = False

    for path in files:
        snippets_dir = scoped_snippets_dir(snippets_root, path)
        seen = seen_by_dir.setdefault(snippets_dir, {})
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        blocks = parse_snippet_blocks(lines)
        for block in blocks:
            content = "".join(lines[block.code_start_idx : block.code_end_idx])
            if block.name in seen and seen[block.name] != content:
                print(
                    f"{path}:{block.open_line_no}: conflicting snippet content for {block.name}",
                    file=sys.stderr,
                )
                has_error = True
                continue
            seen[block.name] = content

    if has_error:
        return 1

    total = 0
    removed = 0
    for snippets_dir, seen in seen_by_dir.items():
        snippets_dir.mkdir(parents=True, exist_ok=True)

        # Remove stale snippet files that are no longer referenced.
        keep_names = set(seen.keys())
        for existing in snippets_dir.glob("*.py"):
            if existing.name not in keep_names:
                existing.unlink()
                removed += 1

        for name, content in seen.items():
            (snippets_dir / name).write_text(content, encoding="utf-8")
            total += 1

    print(f"Wrote {total} snippet files under {snippets_root} (removed {removed} stale .py files)")
    return 0


def apply_snippets(files: list[Path], snippets_root: Path) -> int:
    missing_errors: list[str] = []
    targets: list[tuple[Path, Path, list[str], list[SnippetBlock]]] = []

    for path in files:
        snippets_dir = scoped_snippets_dir(snippets_root, path)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        blocks = parse_snippet_blocks(lines)
        if not blocks:
            continue
        targets.append((path, snippets_dir, lines, blocks))

        for block in blocks:
            snippet_path = snippets_dir / block.name
            if not snippet_path.is_file():
                missing_errors.append(f"{path}:{block.open_line_no}: snippet file not found: {snippet_path}")

    if missing_errors:
        for msg in missing_errors:
            print(msg, file=sys.stderr)
        return 1

    for path, snippets_dir, lines, blocks in targets:
        updated = list(lines)
        changed = False
        for block in reversed(blocks):
            snippet_path = snippets_dir / block.name
            snippet_text = snippet_path.read_text(encoding="utf-8")
            if snippet_text and not snippet_text.endswith("\n"):
                snippet_text += "\n"
            snippet_lines = snippet_text.splitlines(keepends=True)

            original_lines = updated[block.code_start_idx : block.code_end_idx]
            if original_lines != snippet_lines:
                updated[block.code_start_idx : block.code_end_idx] = snippet_lines
                changed = True

        if changed:
            path.write_text("".join(updated), encoding="utf-8")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync markdown fenced python snippets and files under snippets/.")
    parser.add_argument(
        "--snippets-dir",
        default="snippets",
        help="Root directory to read/write snippet files (default: snippets)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_extract = subparsers.add_parser("extract", help="Extract fenced snippets to files.")
    p_extract.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_TARGET_PATHS,
        help="Files or directories to scan (default: series)",
    )

    p_apply = subparsers.add_parser("apply", help="Replace fenced snippets from files.")
    p_apply.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_TARGET_PATHS,
        help="Files or directories to update (default: series)",
    )

    args = parser.parse_args()
    files = iter_markdown_files(args.paths)
    snippets_dir = Path(args.snippets_dir)

    if args.command == "extract":
        return extract_snippets(files, snippets_dir)
    if args.command == "apply":
        return apply_snippets(files, snippets_dir)
    return 2


if __name__ == "__main__":
    sys.exit(main())
