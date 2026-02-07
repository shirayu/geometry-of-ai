#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DETAILS_OPEN_RE = re.compile(r"<details(?:\s+[^>]*)?>", re.IGNORECASE)
DETAILS_CLOSE_RE = re.compile(r"</details>", re.IGNORECASE)
SUMMARY_RE = re.compile(r"^<summary>\s*コード例:\s*([^<]+?)\s*</summary>\s*$")
IMPL_NOTE_HEADING_RE = re.compile(r"^(#{1,6})\s+.*実装ノート")


def count_lines(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8"))


def iter_snippet_files(paths: list[str]) -> list[Path]:
    files: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".py":
            files.add(path)
            continue
        if path.is_dir():
            files.update(p for p in path.rglob("*.py") if p.is_file())
    return sorted(files)


def iter_markdown_files(paths: list[str]) -> list[Path]:
    files: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".md":
            files.add(path)
            continue
        if path.is_dir():
            files.update(p for p in path.rglob("*.md") if p.is_file())
    return sorted(files)


def collect_fence_locations(markdown_files: list[Path]) -> dict[str, list[tuple[Path, int, bool]]]:
    locations: dict[str, list[tuple[Path, int, bool]]] = {}
    for path in markdown_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        in_fence = False
        fence_marker = ""
        details_depth = 0

        for line_no, line in enumerate(lines, start=1):
            stripped = line.lstrip()

            if not in_fence:
                details_depth += len(DETAILS_OPEN_RE.findall(line))
                details_depth -= len(DETAILS_CLOSE_RE.findall(line))
                if details_depth < 0:
                    details_depth = 0

            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if not in_fence:
                    parts = stripped[3:].strip().split(maxsplit=1)
                    fence_name = parts[0] if parts else ""
                    if fence_name.endswith(".py"):
                        locations.setdefault(fence_name, []).append((path, line_no, details_depth > 0))
                    in_fence = True
                    fence_marker = marker
                elif marker == fence_marker:
                    in_fence = False
                    fence_marker = ""

    return locations


def markdown_scope_for_snippet(path: Path) -> list[str]:
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "snippets" and parts[1] in {"series", "exercise"}:
        return [parts[1]]
    return ["series", "exercise"]


def check_summary_filename_consistency(markdown_files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in markdown_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            match = SUMMARY_RE.match(line.strip())
            if not match:
                continue

            expected_name = match.group(1).strip()
            next_idx = idx + 1
            while next_idx < len(lines) and lines[next_idx].strip() == "":
                next_idx += 1

            if next_idx >= len(lines):
                errors.append(
                    f"ERROR {path}:{idx + 1}: summary expects `{expected_name}` but next code fence is missing",
                )
                continue

            stripped = lines[next_idx].lstrip()
            if not (stripped.startswith("```") or stripped.startswith("~~~")):
                errors.append(
                    f"ERROR {path}:{idx + 1}: summary expects `{expected_name}` but next line is not a code fence",
                )
                continue

            actual_name = stripped[3:].strip().split(maxsplit=1)
            actual_name = actual_name[0] if actual_name else ""
            if actual_name != expected_name:
                errors.append(
                    f"ERROR {path}:{idx + 1}: summary `{expected_name}` does not match fence `{actual_name}`",
                )
    return errors


def check_impl_note_details(series_files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in series_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        in_fence = False
        fence_marker = ""
        details_depth = 0
        in_impl_note = False
        impl_note_level = 0

        for idx, line in enumerate(lines, start=1):
            stripped = line.lstrip()

            if not in_fence:
                details_depth += len(DETAILS_OPEN_RE.findall(line))
                details_depth -= len(DETAILS_CLOSE_RE.findall(line))
                if details_depth < 0:
                    details_depth = 0

                match = IMPL_NOTE_HEADING_RE.match(stripped)
                if match:
                    in_impl_note = True
                    impl_note_level = len(match.group(1))
                elif stripped.startswith("#"):
                    level = len(stripped) - len(stripped.lstrip("#"))
                    if in_impl_note and level <= impl_note_level:
                        in_impl_note = False

            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if not in_fence:
                    if in_impl_note and details_depth == 0:
                        errors.append(
                            f"ERROR {path}:{idx}: code fence in 実装ノート is not wrapped by <details>",
                        )
                    in_fence = True
                    fence_marker = marker
                elif marker == fence_marker:
                    in_fence = False
                    fence_marker = ""

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check snippet line counts (error: >= 60 lines, info: >= 40 lines).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["snippets"],
        help="Files or directories to scan (default: snippets)",
    )
    parser.add_argument(
        "--error-threshold",
        type=int,
        default=60,
        help="Line-count threshold for error (default: 60)",
    )
    parser.add_argument(
        "--info-threshold",
        type=int,
        default=40,
        help="Line-count threshold for info (default: 40)",
    )
    args = parser.parse_args()

    if args.info_threshold > args.error_threshold:
        print("--info-threshold must be less than or equal to --error-threshold")
        return 2

    files = iter_snippet_files(args.paths)
    if not files:
        print("No snippet .py files found.")
        return 0

    has_error = False
    markdown_files = iter_markdown_files(["series", "exercise"])
    for error in check_summary_filename_consistency(markdown_files):
        print(error)
        has_error = True
    for error in check_impl_note_details(iter_markdown_files(["series"])):
        print(error)
        has_error = True

    fence_locations = collect_fence_locations(markdown_files)
    for path in files:
        lines = count_lines(path)
        if lines < args.info_threshold:
            continue

        fence_name = path.name
        locations = fence_locations.get(fence_name, [])

        if lines >= args.error_threshold:
            scoped_markdown_dirs = set(markdown_scope_for_snippet(path))
            scoped_locations = [loc for loc in locations if loc[0].parts and loc[0].parts[0] in scoped_markdown_dirs]
            if not scoped_locations:
                print(
                    f"ERROR {path}: {lines} lines (>= {args.error_threshold}), "
                    f"fence `{fence_name}` not found in markdown",
                )
                has_error = True
                continue

            unwrapped = [loc for loc in scoped_locations if not loc[2]]
            if unwrapped:
                joined = ", ".join(f"{p}:{ln}" for p, ln, _ in unwrapped)
                print(
                    f"ERROR {path}: {lines} lines (>= {args.error_threshold}), not wrapped by <details> at {joined}",
                )
                has_error = True
        else:
            print(f"INFO  {path}: {lines} lines (>= {args.info_threshold})")

    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
