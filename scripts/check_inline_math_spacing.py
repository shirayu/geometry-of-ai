#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def is_escaped(s: str, idx: int) -> bool:
    backslashes = 0
    k = idx - 1
    while k >= 0 and s[k] == "\\":
        backslashes += 1
        k -= 1
    return (backslashes % 2) == 1


def process_line(line: str):
    out = []
    issues = []
    i = 0
    in_code_span = False
    code_span_ticks = 0

    while i < len(line):
        ch = line[i]

        if ch == "`":
            n = 1
            while i + n < len(line) and line[i + n] == "`":
                n += 1
            if not in_code_span:
                in_code_span = True
                code_span_ticks = n
            elif n == code_span_ticks:
                in_code_span = False
                code_span_ticks = 0
            out.append(line[i : i + n])
            i += n
            continue

        if in_code_span:
            out.append(ch)
            i += 1
            continue

        if ch == "$" and not is_escaped(line, i):
            if i + 1 < len(line) and line[i + 1] == "$":
                end = line.find("$$", i + 2)
                if end == -1:
                    out.append(line[i:])
                    return "".join(out), issues
                out.append(line[i : end + 2])
                i = end + 2
                continue

            j = i + 1
            while j < len(line):
                if line[j] == "$" and not is_escaped(line, j):
                    if j + 1 < len(line) and line[j + 1] == "$":
                        j += 1
                        continue
                    break
                j += 1

            if j >= len(line):
                out.append(ch)
                i += 1
                continue

            content = line[i + 1 : j]
            stripped = content.strip()
            if content != stripped:
                issues.append((i + 1, "space-inside"))

            if i > 0 and not line[i - 1].isspace():
                issues.append((i + 1, "missing-space-before"))
                last = out[-1][-1] if out else ""
                if last and not last.isspace():
                    out.append(" ")

            out.append("$" + stripped + "$")

            if j + 1 < len(line) and not line[j + 1].isspace():
                issues.append((j + 1, "missing-space-after"))
                out.append(" ")

            i = j + 1
            continue

        out.append(ch)
        i += 1

    return "".join(out), issues


def iter_files(paths):
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from path.rglob("*.md")
        elif path.is_file():
            yield path


def process_file(path: Path, fix: bool):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    new_lines = []
    problems = []
    in_fence = False
    fence_marker = ""

    for idx, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            new_lines.append(line)
            continue

        if in_fence:
            new_lines.append(line)
            continue

        new_line, issues = process_line(line)
        if issues:
            for col, kind in issues:
                problems.append((path, idx, col, kind))
        new_lines.append(new_line if fix else line)

    if fix and new_lines != lines:
        path.write_text("".join(new_lines), encoding="utf-8")

    return problems


def main():
    parser = argparse.ArgumentParser(
        description="Check spacing around inline math delimiters ($...$)."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["series", "exercise"],
        help="Files or directories to scan (default: series)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes in-place.",
    )
    args = parser.parse_args()

    files = sorted({p for p in iter_files(args.paths)})
    if not files:
        print("No markdown files found.")
        return 0

    all_problems = []
    for path in files:
        all_problems.extend(process_file(path, args.fix))

    if all_problems:
        for path, line, col, kind in all_problems:
            print(f"{path}:{line}:{col}: {kind}")
        if not args.fix:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
