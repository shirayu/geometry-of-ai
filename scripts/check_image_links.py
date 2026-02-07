#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

RE_INLINE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
RE_REF = re.compile(r"!\[[^\]]*\]\[([^\]]*)\]")
RE_REF_DEF = re.compile(r"^\s*\[([^\]]+)\]:\s*(\S+)")

REMOTE_PREFIXES = (
    "http://",
    "https://",
    "data:",
    "mailto:",
    "tel:",
    "//",
)


def git_ls_files_md() -> list[Path]:
    try:
        out = subprocess.check_output(["git", "ls-files", "*.md"], text=True)
    except Exception:
        return sorted(p for p in Path(".").rglob("*.md") if "node_modules" not in p.parts)
    files = [Path(p) for p in out.splitlines() if p.strip()]
    return files


def normalize_url(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    if s.startswith("<") and ">" in s:
        s = s[1 : s.index(">")].strip()
    # Split title if present (spaces allowed only inside <> per spec)
    if " " in s and not s.startswith("<"):
        s = s.split()[0]
    return s


def is_remote(url: str) -> bool:
    return url.startswith(REMOTE_PREFIXES)


def strip_query_fragment(url: str) -> str:
    base = url.split("#", 1)[0]
    base = base.split("?", 1)[0]
    return base


def collect_ref_defs(lines: Iterable[str]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for line in lines:
        m = RE_REF_DEF.match(line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        url = normalize_url(m.group(2))
        refs[key] = url
    return refs


def resolve_path(md_path: Path, url: str, repo_root: Path) -> Path:
    if url.startswith("/"):
        return repo_root / url.lstrip("/")
    return (md_path.parent / url).resolve()


def check_file(md_path: Path, repo_root: Path) -> list[tuple[int, str, Path | None]]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    ref_defs = collect_ref_defs(lines)
    errors: list[tuple[int, str, Path | None]] = []

    for idx, line in enumerate(lines, start=1):
        for m in RE_INLINE.finditer(line):
            raw = m.group(1)
            url = normalize_url(raw)
            if url == "":
                errors.append((idx, raw, None))
                continue
            if is_remote(url) or url.startswith("#"):
                continue
            base = strip_query_fragment(url)
            if base == "":
                errors.append((idx, url, None))
                continue
            resolved = resolve_path(md_path, base, repo_root)
            if not resolved.is_file():
                errors.append((idx, url, resolved))

        for m in RE_REF.finditer(line):
            ref_id = m.group(1).strip()
            if ref_id == "":
                # Empty means use alt text; leave to markdown, skip
                continue
            key = ref_id.lower()
            url = ref_defs.get(key, "")
            if url == "":
                errors.append((idx, f"[{ref_id}]", None))
                continue
            if is_remote(url) or url.startswith("#"):
                continue
            base = strip_query_fragment(url)
            resolved = resolve_path(md_path, base, repo_root)
            if not resolved.is_file():
                errors.append((idx, url, resolved))

    return errors


def main() -> int:
    repo_root = Path(".").resolve()
    md_files = git_ls_files_md()
    all_errors: list[tuple[Path, int, str, Path | None]] = []

    for md_path in md_files:
        errs = check_file(md_path, repo_root)
        for line_no, url, resolved in errs:
            all_errors.append((md_path, line_no, url, resolved))

    if not all_errors:
        return 0

    for md_path, line_no, url, resolved in all_errors:
        if resolved is None:
            msg = f"{md_path}:{line_no}: Broken image link: {url}"
        else:
            msg = f"{md_path}:{line_no}: Broken image link: {url} -> {resolved}"
        print(msg, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
