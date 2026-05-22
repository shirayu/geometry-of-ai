"""site/series/ の生成済み Markdown を静的検査するスクリプト。

検査内容:
- 画像パス切れ: src="/series/..." が site/public/series/ に存在するか
- 内部リンク切れ: [text](/series/foo) が site/series/foo.md に存在するか
- Python fence 変換漏れ: ```filename.py が残っていないか（変換後は ```python [...] になるはず）
"""

import re
import sys
from pathlib import Path

SITE_SERIES_DIR = Path(__file__).parent.parent / "site" / "series"
SITE_PUBLIC_SERIES_DIR = Path(__file__).parent.parent / "site" / "public" / "series"

IMG_SRC_RE = re.compile(r'src="(/series/[^"]+)"')
LINK_RE = re.compile(r"\[([^\]]*)\]\((/series/[^)#\s]+?)(?:#[^)]*)?\)")
RAW_PYTHON_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_]+\.py$", re.MULTILINE)


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")

    for m in IMG_SRC_RE.finditer(text):
        img_path = m.group(1)  # /series/foo.svg
        filename = img_path.removeprefix("/series/")
        if not (SITE_PUBLIC_SERIES_DIR / filename).exists():
            errors.append(f"  画像切れ: {img_path}")

    for m in LINK_RE.finditer(text):
        link = m.group(2)  # /series/foo or /series/foo.md
        slug = link.removeprefix("/series/")
        slug = slug.removesuffix(".md")
        if not (SITE_SERIES_DIR / f"{slug}.md").exists():
            errors.append(f"  リンク切れ: {link}")

    if RAW_PYTHON_FENCE_RE.search(text):
        errors.append("  Python fence 変換漏れ: ```*.py が残っています")

    return errors


def main() -> None:
    if not SITE_SERIES_DIR.exists():
        print("ERROR: site/series/ が存在しません。先に task site:generate を実行してください。")
        sys.exit(1)

    md_files = sorted(SITE_SERIES_DIR.glob("*.md"))
    if not md_files:
        print("ERROR: site/series/ に .md ファイルがありません。")
        sys.exit(1)

    total_errors = 0
    for md in md_files:
        errors = check_file(md)
        if errors:
            print(f"{md.name}:")
            for e in errors:
                print(e)
            total_errors += len(errors)

    if total_errors:
        print(f"\n{total_errors} 件のエラーが見つかりました。")
        sys.exit(1)
    else:
        print(f"OK: {len(md_files)} ファイルを検査しました。エラーなし。")


if __name__ == "__main__":
    main()
