"""series/ から site/ 向け Markdown を生成するスクリプト。

変換内容:
- 独自 Python fence (```filename.py) → ```python [filename.py]
- <details><summary>...</summary> → :::details ...
- <img src="file.svg"> の相対パス → /series/file.svg
- 画像ファイルを site/public/series/ にコピー
- keywords.md / references.md: aside: false の frontmatter を付与

オプション:
  --site-series-dir <dir>  生成先の series ディレクトリを上書き（lint 用）
"""

import argparse
import re
import shutil
from pathlib import Path

SERIES_DIR = Path(__file__).parent.parent / "series"
SITE_SERIES_DIR = Path(__file__).parent.parent / "site" / "series"
SITE_PUBLIC_SERIES_DIR = Path(__file__).parent.parent / "site" / "public" / "series"

IMAGE_EXTENSIONS = {".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
PYTHON_FENCE_RE = re.compile(r"^```([a-zA-Z0-9_]+\.py)$", re.MULTILINE)
IMG_SRC_RE = re.compile(r'src="([^/][^"]*\.(svg|png|jpg|jpeg|gif|webp))"')


def convert_python_fences(text: str) -> str:
    return PYTHON_FENCE_RE.sub(lambda m: f"```python [{m.group(1)}]", text)


def convert_details_blocks(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip() == "<details>":
            if i + 1 < len(lines):
                summary_match = re.match(r"<summary>(.*?)</summary>", lines[i + 1].rstrip())
                if summary_match:
                    label = summary_match.group(1)
                    out.append(f":::details {label}\n")
                    i += 2
                    continue
            out.append(line)
        elif line.rstrip() == "</details>":
            out.append(":::\n")
        else:
            out.append(line)
        i += 1
    return "".join(out)


def convert_image_paths(text: str) -> str:
    return IMG_SRC_RE.sub(lambda m: f'src="/series/{m.group(1)}"', text)


WIDE_FRONTMATTER = "---\naside: false\n---\n\n"


def prepend_frontmatter(text: str, frontmatter: str) -> str:
    """既存の frontmatter がなければ先頭に追加する。"""
    if text.lstrip().startswith("---"):
        return text
    return frontmatter + text.lstrip()


def convert(text: str, *, filename: str = "") -> str:
    text = convert_python_fences(text)
    text = convert_details_blocks(text)
    text = convert_image_paths(text)

    if filename in ("keywords.md", "references.md"):
        text = prepend_frontmatter(text, WIDE_FRONTMATTER)

    return text


def generate(site_series_dir: Path = SITE_SERIES_DIR) -> None:
    site_series_dir.mkdir(parents=True, exist_ok=True)
    SITE_PUBLIC_SERIES_DIR.mkdir(parents=True, exist_ok=True)

    for f in site_series_dir.iterdir():
        f.unlink()
    # --site-series-dir が上書きされていない場合のみ画像出力先もクリア
    if site_series_dir == SITE_SERIES_DIR:
        for f in SITE_PUBLIC_SERIES_DIR.iterdir():
            f.unlink()

    for src in sorted(SERIES_DIR.iterdir()):
        if src.suffix == ".md":
            dst = site_series_dir / src.name
            dst.write_text(
                convert(src.read_text(encoding="utf-8"), filename=src.name),
                encoding="utf-8",
            )
        elif src.suffix in IMAGE_EXTENSIONS and site_series_dir == SITE_SERIES_DIR:
            shutil.copy2(src, SITE_PUBLIC_SERIES_DIR / src.name)

    n_images = len(list(SITE_PUBLIC_SERIES_DIR.iterdir())) if site_series_dir == SITE_SERIES_DIR else 0
    print(
        f"Generated {len(list(site_series_dir.iterdir()))} md files"
        + (f", {n_images} image files." if n_images else ".")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-series-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.site_series_dir:
        generate(args.site_series_dir)
    else:
        generate()
