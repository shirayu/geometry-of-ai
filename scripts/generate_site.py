"""series/ から site/ 向け Markdown を生成するスクリプト。

変換内容:
- 独自 Python fence (```filename.py) → ```python [filename.py]
- <details><summary>...</summary> → :::details ...
- <img src="file.svg"> の相対パス → /series/file.svg
- 画像ファイルを site/public/series/ にコピー
"""

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
        # <details> の開始
        if line.rstrip() == "<details>":
            # 直後の <summary>...</summary> を探す
            if i + 1 < len(lines):
                summary_match = re.match(r"<summary>(.*?)</summary>", lines[i + 1].rstrip())
                if summary_match:
                    label = summary_match.group(1)
                    out.append(f":::details {label}\n")
                    i += 2  # <details> と <summary> をスキップ
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


def convert(text: str) -> str:
    text = convert_python_fences(text)
    text = convert_details_blocks(text)
    text = convert_image_paths(text)
    return text


def generate() -> None:
    SITE_SERIES_DIR.mkdir(parents=True, exist_ok=True)
    SITE_PUBLIC_SERIES_DIR.mkdir(parents=True, exist_ok=True)

    # 生成先をクリア（手編集防止）
    for f in SITE_SERIES_DIR.iterdir():
        f.unlink()
    for f in SITE_PUBLIC_SERIES_DIR.iterdir():
        f.unlink()

    for src in sorted(SERIES_DIR.iterdir()):
        if src.suffix == ".md":
            dst = SITE_SERIES_DIR / src.name
            dst.write_text(convert(src.read_text(encoding="utf-8")), encoding="utf-8")
        elif src.suffix in IMAGE_EXTENSIONS:
            shutil.copy2(src, SITE_PUBLIC_SERIES_DIR / src.name)

    print(
        f"Generated {len(list(SITE_SERIES_DIR.iterdir()))} md files, "
        f"{len(list(SITE_PUBLIC_SERIES_DIR.iterdir()))} image files."
    )


if __name__ == "__main__":
    generate()
