# AGENTS.md

このリポジトリで作業する AI/開発者向けの最小ガイドです。

執筆・編集規約は [STYLE_GUIDE.md](./STYLE_GUIDE.md) を参照。

## 主要ディレクトリ

- `site/series/`: 講義本文
- `site/public/series/`: 画像ファイル
- `snippets/`: 抽出・適用対象のコード片
- `scripts/`: lint, format スクリプト群

## セットアップ

1. Python 環境: `uv sync`
2. Node 環境: `pnpm install --frozen-lockfile`

## よく使うコマンド

- 総合 lint: `task lint`
- 総合 format: `task format`
- snippets 抽出: `task snippets:extract`
- snippets 反映: `task snippets:apply`
- 開発サーバー起動: `task site:dev` (オプション: `HOST=0.0.0.0 PORT=5174`)
- サイトビルド: `task site:build`

## 本文記述注意点

- Code fence
    - pythonコードは `04_stable_softmax.py` のような命名でfence。`snippets/`と同期。
    - txtは `txt` でfence
- 長いPythonコードの表示方針（`series/*`）
    - `60`行以上: `<details><summary>...</summary>` で囲む（文書内の一貫性を優先）
    - `40`〜`59`行: そのまま表示を基本（必要なら同一文書内の見た目に合わせて折りたたみ可）
    - `40`行未満: 原則そのまま表示（同一文書内の一貫性が必要なら折りたたみ可）
    - 行数判定は `task lint:snippet_lines` を使用（`lint:snippets` で同期が取れていることが前提）

## PR 前チェック

1. `task snippets:apply`
    - markdown中のpythonコードは上書きされるので注意
2. `task format`
    - markdownlint, ruffなどで`.md`, `.py`を整形する
3. `task lint`
    - snippetsの同期が取れていないとエラー
4. 差分に意図しない `snippets/` 変更がないか確認
