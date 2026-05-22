# AGENTS.md

このリポジトリで作業する AI/開発者向けの最小ガイドです。

## 主要ディレクトリ

- `series/`: 講義本文
- `snippets/`: 抽出・適用対象のコード片
- `scripts/`:  文中の Python コード断片の同期元, lint, format対象

## セットアップ

1. Python 環境: `uv sync`
2. Node 環境: `pnpm install --frozen-lockfile`

## よく使うコマンド

- 総合 lint: `pnpm lint`
- 総合 format: `pnpm format`
- snippets 抽出: `pnpm snippets:extract`
- snippets 反映: `pnpm snippets:apply`

## 本文記述注意点

- Code fence
    - pythonコードは `04_stable_softmax.py` のような命名でfence。`snippets/`と同期。
    - txtは `txt` でfence
- 長いPythonコードの表示方針（`series/*`）
    - `60`行以上: `<details><summary>...</summary>` で囲む（文書内の一貫性を優先）
    - `40`〜`59`行: そのまま表示を基本（必要なら同一文書内の見た目に合わせて折りたたみ可）
    - `40`行未満: 原則そのまま表示（同一文書内の一貫性が必要なら折りたたみ可）
    - 行数判定は `pnpm run lint:snippet_lines` を使用（`lint:snippets` で同期が取れていることが前提）

## PR 前チェック

1. `pnpm format`
    - markdown中のpythonコードを`snippets/`に反映させるには先に `pnpm snippets:apply`
    - markdownlint, ruffなどで`.md`, `.py`を整形する
2. `pnpm snippets:apply`
    - markdown中のpythonコードは上書きされるので注意
3. `pnpm lint`
    - snippetsの同期が取れていないとエラー
4. 差分に意図しない `snippets/` 変更がないか確認
