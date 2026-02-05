# AGENTS.md

このリポジトリで作業する AI/開発者向けの最小ガイドです。

## 主要ディレクトリ

- `series/`: 講義本文
- `exercise/`: 各回の演習
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

## PR 前チェック

1. `pnpm format`
    - markdown中のpythonコードを`snippets/`に反映させるには先に `pnpm snippets:apply`
    - markdownlint, ruffなどで`.md`, `.py`を整形する
2. `pnpm snippets:apply`
    - markdown中のpythonコードは上書きされるので注意
3. `pnpm lint`
    - snippetsの同期が取れていないとエラー
4. 差分に意図しない `snippets/` 変更がないか確認
