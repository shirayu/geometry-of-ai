# 開発者向けドキュメント

「情報幾何学とAI」講義サイトのセットアップ・ビルド・構成の説明です。
執筆・編集の規約は [../STYLE_GUIDE.md](../STYLE_GUIDE.md)、AI/開発者向けの最小ガイドは [../AGENTS.md](../AGENTS.md) を参照してください。

## セットアップ

ツール類のバージョンは [mise](https://mise.jdx.dev/) で管理しています。

```sh
mise install
task setup   # uv sync + pnpm install --frozen-lockfile
```

依存関係を最新化する場合は `task update` を使います。

## よく使うコマンド

| コマンド | 内容 |
| --- | --- |
| `task dev` | 開発サーバー起動（`HOST`/`PORT` 変数で上書き可） |
| `task build` | サイトビルド |
| `task preview` | ビルド結果のプレビュー（`HOST`/`PORT` 変数で上書き可） |
| `task format` | markdownlint・ruff などによる整形 |
| `task lint` | snippets・参考文献の同期などを含む総合チェック |
| `task snippets:extract` / `task snippets:apply` | `snippets/` と本文コードの抽出・反映 |
| `task references:apply` | `references.bib` / `site/series/refs/*.yml` から各章の参考文献セクションを再生成 |

`HOST`/`PORT` を指定する例（外部からアクセスできるように `0.0.0.0` を指定する場合など）:

```sh
task dev HOST=0.0.0.0 PORT=5174
task preview HOST=0.0.0.0 PORT=4174
```

## 構成

- `site/`: [VitePress](https://vitepress.dev/) 製のサイト本体（pnpm ワークスペース内の別パッケージ）
    - `site/series/`: 講義本文（Markdown）
    - `site/public/series/`: 章に紐づく画像ファイル
- `snippets/`: 本文コードの抽出元・反映先となるコード片。`task snippets:*` で本文と同期
- `references.bib`: 参考文献の書誌情報（唯一の正）。`site/series/refs/<chapter>.yml` と組み合わせて章末の参考文献セクションを生成
- `scripts/`: lint・format・sync 用の Python/Node スクリプト群（`taskfiles/` の各タスクから呼ばれる）
- `taskfiles/`: `task` コマンドのサブタスク定義（`format` / `lint` / `references` / `snippets` / `site`）
- `style/`: 用語集など機械検証用の設定（例: `style/terms.yaml`）

## PR前チェック

`AGENTS.md` の「PR 前チェック」を参照してください。要点は次の順序です。

1. `task snippets:apply`
2. `task references:apply`
3. `task format`
4. `task lint`
5. `snippets/` に意図しない差分がないか確認
