# `check-all` ブランチ変更の妥当性レビュー

対象: `origin/main...HEAD` のうち `check-all/**` を除く差分
レビュー日: 2026-08-23
確認時コミット: `fcfd5cb`

## 総評

初回レビューでは、数理・実装上の修正必須2件、誤解を招く説明などの改善推奨11件、実験条件の任意改善1件を検出した。
これら14件を修正し、関連する本文、snippet、参考文献構成、クイズ参照を同期した。

修正後は総合lintとサイトビルドが成功している。
初回レビューで確認済みだった既存変更にも、新たな回帰は見つからなかった。
したがって、依存パッケージを要する講義用Pythonコードの未実行という制約を除き、対象変更は妥当と判断する。

## 指摘への対応結果

| # | 初回重要度 | 対象 | 対応結果 | コミット |
| --- | --- | --- | --- | --- |
| 1 | A | Appendix 6のRLCT | $K$ をKLダイバージェンスとして定義し、測度側の指数 $h_{\alpha j}$ とチャート最小化を含む一般式へ修正 | `ac793ec` |
| 2 | A | Appendix 1の量子化 | $q_{\max}=2^{b-1}-1$ の符号付き対称格子へ変更し、0を厳密に保存 | `ac793ec` |
| 3 | B | 第0回の統計モデル | 写像の向きを $\theta\mapsto p(\cdot;\theta)$ に直し、微分のフルランク条件を追加 | `59e8bdc` |
| 4 | B | 第3回の射影正規分布 | 正当な方向分布であることを認め、vMFの採用理由を平均方向、集中度、解析容易性へ限定 | `59e8bdc` |
| 5 | B | 第4回の最大エントロピー | $z_i=\lambda f_i$ と複数特徴量の場合を明記し、無条件の必然という表現を除去 | `59e8bdc` |
| 6 | B | 第5回のArcFace | 学習時のsoft lossであること、全標本への保証ではないこと、推論時は通常マージンを外すことを明記 | `59e8bdc` |
| 7 | B | 第6回のLayerNorm | LayerNorm直後と $W_Q,W_K$ 射影後のノルムを分離し、角度近似の追加条件を明記 | `59e8bdc` |
| 8 | B | 第7回のvMFエントロピー | 高 $\kappa$ 近似の完全な形を示し、383.5倍なのは $\log\kappa$ に対する傾きだと修正 | `59e8bdc` |
| 9 | B | 第8回の比較コード | 同じ初期値とベクトル場を用いるEuler法対ODEソルバーの比較へ変更し、最終L2誤差を出力 | `7c3c6ca` |
| 10 | B | 第11回のモダリティギャップ | 初期化と対照学習目的の相互作用、反発項、温度依存性を明記 | `59e8bdc` |
| 11 | B | 第14回のPH微分可能性 | 固定pairing領域での勾配と非滑らかな切替点を区別し、微分可能なTopology Layerを引用 | `59e8bdc` |
| 12 | B | Appendix 2のCoT参照 | 将来の仮説から誤った第8回リンクを除去し、本編接続を実在する第10回の内容へ変更 | `fcfd5cb` |
| 13 | B | Appendix 3の動的剪定 | 標準Attention、Sparse Attention、MoE、MoD、GQA、LoRA、FlashAttentionを機構別に再分類 | `fcfd5cb` |
| 14 | C | 第14回のNaitzat実験条件 | 二値分類、実・合成点群、学習状態、ReLU/tanh、浅層/深層の比較条件を追加 | `59e8bdc` |

## 追加で修正した問題

- Naitzat et al.の論文にないSkip connectionの効果を同論文へ帰属していた記述を削除した。
- 第14回へ微分可能なトポロジー層の一次文献を追加し、`references.bib` と `site/series/refs/14.yml` から参考文献節を再生成した。
- 第8回の4個の乱数使用例に `torch.manual_seed(42)` を追加した。
- Appendix 3の見出し変更に合わせてクイズの参照slugを更新した。

## 根拠として確認した一次文献

- Watanabe (2013): RLCTの一般式に $k_j$ だけでなく $h_j$ が入り、局所チャート間でも最小化することを確認した。
  - <https://jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf>
- Liang et al. (2022): モダリティギャップを初期化と対照学習最適化の組合せとして説明していることを確認した。
  - <https://arxiv.org/abs/2203.02053>
- Naitzat et al. (2020): 二値分類、十分に学習されたモデル、実・合成点群、活性化関数と深さの比較条件を確認した。
  - <https://www.jmlr.org/papers/v21/20-345.html>
- Gabrielsson et al. (2020): パーシステントホモロジーを利用する微分可能なTopology Layerが提案済みであることを確認した。
  - <https://proceedings.mlr.press/v108/gabrielsson20a.html>

## 機械検証

実行順は、リポジトリのPR前チェックに合わせた。

1. `task snippets:apply`: 成功
2. `task references:apply`: 成功
3. `task format:all`: 成功
4. `task lint:all`: 成功
5. `task site:build`: 成功
6. `git diff --check origin/main...HEAD -- . ':(exclude)check-all/**'`: 成功

サイトビルドでは500 kBを超えるchunkの警告が出るが、ビルド自体は完了している。
この警告は今回の妥当性判断を覆すエラーではない。

## 検証上の制約

標準の `uv sync` で導入される依存関係にはPyTorch、Matplotlib、torchdiffeqが含まれない。
このため、これらに依存する講義用snippetのランタイム実行は行っていない。
静的にはruff、Python書式、snippet同期、行数規約を通過している。

## 最終判定

未解決のA級、B級、C級指摘は0件である。
機械検証と差分レビューの範囲では、ブランチの変更は妥当である。
