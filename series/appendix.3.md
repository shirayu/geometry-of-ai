# Appendix 3: 動的剪定の幾何学: 柔軟な回路がもたらす知能

## 注意事項

本Appendixで扱う内容には、確立された数学的事実と、教育的な解釈が混在している。特に以下の点に留意されたい：

- **「剪定」という用語の定義**：本資料では「剪定」を二つの意味で使い分ける：
  - **情報的剪定（重み付けによる寄与の調整）**：標準Attentionはこちら。計算自体はO(n²)で密に行われるが、Softmaxによって一部の接続の重みが小さくなる。
  - **計算的剪定（計算自体の省略）**：Sparse Attention（Top-K Attention等）やMoEはこちら。実際に計算を省く。
  この区別は重要であり、混同すると誤解を招く。
- **スパース性の「必然」について**：「高次元→スパース化が必然→性能向上」という流れは、**特定の条件下**（無関係な次元が多い、ノイズが独立、表現が過剰次元など）での傾向であり、普遍的な法則ではない。Dense表現が有利な局面（タスクが全次元の微細な相関を要求する、スパース化によるルーティング崩壊など）も存在する。
- MoEの各Expertが「直交部分空間を担当」するという記述は理想化された仮説であり、実際の学習済みモデルでは部分的な重なりや共有専門性が観察される。完全な直交分離が常に達成されるわけではない。
- FlashAttentionの「幾何学的適合」という表現は比喩である。本質はHBM↔SRAM間のI/O削減と再計算戦略の組合せであり、厳密には計算複雑性とメモリアクセスパターンの最適化問題である。
- 「抽象化」と「剪定」の関係は哲学的考察を含む。認知科学・情報理論との接続は研究途上である。

## 導入：静的な地図から動的な回路へ

### 古典的手法の「固定性」

第1回で見たように、従来の機械学習手法は**入力に依存しない固定された計算経路**を持っていた。

畳み込みニューラルネットワーク（CNN）を例に取ろう。画像がどのようなものであっても、すべての画素はすべてのフィルターを通過する。猫の画像でも、空の写真でも、同じカーネルが同じ順序で適用される。これは、**静的な地図**を持ち歩くようなものだ。地形が変わっても、同じ地図を読み続ける。

```python
# CNNの典型的な計算フロー
x = input_image  # どんな画像でも
x = conv1(x)     # 常に conv1 が適用される
x = relu(x)
x = conv2(x)     # 常に conv2 が適用される
x = relu(x)
# ... 入力に関わらず同じ経路
```

### Transformer以降の「動的性」

Transformer（Vaswani et al., 2017）の登場は、この前提を根本から覆した。Attention機構は、**入力データ自身が計算経路を決定する**設計である。

「I saw a bat in the cave」という文では、「bat」は動物として処理される。「I saw a bat on the field」では、野球道具として処理される。同じ単語「bat」でも、周囲のトークン（cave vs field）との関係性によって、**異なる重み付けのAttentionパターン**が生まれる。

これは、**動的な回路**を持つことに等しい。入力が変われば、情報の流れ自体が変化する。

| 特性 | 従来（CNN等） | Transformer以降 |
| --- | --- | --- |
| 計算経路 | 入力に依存しない | 入力が決定する |
| 情報の流れ | 固定 | 動的 |
| 比喩 | 静的な地図 | 動的な回路・測量 |
| 適応性 | 低い | 高い |

> [!NOTE]
> **CNNの動的要素:** 厳密には、CNNもActivation関数によって非線形な経路選択を行う。また、SENet（Squeeze-and-Excitation Networks）などは、入力依存のチャネル重み付けを導入しており、部分的に動的である。しかし、Transformerの「トークン間の全結合を入力ごとに決める」レベルの動的性とは質的に異なる。

この動的性こそが、現代のAIモデルが持つ「適応性」の根源である。次項以降で、この適応性を「剪定（Pruning）」という概念で統一的に理解していく。

## Attention：空間内のミクロな動的枝刈り

### 全結合からの選択的遮断

Self-Attentionは、形式的にはすべてのトークン間の全結合グラフを考える。シーケンス長 $n$ のとき、 $n \times n$ の接続が存在する。

しかし、Softmaxを通過した後、実際に**有意な重みを持つ接続は一部だけ**である。これは、全結合グラフから「重要でない辺を切り落とす」操作と見なせる。

```txt
全結合グラフ（潜在的な接続）:
Token1 ------ Token2
  |    \      /  |
  |     \    /   |
  |      \  /    |
Token3 ------ Token4

Attention適用後（太線 = 高重み, 細線 = 低重み）:
Token1 ====== Token2
  |              |
  |              |
  ·              ·
Token3        Token4
（Token1-Token4 間の接続は実質的に遮断）
```

### 幾何学的解釈：内積による「視界の制限」

第6回で見たように、Attention scoreは Query と Key の内積（またはコサイン類似度）で計算される。

$$\text{score}(q_i, k_j) = \frac{q_i^\top k_j}{\sqrt{d_k}}$$

内積は、高次元空間における**方向の類似度**を測る。内積が大きい = 角度が小さい = ベクトルが「似た方向を向いている」。

Softmaxは、この類似度を確率分布に変換する。結果として、**方向の類似度が低い（内積が小さい）トークンへの接続は、確率的に遮断される**。

これは、高次元空間で「見るべき方向」以外を遮断する**視界の制限**と解釈できる。

> [!NOTE]
> **天体観測メタファーとの接続:** 第6回で導入した天体観測メタファーでは、Queryは「望遠鏡のフィルター」、Keyは「星の輝き」だった。フィルター（Query）が特定の波長に設定されているとき、その波長を放つ星（Key）だけが見える。他の星は存在するが、観測者には見えない ≒ Attention重みが低い。

### 情報的剪定としてのAttention：計算は省かない

従来の剪定（Pruning）は、ニューラルネットワークの重みやニューロンを恒久的に削除する**計算的剪定**である（LeCun et al., 1990）。

標準Attention（Scaled Dot-Product Attention）は、**情報的剪定**である。重要な違いとして：

- **計算自体は省略しない**：すべてのトークン間で内積を計算し（O(n²)）、Softmaxを適用する。この意味で密（Dense）な計算である。
- **情報流を重み付ける**：Softmaxによって、一部の接続の重みが小さくなり、「実質的に寄与が小さい」状態になる。しかし、計算は行われる。

| 種類 | 計算の省略 | 情報の重み付け | 動的性 | 例 |
| --- | --- | --- | --- | --- |
| **計算的剪定** | ○（計算を省く） | △（削除or保持） | 静的 | 構造化Pruning（実効速度は実装依存） |
| **標準Attention** | ✗（全計算） | ○（Softmax重み） | 動的 | Transformer |
| **Sparse Attention** | ○（Top-K等） | ○（重み付け） | 動的 | Top-K Attention, Local Attention, Block-sparse Attention |

> [!IMPORTANT]
> **「剪定」という用語の注意**：本資料では教育的メタファーとして「Attention=動的剪定」と表現しているが、厳密には標準Attentionは**計算を省いていない**。真の計算削減を行うのは、Sparse Attention（Top-K Attention、Local Attention、Block Sparse Attention等）やMoEである。

この視点から見ると、Attentionは「どの情報源（Key-Value）に**重みを割り当てるか**」を、入力ごとに最適化する**動的な資源配分機構**と理解できる。ただし、「資源」は計算時間ではなく、情報の伝播における「影響力」である。

### MoEへの接続

この「ミクロな動的剪定」のアイデアを、モデル全体のスケールに拡張したものが、次項で扱うMixture of Experts（MoE）である。

- **Attention:** トークン間のどの**接続**を使うか
- **MoE:** モデル全体のどの**パラメータ領域**を使うか

両者とも、「全体を持ちながら、必要な部分だけを活性化する」という共通原理に基づいている。

## Mixture of Experts (MoE)：マクロな部分空間スイッチング

### MoEの基本構造（復習）

第13回で導入したMixture of Experts（MoE）を、動的剪定の視点から再訪しよう。

MoEは、複数の「専門家」（Expert）ネットワークを用意し、**入力に応じて一部の専門家だけを活性化する**アーキテクチャである。

```txt
入力 x
   ↓
ルーター g(x)  ← 「どのExpertを使うか」を決定
   ↓
[Expert_1, Expert_2, ..., Expert_N]
   ↓（Top-K選択）
活性化されたExpertのみが計算される
   ↓
重み付き和 → 出力
```

例えば、8つのExpertのうち上位2つだけを活性化するTop-2 routingでは：

- **計算量:** 約 $2/8 = 1/4$ （Denseモデルと比較）
- **パラメータ総数:** 約8倍
- **結果:** 「大容量だが軽量」なモデル

### 「全知識の動員」vs「必要な近傍の活性化」

Denseモデル（MoEでないモデル）は、どんな入力に対しても**すべてのパラメータ**を使う。これは「全知識を動員する」アプローチである。

MoEは、入力に関連する**部分的な知識（Expert）だけを活性化**する。これは「必要な近傍だけを探索する」アプローチであり、第13回で扱ったkNNとの構造的類似性がある。

| 比較軸 | Denseモデル | MoE |
| --- | --- | --- |
| 活性化パラメータ | 全体 | 一部（Top-K） |
| 哲学 | 全知識の動員 | 必要な部分の選択 |
| 計算量 | $O(N)$ | $O(K)$ （ $K \ll N$ ） |
| 表現容量 | パラメータ数に比例 | パラメータ数の $N$ 倍（ $N$ : Expert数） |

### ルーティングとしての動的剪定

MoEのルーター（どのExpertを使うかを決める機構）は、典型的には以下のように実装される：

$$g_i(x) = \text{softmax}(\mathbf{W}_g \mathbf{x})_i$$

$$\text{output} = \sum_{i \in \text{TopK}(g(x))} g_i(x) \cdot \text{Expert}_i(x)$$

ここで、 $\mathbf{W}_g$ の各行ベクトルは、各Expertの「ゲートベクトル」と解釈できる。ルーティングは、**入力ベクトルとゲートベクトルの内積が大きいExpertを選ぶ**操作である。

これは本質的に、Attentionと同じ構造である：

| 機構 | Query | Key | Value | 選択方式 |
| --- | --- | --- | --- | --- |
| Attention | $\mathbf{q}_i$ | $\mathbf{k}_j$ | $\mathbf{v}_j$ | Softmax + 加重和 |
| MoE | $\mathbf{x}$ | $\mathbf{W}_g$ の行 | Expert関数 | TopK + 加重和 |

> [!NOTE]
> **MoEとkNNの類似性:** ルーティングは「入力に最も近い（内積が大きい）Expertを選ぶ」という意味で、kNNの変種と見なせる。高次元空間での近傍探索問題として、MoEを理解することもできる。

### 部分空間スイッチングとしてのMoE（仮説的解釈）

MoEの一つの**解釈的仮説**として、各Expertが**異なる部分空間を担当**しているという見方がある。これは理想化されたモデルであり、実際の学習済みモデルで厳密に成り立つとは限らないが、直感的理解には有用である。

高次元表現空間 $\mathbb{R}^d$ において、各Expertが主に活性化される領域が存在すると考える：

```txt
高次元空間の分割（概念図・理想化）:

      ┌─────────────┐
      │  Expert 3   │ ← 主に言語的ニュアンス関連で活性化
      │   (領域3)    │
      └─────────────┘
     ╱              ╲
 ┌──────┐        ┌──────┐
 │Expert│        │Expert│
 │  1   │        │  2   │
 │(数学)│        │(コード)│
 └──────┘        └──────┘
```

> [!CAUTION]
> **理想と現実のギャップ**：
> - **完全な直交分離は稀**：実際のMoEでは、Expertの担当領域は部分的に重なる。完全に直交した部分空間に分離されるわけではない。
> - **共有専門性の存在**：複数のExpertが共通の知識を持つことも多い（例：すべてのExpertが基本的な言語理解を共有）。
> - **学習条件依存**：Expert間の分化の程度は、補助損失（Load Balancing Loss）、学習率、データ分布、モデルサイズなどに強く依存する。
> - **測定の困難**：「Expert間の直交性」を定量的に測定する方法自体が研究課題である。
> 
> 一部の研究（Kudugunta et al., 2021）では学習が進むとExpert間の類似度が下がる傾向が観察されているが、これが普遍的かつ「直交部分空間への分離」を意味するかは未解明である。

### より現実的なMoE解釈

実用的には、MoEの各Expertは以下のように理解される方が安全である：

- **専門化の傾向**：各Expertは特定のデータパターンや入力分布の部分領域に対して、他のExpertよりも「得意」になる傾向がある。
- **部分的重なり**：Expert間で完全に独立ではなく、基礎的な処理は共有し、高度な処理で分化する。
- **動的協調**：複数のExpertが協調して一つの入力を処理する（Top-K routing）。

### MoEの課題：ルーティング崩壊（Routing Collapse）

MoEの実装において最も重要な課題の一つが、**ルーティング崩壊**（Routing Collapse / Expert Collapse）である。

**何が起きるか**：
すべて（または大多数）のトークンが**同じExpertに集中**し、他のExpertがほとんど使われなくなる現象。

```txt
理想的なMoE（均等な負荷分散）:
Expert1: ████ (25%)
Expert2: ████ (25%)
Expert3: ████ (25%)
Expert4: ████ (25%)

ルーティング崩壊後:
Expert1: ████████████████████████ (90%)
Expert2: ■ (3%)
Expert3: ■ (3%)
Expert4: ██ (4%)
```

**問題点**：
- MoEの利点（容量の拡大、専門化）が失われる
- 実質的に「小さなDenseモデル」になる
- 大部分のパラメータが未使用のまま

**幾何学的解釈**：
表現空間が一部の部分空間（または領域）に**縮退**し、高次元空間の大部分が未使用になる。これは第13回で扱った「次元崩壊」の変種である。

#### 対策：Load Balancing Loss（負荷分散損失）

ルーティング崩壊を防ぐため、**Load Balancing Loss**が訓練時に追加される。Switch Transformer（Fedus et al., 2022）では、以下のような補助損失が導入される：

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N_{\text{experts}} \cdot \sum_{i=1}^{N_{\text{experts}}} f_i \cdot p_i$$

**記号の定義**（論文・解説により記号と概念の対応が入れ替わることがあるため注意）：
- $f_i$ : Expert $i$ に実際に割り当てられたトークンの割合（**load** - 実測値）
- $p_i$ : Expert $i$ へのルーティング確率の平均（**importance** - 期待値）
- $N_{\text{experts}}$ : Expert数
- $\alpha$ : バランス係数

> [!NOTE]
> **記号の揺れについて**：文献によって、$f_i$ を「importance（確率質量の総和）」、$p_i$ を「load（割当トークン比率）」と逆に定義する場合もある。本質は「**実測の負荷** × **期待される重要度** の積を最小化する」ことであり、記号より概念が重要である。

**仕組み**：
- 特定のExpertに負荷が集中すると、そのExpertの $f_i \cdot p_i$ が大きくなる
- この項を損失に加えることで、集中に対してペナルティを与える
- 訓練中、モデルは負荷を分散させるようルーティングを調整する

> [!WARNING]
> 上記は概念を示すための簡略形であり、**この式をそのまま実装に使用してはならない**。実際の論文では：
> - バッチ内での補助損失として定義
> - importance loss と load loss を分けて定義
> - Expertごとの確率分布の扱い
> - バッチサイズとの関係
> など、より精緻な形式が用いられている。実装には原論文を必ず参照されたい。

**幾何学的解釈**：
Load Balancing Lossは、入力空間の分割を均等化するよう促す**正則化**である。表現空間が特定の部分空間に縮退することを防ぎ、高次元空間を「まんべんなく使う」ことを強制する。

#### 他の対策

- **Capacity Factor**：各Expertが受け入れるトークン数に上限を設ける。上限を超えたトークンは他のExpertに割り当てられる。
- **Expert Choice Routing**（Zhou et al., 2022）：トークンがExpertを選ぶのではなく、ExpertがトークンをTop-K選択する。
- **Shared Expert**：すべての入力に対して常に活性化する共通Expertを設ける（DeepSeekMoE, Dai et al., 2024で明示的に採用）。共通知識の重複を防ぎつつ、専門化を促進。Shared Expertは負荷分散の直接的な手段ではないが、専門Expertの役割を明確化する効果がある。

#### 実装ノートとの接続

本資料の実装ノートセクション（後述）で提示したMoEの簡略実装には、Load Balancing Lossが**含まれていない**。これは教育目的の簡略化である。

実際のMoE（Mixtral、Switch Transformer、DeepSeek-V3等）では、この補助損失が訓練の安定性と性能に**極めて重要**である。Load Balancing Lossや類似の負荷分散機構なしで訓練すると、**ルーティング崩壊が発生しやすく**、実運用規模では何らかの負荷分散機構（補助損失、ルーターバイアス調整、Capacity Factor等）が**重要**となる。

> [!NOTE]
> **負荷分散の多様性**：Load Balancing Lossは代表的な手法だが、唯一の解ではない。ルーター設計、初期化、Top-Kの選び方、Capacity Factorの調整などでも負荷分散に影響する。近年は「補助損失なし／弱い補助損失」でも負荷を保つ手法の研究もある（例：Auxiliary-Loss-Free Load Balancing Strategy, He et al., 2024）。

## スパース性（疎性）の幾何学：条件依存の有効性

### 高次元空間の「空虚さ」

第2回・第13回で繰り返し述べてきたように、高次元空間ではランダムベクトル同士が**ほぼ直交**する。

$d$ 次元単位球面上で一様ランダムに選んだ2つのベクトル $\mathbf{u}, \mathbf{v}$ の内積は、 $d \to \infty$ で平均0、分散 $1/d$ に集中する：

$$\mathbb{E}[\mathbf{u}^\top \mathbf{v}] = 0, \quad \text{Var}[\mathbf{u}^\top \mathbf{v}] = \frac{1}{d}$$

これは、高次元空間のほとんどの方向が**互いに無関係**であることを意味する。言い換えれば、**空間のほとんどが「空」**である。

### スパース化の有効性：条件依存

この性質から、**特定の条件下で**スパース化が有効になる可能性が示唆される：

> [!NOTE]
> **スパース化が有効となる条件（例）**：
> - データが実際には低次元多様体上に存在する（内在次元 << アンビエント次元）
> - 無関係な次元が独立なノイズとして振る舞う
> - タスクが特定の部分空間に関連する特徴のみを要求する
> - 表現空間が過剰に大きく、冗長性が高い

これらの条件下では、無関係な次元を計算に含めることは、ノイズを増やすことに等しくなりうる。

### SNR観点の例示モデル（概念的理解のための簡略化）

**以下は教育的な概念式であり、厳密な証明や一般的な定理ではない**ことに注意されたい。

ある**極めて理想化された**モデルを考える：
- 意味のある信号が $k$ 次元に存在
- 残りの $(d - k)$ 次元が**独立な**平均0のノイズ
- 各次元の信号とノイズが**加法的に分離可能**
- 各次元の**分散が等しい**（または同程度のオーダー）

この理想化された設定下では、信号対雑音比（SNR）は概念的に以下のオーダーとなる：

$$\text{SNR} \sim \frac{\text{信号次元}}{\text{ノイズ次元}} \sim \frac{k}{d-k}$$

（ここで $\sim$ はオーダーの意味であり、厳密な等式ではない）

$d$ が大きく $k$ が固定なら、SNRは悪化する傾向がある。したがって、**このモデルが成り立つ特定の条件下では**、無関係な次元を計算から除外することが情報の純度を高める可能性がある。

> [!CAUTION]
> 上記のSNRモデルは以下の強い仮定に基づいており、一般には成り立たない：
> - 信号とノイズが明確に分離可能
> - ノイズが独立同分布
> - 各次元の寄与が加法的
> 
> 実際のデータでは、「何が信号で何がノイズか」はタスク・学習・分布に依存し、事前には分からない。また、Dense表現が全次元の微細な相関を捉えて性能向上する場合もある。

### Dense表現が有利な場合

スパース化が常に最適とは限らない。Dense表現が有利な状況として：

| 状況 | 理由 |
| --- | --- |
| **タスクが全次元の相関を要求** | 微細な相関情報を捉えるには全次元が必要 |
| **スパース化によるルーティング崩壊** | MoEで特定Expertに負荷集中し、容量不足 |
| **表現力の不足** | 過度なスパース化で必要な情報まで失う |
| **学習の不安定性** | スパースな勾配による学習困難 |
| **データが真に高次元** | 内在次元がアンビエント次元に近い場合 |

> [!IMPORTANT]
> **スパース化の位置づけ**：本資料の議論は「高次元空間でスパース化が有効となる**条件**を理解する」ことが目的であり、「スパース化が常に優れている」という主張ではない。Dense vs Sparseの選択は、データ・タスク・モデル・計算資源などの文脈依存である。

## 効率化技術の体系化（GQA, LoRA, MoD）

MoEに限らず、近年のTransformer効率化手法の多くは、「空間のどの部分を剪定するか」という軸で整理できる。

### GQA (Grouped-Query Attention)：視点の冗長性削減

Multi-head Attentionでは、各ヘッドが独立した $Q, K, V$ を持つ。しかし、すべてのヘッドが完全に独立な情報を捉えているとは限らない。

**GQA**（Ainslie et al., 2023）は、複数のQueryヘッドで**同じKeyとValueを共有**する設計である。

```txt
標準Multi-head Attention:
Head1: Q1, K1, V1
Head2: Q2, K2, V2
Head3: Q3, K3, V3
Head4: Q4, K4, V4

GQA (グループサイズ2):
Group1: Q1, Q2 → 共有 K1, V1
Group2: Q3, Q4 → 共有 K2, V2
```

幾何学的には、これは**Key-Value空間の次元削減**である。「視点」の冗長性を剪定している。

#### 実務上の最大のメリット：KVキャッシュの削減

幾何学的解釈に加えて、GQAの**実装上の最大のメリット**は、推論時の**KVキャッシュのメモリ削減**にある。

**KVキャッシュとは**：
自己回帰的な生成（テキスト生成等）では、各ステップで過去のトークンのKey/Valueを再利用する。これを「KVキャッシュ」として保存することで、計算を省略する。

```txt
生成プロセス（例：文章生成）:
ステップ1: "The" → K1, V1 をキャッシュ
ステップ2: "The cat" → K1,V1を再利用、K2,V2を追加
ステップ3: "The cat sat" → K1,V1,K2,V2を再利用、K3,V3を追加
...
シーケンス長が長くなると、キャッシュサイズが線形増加
```

**問題点（Memory Wall - 容量と帯域の制約）**：
- 長いシーケンス（数千〜数万トークン）では、KVキャッシュが**数GB〜数十GB**に達する
- GPUのHBM容量を圧迫し、バッチサイズを制限（スループット低下）
- これは**Memory Wall**と呼ばれる問題の一側面であり、特に推論時の**メモリ容量制約**と**メモリ帯域制約**の両面から性能ボトルネックとなる
  - **容量制約**：KVキャッシュが大きすぎてHBMに載らない、またはバッチサイズを小さくせざるを得ない
  - **帯域制約**：HBMへのアクセス速度が計算速度に追いつかない（FlashAttentionが主に対処する問題）

**GQAの効果**：
標準Multi-head Attention（8ヘッド）では、各ヘッドがK,Vを持つため、キャッシュサイズは $8 \times d$ に比例。

GQA（8 QueryヘッドをKVヘッド2つで共有）では、キャッシュサイズは $2 \times d$ に比例。

$$\text{KVキャッシュ削減率} = 1 - \frac{2}{8} = 75\%$$

この削減により：
- 同じメモリ容量で**より長いシーケンス**を処理可能
- または、**バッチサイズを拡大**してスループット向上

> [!NOTE]
> **FlashAttentionとの一貫性**：GQAのメモリ削減は、FlashAttentionのI/O削減と同じ「メモリ階層の制約」という文脈にある。ただし、対処する側面が異なる：
> - **FlashAttention**：計算時のHBM↔SRAM間I/O削減（**帯域制約**への対処）
> - **GQA**：推論時のHBM使用量削減（**容量制約**への対処）
> 
> 両者は相補的であり、併用することで推論効率を大きく改善できる。

### LoRA (Low-Rank Adaptation)：更新ランクの削減

**LoRA**（Hu et al., 2021）は、ファインチューニング時に重み行列の**低ランク分解**を学習する手法である。

元の重み行列 $W \in \mathbb{R}^{d \times d}$ を更新する代わりに、低ランク行列 $BA$ を学習する：

$$W' = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}, \quad r \ll d$$

幾何学的には、重みの更新を**低次元部分空間に制約**している。「重み更新の自由度」を剪定している。

### MoD (Mixture of Depths)：深さ方向の剪定

**MoD**（Raposo et al., 2024）は、各トークンが通過する**層の数を動的に変える**設計である。

```txt
標準Transformer:
すべてのトークンが全層を通過
Token1: L1 → L2 → L3 → L4
Token2: L1 → L2 → L3 → L4
Token3: L1 → L2 → L3 → L4

MoD:
トークンごとに通過する層が異なる
Token1: L1 → L2 → skip → L4  （L3をスキップ）
Token2: L1 → L2 → L3 → L4    （全層通過）
Token3: L1 → skip → skip → L4 （L2, L3をスキップ）
```

幾何学的には、トークンの「変換の深さ」を剪定している。重要なトークンは深く処理され、周辺的なトークンは浅い処理で済ませる。

### 統一的整理

これらの手法を、「幾何学的剪定のバリエーション」として整理できる：

| 手法 | 剪定の対象 | 剪定の次元 | 動的性 |
| --- | --- | --- | --- |
| **Attention** | トークン間接続 | 接続方向 | 動的（入力依存） |
| **MoE** | パラメータ領域 | Expert方向 | 動的（入力依存） |
| **GQA** | Key-Valueヘッド | Head方向 | 静的（設計時固定） |
| **LoRA** | 重み更新空間 | ランク方向 | 静的（学習時固定） |
| **MoD** | 層の通過 | 深さ方向 | 動的（入力依存） |

個別の「流行技術」ではなく、**幾何学的な空間効率化**という統一的視点で理解できる。

## FlashAttention：メモリI/O最適化としての設計

### GPUメモリ階層という制約

これまで扱ってきた幾何学は、数理的な表現空間の話だった。しかし、実際の計算は**物理デバイス上で実行**される。ここに、もう一つの「制約空間」がある。

現代のGPUは、階層的なメモリ構造を持つ：

```txt
メモリ階層（上に行くほど高速・小容量）:
*数値はGPU世代による代表例

┌─────────────────┐
│ レジスタ (Registers)         │ ～ KB級, 超高速
├─────────────────┤
│ SRAM (Shared Memory/L1キャッシュ) │ 数十～数百KB, 高速（オンチップ）
├─────────────────┤
│ HBM (High Bandwidth Memory)    │ ～ 数十GB, 低速（オンチップではない外部メモリ）
└─────────────────┘
```

標準Attentionは巨大な行列（シーケンス長 $n$ で $n \times n$ ）を扱うため、通常HBMに格納される。しかし、**HBM↔SRAM間のデータ転送（I/O）が計算時間のボトルネック**となる。

### FlashAttentionの核心：I/O削減と再計算戦略

**FlashAttention**（Dao et al., 2022）の本質は、以下の**複数の技術の組合せ**である：

1. **タイリング（ブロック分割）**：行列を小さなタイル（ブロック）に分割し、各タイルをSRAMに載せて高速計算
2. **オンラインSoftmax**：Softmaxの計算を段階的に更新することで、全行列を一度にメモリに載せる必要を回避
3. **再計算（Recomputation）**：Backward時にAttention行列を保存せず再計算することで、Forward時のメモリ保存を削減

これらにより、HBM↔SRAM間のI/O回数を劇的に削減する。

```txt
通常のAttention:
1. Q, K, V を HBM から読み込み
2. Q @ K^T を計算 → HBM に保存（巨大）
3. Softmax を計算 → HBM に保存
4. Attention @ V を計算
→ HBM への読み書きが多い（遅い）

FlashAttention:
1. Q, K, V を小さなタイルに分割
2. 各タイルを SRAM に載せて計算
3. オンラインSoftmaxで段階的に更新
4. 中間結果を HBM に保存せず、必要時に再計算
→ HBM ↔ SRAM の I/O が劇的に削減（速い）
```

### 「パッキング」比喩の限界と正確な理解

本資料では教育的比喩として「物理空間へのパッキング」と表現したが、これは厳密な説明ではない。より正確には：

- **最適化目標**：HBM↔SRAM間のデータ転送回数を最小化
- **制約**：SRAMの容量制限（～100 KB）
- **手法**：タイリング + オンライン更新 + 再計算のトレードオフ

| 比喩的表現 | より正確な表現 |
| --- | --- |
| 「SRAMという容器にパッキング」 | 「I/O回数を削減するブロック分割」 |
| 「幾何学的適合」 | 「メモリアクセスパターンの最適化」 |
| 「物理空間の制約」 | 「計算複雑性とメモリ階層の制約」 |

> [!NOTE]
> **比喩の役割**：「パッキング」という比喩は、「アルゴリズム設計が物理制約に適合する必要がある」という直感を伝えるには有用である。しかし、FlashAttentionの真価はI/O削減にあり、これは計算複雑性理論とハードウェアアーキテクチャの共設計（co-design）の問題である。

## 結論：知能と抽象化

### 「何を計算しないか」を決める知能

本Appendixで見てきたように、動的剪定の本質は**「何を計算するか」ではなく「何を計算しないか」を決めるプロセス**である。

- **Attention:** 情報流の重み付けにより、寄与の小さい接続の影響を低減
- **Sparse Attention:** 計算自体を省略し、重要な接続のみ処理
- **MoE:** 無関係なExpertを活性化せず、必要な部分領域のみ計算

これは、情報を**選択的に処理する**行為である。

### 抽象化としての選択

認知科学の観点から見ると、「情報を選択してエッセンスを抽出する」という行為は、**抽象化**（Abstraction）の一形態である。

人間は、世界のすべての詳細を記憶しない。重要な特徴だけを抽出し、それを元に推論する。これは、高次元の入力を低次元の表現に**射影**する行為であり、幾何学的には次元削減である。

動的剪定は、この抽象化を**動的に・文脈依存的に**行う。同じ情報でも、文脈が変われば重要度が変わる。これは、人間の注意（Attention）の働きに近い。

> [!NOTE]
> **知能の本質としての選択:** 情報理論の創始者Claude Shannon は、「情報は予測不可能性である」と述べた。しかし、予測不可能性をそのまま保持することは、ノイズを保持することでもある。知能は、予測可能な（冗長な）情報を捨て、予測不可能な（本質的な）情報を保持する能力である。この意味で、選択的処理は知能の中核的機能と言える。

### 本資料の位置づけと限界

本Appendixは、**教育的な統一視点を提供する**ことを目的としており、以下の点を理解されたい：

**意図（達成できること）**：
- AttentionとMoEを「選択的処理」という共通原理で理解する
- 効率化技術（GQA, LoRA, MoD）を統一的に整理する
- 高次元空間の性質とスパース化の関係を直感的に把握する
- ルーティング崩壊やKVキャッシュなど、実装上の重要課題を認識する

**限界（注意すべきこと）**：
- 「剪定」という用語は比喩的であり、厳密な定義ではない
- 「スパース化が常に有効」ではなく、条件依存である
- MoEの「部分空間分割」は理想化された仮説であり、実際は複雑
- FlashAttentionの「パッキング」は教育的比喩であり、本質はI/O削減
- Load Balancing Lossの簡略式は概念示唆であり、実装には使用不可
- Load Balancing Lossの記号定義（$f_i, p_i$）は文献により揺れがある
- ルーティング崩壊の発生条件は設計・データ・初期化に依存し、一概には言えない

**読者へのメッセージ**：
本資料は「正解」を示すものではなく、「見方」を提供するものである。この視点が有用かどうかは、読者が扱う問題・データ・モデルに依存する。批判的に読み、自身の状況に適用可能かを判断されたい。

### 最終的な問い

本講義シリーズに通底する問いに戻ろう：

**「AIとは何か」**

動的剪定の視点から見れば、AIは「高次元空間における適応的な情報選択システム」である。どの情報を保持し、どの情報を捨てるかを、文脈に応じて動的に決定する。

しかし、この選択基準自体は、**学習データと目的関数に依存する**。何を「重要」と見なすかは、設計者と訓練データが決める。

ここに、技術的問題を超えた、価値判断の問題が現れる。本講義では扱わないが、次のステップとして意識すべき視点である。

## 実装ノート

### Attentionの計算パターン比較

> [!WARNING]
> **教育目的のコード**：以下の`topk_reweighting_attention`は、「情報的剪定（重み付け）」と「計算的剪定（省略）」の違いを示すための教育的実装である。実際には以下の理由で実用的でない：
> 1. **全体のスコア行列を計算**している（O(n²)の計算は省略されていない）
> 2. **ゼロ行列を作ってscatter**する実装は、メモリ効率が悪く、標準実装より遅い可能性が高い
> 3. 真のSparse Attentionには、専用のカーネル（CUDA実装）やCSR等のスパースデータ構造が必要
> 
> 本コードの目的は「Top-K選択の概念」を示すことであり、性能改善ではない。速度比較は意味を持たない。

```python
import torch
import time

def standard_attention(Q, K, V):
    """標準的なAttention（密な計算、O(n²)メモリ）"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attention_weights = torch.softmax(scores, dim=-1)  # O(n^2) のメモリ
    output = torch.matmul(attention_weights, V)
    return output

def topk_reweighting_attention(Q, K, V, k=10):
    """Top-Kによる情報的重み付け（教育目的の概念実装）
    
    注意：このコードは計算を省略していない（全スコアを計算している）。
    真の計算的剪定には、スコア計算自体をスキップする必要がある。
    実行速度は標準実装と同等か、むしろ遅い可能性が高い。
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # ← 全計算（省略なし）
    
    # Top-K選択（各クエリについて上位k個のキーのみ保持）
    topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1)
    
    # スパース行列を構築（実際にはより効率的な実装が必要）
    sparse_weights = torch.zeros_like(scores)  # ← ゼロ行列作成（非効率）
    sparse_weights.scatter_(-1, topk_indices, torch.softmax(topk_scores, dim=-1))
    
    output = torch.matmul(sparse_weights, V)
    return output

# ベンチマーク（参考：速度比較は意味を持たない）
batch_size, num_heads, seq_len, d_k = 2, 8, 512, 64
Q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda')
K = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda')
V = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda')

# 標準Attention
start = time.time()
out1 = standard_attention(Q, K, V)
torch.cuda.synchronize()
time1 = time.time() - start

# Top-K重み付け (k=32) - 教育目的実装
start = time.time()
out2 = topk_reweighting_attention(Q, K, V, k=32)
torch.cuda.synchronize()
time2 = time.time() - start

print(f"Standard Attention: {time1:.4f}s")
print(f"Top-K Reweighting (k=32, 教育実装): {time2:.4f}s")
print(f"Note: 教育実装は標準実装と同等か、むしろ遅い可能性が高い")
print(f"      真の高速化には専用カーネルが必要")
```

### MoEの簡略実装

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMoE(nn.Module):
    """教育目的のMoE実装"""
    def __init__(self, d_model, num_experts=8, expert_capacity=2, expert_hidden=2048):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # ルーター
        self.router = nn.Linear(d_model, num_experts)
        
        # Experts（簡単なFFN）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, d_model)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # ルーティング
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K選択
        topk_probs, topk_indices = torch.topk(router_probs, k=self.expert_capacity, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # 再正規化
        
        # Expert計算（簡略化: バッチ処理を省略）
        output = torch.zeros_like(x)
        for i in range(self.expert_capacity):
            expert_idx = topk_indices[..., i]  # [batch, seq_len]
            expert_weight = topk_probs[..., i].unsqueeze(-1)  # [batch, seq_len, 1]
            
            # 各Expertの出力を加重和（実装簡略化）
            for e in range(self.num_experts):
                mask = (expert_idx == e).unsqueeze(-1)  # [batch, seq_len, 1]
                expert_out = self.experts[e](x)
                output += expert_out * expert_weight * mask
        
        return output

# 使用例
moe = SimpleMoE(d_model=512, num_experts=8, expert_capacity=2)
x = torch.randn(2, 10, 512)  # [batch=2, seq_len=10, d_model=512]
out = moe(x)
print(f"Input shape: {x.shape}, Output shape: {out.shape}")
```

> [!CAUTION]
> 上記は概念理解のための簡略実装である。実際のMoE（Mixtral, Switch Transformerなど）は、負荷分散、Expert collapse対策、効率的なバッチ処理など、多くの工夫が加えられている。実装には Hugging Face Transformers など、検証済みのライブラリを使用することを推奨する。

### GQA (Grouped-Query Attention) の実装

```python
class GroupedQueryAttention(nn.Module):
    """GQA (Grouped-Query Attention) の実装"""
    def __init__(self, d_model, num_query_heads, num_kv_heads):
        super().__init__()
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Q: [batch, seq_len, num_query_heads, d_k]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_query_heads, self.d_k).transpose(1, 2)
        
        # K, V: [batch, seq_len, num_kv_heads, d_k]
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # K, Vを各グループで共有するため、repeatで拡張
        # [batch, num_kv_heads, seq_len, d_k] -> [batch, num_query_heads, seq_len, d_k]
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)
        
        # 通常のAttention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # ヘッドを結合
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(output)

# 使用例: 8つのQueryヘッド、2つのKVヘッド（4:1のグループ化）
gqa = GroupedQueryAttention(d_model=512, num_query_heads=8, num_kv_heads=2)
x = torch.randn(2, 10, 512)
out = gqa(x)
print(f"GQA output shape: {out.shape}")
print(f"Parameters saved: ~{(1 - 2/8) * 100:.1f}% (for K,V)")
```

## 参考文献

### Transformer と Attention

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*. arXiv: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

### Pruning（剪定）

- LeCun, Y., Denker, J. S., & Solla, S. A. (1990). Optimal Brain Damage. *NeurIPS 1989*.
    - ニューラルネットワークの剪定の古典的論文。構造化pruningの場合は実効速度向上が見込めるが、非構造化pruningでは実装次第で速度向上が限定的な場合もある。

### Mixture of Experts (MoE)

- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR 2017*. arXiv: [arXiv:1701.06538](https://arxiv.org/abs/1701.06538).
    - 現代的MoEの基礎となった論文。

- Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR*, 23(120):1-39. arXiv: [arXiv:2101.03961](https://arxiv.org/abs/2101.03961).
    - Top-1 routingによるさらなるスパース化。

- Dai, D., Deng, C., Zhao, C., Xu, R. X., Gao, H., Chen, D., Li, J., Ding, W., Li, X., Xie, Y., Wang, Z., Chen, Y., Wei, Z., Liang, Y., Wu, Y., Yuan, Z., Zhou, J., Zhang, L., & Yu, F. R. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. arXiv: [arXiv:2401.06066](https://arxiv.org/abs/2401.06066).
    - 細粒度Expert + 共有Expertの設計。

- Puigcerver, J., Riquelme, C., Mustafa, B., & Houlsby, N. (2024). From Sparse to Soft Mixtures of Experts. *ICLR 2024*. arXiv: [arXiv:2308.00951](https://arxiv.org/abs/2308.00951).
    - 離散的ルーティングを連続化するSoft MoE。

- Kudugunta, S., Huang, Y., Bapna, A., Krikun, M., Lepikhin, D., Luong, M., & Firat, O. (2021). Beyond Distillation: Task-level Mixture-of-Experts for Efficient Inference. *Findings of EMNLP 2021*. arXiv: [arXiv:2110.03742](https://arxiv.org/abs/2110.03742).
    - Expert間の類似度に関する実験的観察。

- Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A., Chen, Z., Le, Q., & Laudon, J. (2022). Mixture-of-Experts with Expert Choice Routing. *NeurIPS 2022*. arXiv: [arXiv:2202.09368](https://arxiv.org/abs/2202.09368).
    - Expert Choice Routing（ExpertがトークンをTop-K選択）の提案。ルーティング崩壊への対策の一つ。

- He, X., Shen, C., Gan, Z., Tan, L., Wang, G., Zhao, Y., Chen, W., & Xu, Y. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts. arXiv: [arXiv:2408.15664](https://arxiv.org/abs/2408.15664).
    - 補助損失なしでも負荷分散を実現する手法の提案。Load Balancing Lossの代替アプローチの研究例。

### 効率化手法

- Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *EMNLP 2023*. arXiv: [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).
    - Grouped-Query Attentionの提案。

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. arXiv: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
    - 低ランク適応によるパラメータ効率的ファインチューニング。

- Raposo, D., Ritter, S., Richards, B., Lillicrap, T., Conway, P. W., & Santoro, A. (2024). Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models. arXiv: [arXiv:2404.02258](https://arxiv.org/abs/2404.02258).
    - 深さ方向の動的剪定。

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*. arXiv: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).
    - SRAMを活用したメモリ効率的Attention。

### 情報理論と抽象化

- Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3):379-423.
    - 情報理論の基礎。情報を「予測不可能性」として定式化。

## まとめ

本Appendixでは、「動的剪定」という視点で、AttentionとMoEを統一的に理解した。ただし、重要な区別として：

| 概念 | 定義 | 本Appendixでの役割 |
| --- | --- | --- |
| **情報的剪定** | 重み付けによる寄与の調整（計算は省かない） | 標準Attentionの解釈 |
| **計算的剪定** | 計算自体の省略 | Sparse Attention, MoEの解釈 |
| **部分空間スイッチング** | 入力に応じた部分空間の選択（仮説的） | MoEの一つの幾何学的解釈 |
| **スパース性の条件依存的有効性** | 特定条件下でのノイズ除去効果 | 剪定の情報理論的根拠（条件つき） |
| **効率化の体系化** | GQA, LoRA, MoD | 剪定のバリエーション |
| **抽象化** | 情報を捨ててエッセンスを抽出 | 知能の一側面としての剪定 |

### 重要な注意点

1. **「剪定」という用語の二義性**：情報的剪定（重み付け）と計算的剪定（省略）は異なる概念。標準Attentionは前者。
2. **条件依存性**：「スパース化が常に有効」ではなく、データ・タスク・モデルに依存。Dense表現が有利な場合もある。
3. **理想化と現実**：MoEの「直交部分空間」仮説は理想化されたモデル。実際は部分的重なりや共有専門性が存在。
4. **比喩の限界**：「パッキング」「幾何学的適合」は教育的比喩。FlashAttentionの本質はI/O削減。
5. **ルーティング崩壊**：MoEでは特定Expertへの集中が問題となりやすく、Load Balancing Loss等の負荷分散機構が重要。ただし、ルーター設計や初期化などでも影響を受ける。
6. **メモリ階層の制約**：
   - GQAのKVキャッシュ削減：推論時の**容量制約**に対処
   - FlashAttentionのI/O削減：計算時の**帯域制約**に対処
   - 両者は相補的
7. **記号の揺れ**：Load Balancing Lossの $f_i, p_i$ の定義は文献により入れ替わる。概念（load/importance）が重要。

### 講義本編との接続

- **第6回（Attention）:** 情報流の重み付けとして再解釈
- **第13回（MoE）:** 計算的剪定として再解釈
- **第2回（高次元の直交性）:** スパース性の条件依存的根拠
- **第3回（球面）:** 正規化と角度計算の幾何学

### 次のステップ

本Appendixで扱った「剪定」は、**計算資源の効率化**という文脈だった。しかし、同じ原理は**情報の選択**という、より広い文脈にも適用できる。

- Retrieval-Augmented Generation (RAG): 外部知識からの選択的取得
- アライメント: 望ましい行動の選択的強化
- 解釈可能性: 重要な特徴の選択的提示

これらはすべて、「高次元空間からの選択的射影」という幾何学的操作として統一的に理解できる可能性がある。

「何を見て、何を見ないか」を決める能力。それが知能の核心かもしれない。
