# Appendix 3: 動的剪定の幾何学: 柔軟な回路がもたらす知能

## 注意事項

本Appendixで扱う内容には、確立された数学的事実と、教育的な解釈が混在している。特に以下の点に留意されたい：

- Attentionを「動的剪定」と呼ぶのは解釈の一つであり、普遍的な用語ではない。ただし、計算資源の選択的配分という観点からは有用な視点である。
- MoEの各Expertが「直交部分空間を担当」するという記述は理想化された仮説であり、実際の学習済みモデルで厳密に成り立つとは限らない。
- FlashAttentionの「幾何学的適合」という表現は比喩である。本質はSRAMへの効率的なメモリアクセスパターンの設計であり、厳密には物理空間というより計算複雑性の問題である。
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

### ソフトな剪定としてのAttention

従来の剪定（Pruning）は、ニューラルネットワークの重みやニューロンを恒久的に削除する**ハード剪定**である（LeCun et al., 1990）。

Attentionは、**ソフト剪定**である。接続は削除されないが、確率的に重み付けされる。しかも、その重み付けは**入力ごとに変化する**。

| 種類 | 剪定の対象 | 剪定の性質 | 動的性 |
| --- | --- | --- | --- |
| ハード剪定（従来） | 重みやニューロン | 恒久的削除 | 静的 |
| Attention（ソフト剪定） | トークン間の接続 | 確率的重み付け | 動的 |

この視点から見ると、Attentionは「どの情報源（Key-Value）に計算資源を割り当てるか」を、入力ごとに最適化する**動的な資源配分機構**と理解できる。

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

### 部分空間スイッチングとしてのMoE

MoEの幾何学的解釈として、各Expertが**異なる部分空間を担当**しているという仮説がある。

高次元表現空間 $\mathbb{R}^d$ を、複数の部分空間に分割する：

$$\mathbb{R}^d \approx \text{Span}(\text{Expert}_1) \cup \text{Span}(\text{Expert}_2) \cup \cdots \cup \text{Span}(\text{Expert}_N)$$

入力 $\mathbf{x}$ が来ると、ルーターは「この入力はどの部分空間に属するか」を判定し、対応するExpertにルーティングする。これは**部分空間のスイッチング**である。

```txt
高次元空間の分割（概念図）:

      ┌─────────────┐
      │  Expert 3   │ ← 言語的ニュアンス
      │   (部分空間3)│
      └─────────────┘
     ╱              ╲
 ┌──────┐        ┌──────┐
 │Expert│        │Expert│
 │  1   │        │  2   │
 │(数学)│        │(コード)│
 └──────┘        └──────┘
```

> [!CAUTION]
> **理想と現実のギャップ:** 「各Expertが直交部分空間を担当」は理想化された仮説である。実際の学習済みMoEモデルで、Expertの担当領域が厳密に直交しているかは未解明である。一部の研究（Kudugunta et al., 2021）では、学習が進むとExpert間の類似度が下がる傾向が観察されているが、これが常に成り立つわけではなく、モデル・タスク・学習条件に依存する。

## スパース性（疎性）の幾何学的必然

### 高次元空間の「空虚さ」

第2回・第13回で繰り返し述べてきたように、高次元空間ではランダムベクトル同士が**ほぼ直交**する。

$d$ 次元単位球面上で一様ランダムに選んだ2つのベクトル $\mathbf{u}, \mathbf{v}$ の内積は、 $d \to \infty$ で平均0、分散 $1/d$ に集中する：

$$\mathbb{E}[\mathbf{u}^\top \mathbf{v}] = 0, \quad \text{Var}[\mathbf{u}^\top \mathbf{v}] = \frac{1}{d}$$

これは、高次元空間のほとんどの方向が**互いに無関係**であることを意味する。言い換えれば、**空間のほとんどが「空」**である。

### スパース化 = ノイズ除去

この性質から、重要な帰結が導かれる：

> [!IMPORTANT]
> 高次元空間では、ランダムな次元（無関係な次元）を計算に含めることは、ノイズを増やすことに等しい。

信号対雑音比（SNR, Signal-to-Noise Ratio）の観点から考えよう。意味のある信号が $k$ 次元に存在し、残りの $(d - k)$ 次元がノイズだとする。すべての次元を計算に含めると：

$$\text{SNR} \propto \frac{k}{d-k}$$

$d$ が大きく $k$ が固定なら、SNRは急速に悪化する。したがって、**無関係な次元を計算から除外する**（スパース化）ことは、単なる効率化ではなく、**情報の純度を高めるための幾何学的必然**である。

### スパース化の誤解を解く

「スパース化 = サボり」という誤解がある。しかし、上記の議論から明らかなように、スパース化は**積極的な情報処理戦略**である。

高次元空間において、密（Dense）であることは必ずしも良いことではない。無関係な情報を含めることは、計算コストだけでなく、**ノイズによる性能劣化**を招く。

| 視点 | Dense（密） | Sparse（疎） |
| --- | --- | --- |
| 計算量 | 高い | 低い |
| ノイズ耐性 | 低い（無関係な次元を含む） | 高い（関連次元のみ） |
| 解釈 | 全次元の平等な扱い | 重要次元への集中 |
| 哲学 | 「すべてを見る」 | 「必要なものだけ見る」 |

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

## FlashAttention：物理空間への幾何学的適合

### GPUメモリ階層という「物理空間」

これまで扱ってきた幾何学は、数理的な表現空間の話だった。しかし、実際の計算は**物理デバイス上で実行**される。ここに、もう一つの「幾何学」がある。

現代のGPUは、階層的なメモリ構造を持つ：

```txt
メモリ階層（上に行くほど高速・小容量）:

┌─────────────────┐
│ レジスタ (Registers)    │ ～ KB, 超高速
├─────────────────┤
│ SRAM (Shared Memory)    │ ～ MB, 高速
├─────────────────┤
│ HBM (High Bandwidth Memory) │ ～ GB, 低速
└─────────────────┘
```

Attentionは巨大な行列（シーケンス長 $n$ で $n \times n$ ）を扱うため、通常HBMに格納される。しかし、HBMへのアクセスは遅い。

### FlashAttentionのアイデア：タイリングによるパッキング

**FlashAttention**（Dao et al., 2022）は、Attentionの計算を**小さなタイル（ブロック）に分割**し、各タイルをSRAMに載せて高速計算する手法である。

```txt
通常のAttention:
巨大な行列全体をHBMに保持 → 遅い

FlashAttention:
行列を小さなタイルに分割
  → 各タイルをSRAMに載せる
  → 高速計算
  → 結果を統合
```

数学的には、Softmaxの計算を**オンライン更新**（online softmax）で行うことで、全行列を一度にメモリに載せる必要をなくす。

### 物理空間へのパッキング問題

FlashAttentionの設計は、**数理空間の行列を、物理デバイスのメモリ階層に効率的にパッキング**する問題と見なせる。

これは、数理的な幾何学とは異なる「物理的な幾何学」である。SRAMという「容器の形」に、計算という「荷物」を効率的に詰め込む。

> [!NOTE]
> **物理制約としての幾何学:** アルゴリズム設計は、しばしば物理デバイスの制約に適合させる必要がある。これは「ソフトウェアとハードウェアの共設計」の一例であり、広義には「物理空間への幾何学的適合」と呼べる。ただし、厳密には計算複雑性の問題であり、「幾何学」という用語は比喩的である。

| 側面 | 数理的幾何学 | 物理的「幾何学」 |
| --- | --- | --- |
| 空間 | 表現空間（ $\mathbb{R}^d$ ） | メモリ階層（SRAM/HBM） |
| 最適化目標 | 表現力・汎化性能 | 計算速度・メモリ効率 |
| 設計手法 | 学習アルゴリズム | システム最適化 |
| 例 | Attention, MoE | FlashAttention, 量子化 |

## 結論：知能と抽象化

### 「何を計算しないか」を決める知能

本Appendixで見てきたように、動的剪定の本質は**「何を計算するか」ではなく「何を計算しないか」を決めるプロセス**である。

- **Attention:** 無関係なトークンへの接続を遮断する
- **MoE:** 無関係なExpertを活性化しない
- **Sparse Activation:** 無関係な次元を計算から外す

これは、情報を**捨てる**ことで**シグナルを研ぎ澄ませる**行為である。

### 抽象化としての剪定

認知科学の観点から見ると、「情報を捨ててエッセンスを抽出する」という行為は、**抽象化**（Abstraction）の本質である。

人間は、世界のすべての詳細を記憶しない。重要な特徴だけを抽出し、それを元に推論する。これは、高次元の入力を低次元の表現に**射影**する行為であり、幾何学的には次元削減である。

動的剪定は、この抽象化を**動的に・文脈依存的に**行う。同じ情報でも、文脈が変われば重要度が変わる。これは、人間の注意（Attention）の働きに近い。

> [!NOTE]
> **知能の本質としての選択:** 情報理論の創始者Claude Shannon は、「情報は予測不可能性である」と述べた。しかし、予測不可能性をそのまま保持することは、ノイズを保持することでもある。知能は、予測可能な（冗長な）情報を捨て、予測不可能な（本質的な）情報を保持する能力である。この意味で、剪定は知能の中核的機能と言える。

### 最終的な問い

本講義シリーズに通底する問いに戻ろう：

**「AIとは何か」**

動的剪定の視点から見れば、AIは「高次元空間における適応的な情報選択システム」である。どの情報を保持し、どの情報を捨てるかを、文脈に応じて動的に決定する。

しかし、この選択基準自体は、**学習データと目的関数に依存する**。何を「重要」と見なすかは、設計者と訓練データが決める。

ここに、技術的問題を超えた、価値判断の問題が現れる。本講義では扱わないが、次のステップとして意識すべき視点である。

## 実装ノート

### Attentionの計算量比較

```python
import torch
import time

def standard_attention(Q, K, V):
    """標準的なAttention（メモリ効率悪い）"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attention_weights = torch.softmax(scores, dim=-1)  # O(n^2) のメモリ
    output = torch.matmul(attention_weights, V)
    return output

def sparse_attention_topk(Q, K, V, k=10):
    """Top-Kによるスパース化（概念実装）"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Top-K選択（各クエリについて上位k個のキーのみ保持）
    topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1)
    
    # スパース行列を構築（実際にはより効率的な実装が必要）
    sparse_weights = torch.zeros_like(scores)
    sparse_weights.scatter_(-1, topk_indices, torch.softmax(topk_scores, dim=-1))
    
    output = torch.matmul(sparse_weights, V)
    return output

# ベンチマーク
batch_size, num_heads, seq_len, d_k = 2, 8, 512, 64
Q = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda')
K = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda')
V = torch.randn(batch_size, num_heads, seq_len, d_k, device='cuda')

# 標準Attention
start = time.time()
out1 = standard_attention(Q, K, V)
time1 = time.time() - start

# スパースAttention (k=32)
start = time.time()
out2 = sparse_attention_topk(Q, K, V, k=32)
time2 = time.time() - start

print(f"Standard Attention: {time1:.4f}s")
print(f"Sparse Attention (k=32): {time2:.4f}s")
print(f"Speedup: {time1/time2:.2f}x")
```

> [!WARNING]
> 上記のスパース実装は教育目的の簡略版である。実際の効率化には、専用のカーネル（CUDA実装）や、より洗練されたデータ構造（CSRなど）が必要である。

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
    - ニューラルネットワークの剪定の古典的論文。

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

本Appendixでは、「動的剪定」という視点で、AttentionとMoEを統一的に理解した。

| 概念 | 定義 | 本Appendixでの役割 |
| --- | --- | --- |
| **動的剪定** | 入力に応じて計算経路を決定 | Attention, MoEの共通原理 |
| **ソフト剪定** | 確率的重み付けによる接続の遮断 | Attentionの幾何学的解釈 |
| **部分空間スイッチング** | 入力に応じた部分空間の選択 | MoEの幾何学的解釈 |
| **スパース性の必然** | 高次元でのノイズ除去 | 剪定の情報理論的根拠 |
| **効率化の体系化** | GQA, LoRA, MoD | 剪定のバリエーション |
| **抽象化** | 情報を捨ててエッセンスを抽出 | 知能の本質としての剪定 |

### 講義本編との接続

- **第6回（Attention）:** ミクロな動的剪定として再解釈
- **第13回（MoE）:** マクロな動的剪定として再解釈
- **第2回（高次元の直交性）:** スパース性の数学的根拠
- **第3回（球面）:** 正規化と角度計算の幾何学

### 次のステップ

本Appendixで扱った「剪定」は、**計算資源の効率化**という文脈だった。しかし、同じ原理は**情報の選択**という、より広い文脈にも適用できる。

- Retrieval-Augmented Generation (RAG): 外部知識からの選択的取得
- アライメント: 望ましい行動の選択的強化
- 解釈可能性: 重要な特徴の選択的提示

これらはすべて、「高次元空間からの選択的射影」という幾何学的操作として統一的に理解できる可能性がある。

「何を見て、何を見ないか」を決める能力。それが知能の核心かもしれない。
