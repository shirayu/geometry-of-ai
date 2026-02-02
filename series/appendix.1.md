
# Appendix: 量子化の幾何学

～重みの離散化が多様体に何をするのか～

## 本Appendixの位置づけ

本Appendixは、講義本編（特に第4回補論「離散と連続の界面」、第13回「スパース性の幾何学」）の発展的内容として、**モデルの量子化（Quantization）を幾何学的視点から考察する**ものである。

2024年以降、1-bit/2-bit LLMなど極端な量子化手法が注目を集めており、「重みを極端に離散化すると多様体の形状にどう影響するか」という問いは、実務的にも理論的にも重要性を増している。

## 1. 量子化とは何か

### 1.1 基本概念

量子化とは、ニューラルネットワークの重み（およびアクティベーション）を、より少ないビット数で表現する技術である。

| 精度 | ビット数 | 表現（概算） | 用途 |
| --- | --- | --- | --- |
| FP32 | 32 | 2^32 通りのビットパターン | 学習時の標準 |
| FP16/BF16 | 16 | 2^16 通りのビットパターン | 学習・推論の高速化 |
| INT8 | 8 | 256段階 | 推論の効率化 |
| INT4 | 4 | 16段階 | 軽量化（GPTQ, AWQ等） |
| INT2 | 2 | 4段階 | 極端な軽量化 |
| 1-bit | 1 | 2段階（例: ±1） | BitNet等の研究 |

> 注: 浮動小数点（FP32/FP16/BF16）の「通り数」は**値の一様な刻み**ではない（指数部を持つため間隔は非一様）。また NaN/Inf/非正規化数なども含まれるため、ここでは「ビットパターン数の規模感」として扱う。

### 1.2 なぜ量子化が必要か

- **メモリ削減:** LLMのパラメータ数は数十億〜数千億。FP16→INT4で概ね約4倍のメモリ削減
- **推論高速化（条件付き）:** 量子化により計算・転送のボトルネックが軽くなることで高速化が期待できる
- **エッジデバイス展開:** スマートフォンやIoT機器での実行を可能に
- **環境負荷軽減:** 計算コスト・消費電力の削減

> 速度についての注意: 「整数演算は常に浮動小数点より速い」とは限らない。実際の速度は **(a)メモリ帯域律速か計算律速か**, **(b)GPU/CPU/アクセラレータの対応カーネル**, **(c)Tensor Core等の対応**, **(d)量子化形式（W4A16, W8A8, KV cacheの扱い）** に強く依存する。

## 2. 量子化の幾何学的解釈

### 2.1 重み空間の「格子化」

量子化を幾何学的に見ると、**連続的な重み空間を離散的な格子点に制限する**操作である。

```txt
連続空間（FP32）:    量子化後（INT4）:
  ↑                    ↑
  │  ・               │  ●───●───●───●
  │    ・             │  │   │   │   │
  │  ・  ・           │  ●───●───●───●
  │      ・           │  │   │   │   │
  └──────→           └──────→

  任意の点を取れる      格子点のみ許容
```

**幾何学的含意:**

- 元の重み $w$ は、最近傍の格子点 $\hat{w}$ に「丸められる」
- これは**重み空間における射影**と見なせる
- 量子化誤差 $\epsilon = w - \hat{w}$ は、この射影による「ずれ」

### 2.2 決定境界への影響

ニューラルネットワークの決定境界（入力空間をクラスに分割する曲面）は、重みによって定まる。量子化は決定境界にどう影響するか？

**直感的理解:**

```txt
量子化前（滑らか）:        量子化後（ガタガタ？）:
      ╭──────╮                 ╭─┬──┬─╮
     ╱        ╲               ╱  │  │  ╲
    ╱   A      ╲             ╱ A │  │   ╲
   ╱            ╲           ╱    │  │    ╲
  ╱      ╭───╮   ╲         ╱   ╭─┴──┴─╮   ╲
 ╱      ╱ B  ╲    ╲       ╱   ╱   B   ╲    ╲
╱      ╱      ╲    ╲     ╱   ╱         ╲    ╲
```

**実務上よく観察される傾向（ただし条件付き）:**

- 適切なPTQ（例: GPTQ/AWQ/SmoothQuant 等）を用いると、**4-bit程度でも性能劣化が限定的**なケースが多い
- 背景には **過剰パラメータ化**や**冗長性**、分布の「鈍感さ」（平均的入力では誤差が相殺されやすい）がある

**ただし例外も重要:**

- **1–2 bit級**では表現力制約が強く、劣化が顕著になりやすい
- **稀な入力パターン**、**長文依存**、**微妙な差分判定（ツール呼び出し、厳密な形式出力など）**では、平均性能が保たれても破綻が出ることがある
- 崩れ方は **量子化方式（per-channel / group-wise / スケール推定）** と **キャリブレーションデータ**に強く依存する

### 2.3 多様体仮説との接続

深層学習の**多様体仮説**によれば、高次元の入力データは実際には低次元の多様体上（またはその近傍）に分布している。

**量子化と多様体の関係:**

1. **データ多様体への射影精度:**

   - 量子化されたネットワークも、データ多様体を十分に近似できるか？
   - 多くの設定で「概ねYes」になりうるが、**量子化方式・精度・キャリブレーション**に強く依存する

2. **表現多様体の変形:**

   - 中間層の表現は、ある多様体上に分布しているとみなせる
   - 量子化により、この多様体が「ピクセル化」される（局所的な分解能が落ちる）
   - ただし位相的性質（連結性、穴の数等）が保存されるかは、層・タスク・精度で異なる（「保存されがち」だが保証はない）

3. **第14回（TDA）との接続:**

   - パーシステントホモロジーで量子化前後の表現を比較
   - トポロジー的特徴が十分に保たれていれば、量子化は「安全」と言える可能性がある

### 2.4 球面上の量子化（nGPTとの接続）

第3回で扱ったnGPTのように、表現が球面上に制約されている場合、量子化はどう振る舞うか？

**球面上の格子:**

- 球面上に均等に点を配置する問題は、古典的な**球面符号化問題**
- 高次元球面では、点の配置がさらに難しくなる（「等間隔」が直感通りに作れない）

**具体的な考察:**

- 正規化された表現を量子化すると、ノルム1の制約が崩れる可能性
- 対策1: 量子化後に再正規化
- 対策2: 球面上で直接量子化（研究段階）

```python
# 概念的なコード
x_normalized = F.normalize(x, dim=-1)  # ノルム1に正規化
x_quantized = quantize(x_normalized)   # 量子化
# この時点で ||x_quantized|| ≠ 1 の可能性
x_renormalized = F.normalize(x_quantized, dim=-1)  # 再正規化
```

## 3. 量子化手法の分類と幾何学的特徴

### 3.1 Post-Training Quantization (PTQ)

学習済みモデルを事後的に量子化する手法。

| 手法 | 特徴 | 幾何学的解釈 |
| --- | --- | --- |
| Round-to-Nearest | 最近傍の格子点に丸め | 最短距離射影 |
| GPTQ | 量子化誤差を後続層で補正 | 誤差の伝播を抑える射影 |
| AWQ | 重要な重みを高精度で保持 | 非一様な格子（重要領域は細かく） |
| SmoothQuant | アクティベーションと重みの分散を均等化 | 空間の「座標変換」後に量子化 |

### 3.2 Quantization-Aware Training (QAT)

量子化を考慮しながら学習する手法。

**Straight-Through Estimator (STE) の再登場:**

- 第4回補論で扱ったSTEが、ここでも活躍
- Forward: 量子化された重みを使用
- Backward: 連続的な勾配を伝播
- **幾何学的解釈:** 「見かけは格子点にいるが、勾配は連続空間から来る」

### 3.3 極端な量子化（1-bit / 2-bit）

**BitNet（Microsoft, 2023-2024）:**

- 重みを {-1, +1} の2値に制限（※実装・派生により表現は変わりうる）
- 行列積が加算・減算中心で計算可能
- **幾何学的解釈:**

    - 重み空間が超立方体の頂点に制限される
    - 表現力は大幅に制限されるが、スケールや構造設計で補う

**1.58-bit (Ternary):**

- 重みを {-1, 0, +1} の3値に制限
- 0を許容することでスパース性も獲得
- **MoE（第13回）との接続（比喩）:** 0の重みは「その経路/寄与が消える」と解釈でき、スパース性の理解に繋がる

## 4. 量子化誤差の幾何学的分析

### 4.1 誤差の定量化

量子化誤差を幾何学的に測る方法：

1. **重み空間での距離:**
   $$\epsilon_w = |W - \hat{W}|_F$$
   （Frobenius ノルム）

2. **出力空間での距離:**
   $$\epsilon_y = |f(x; W) - f(x; \hat{W})|_2$$
   （特定の入力に対する出力の違い）

3. **決定境界のHausdorff距離:**
   量子化前後の決定境界がどれだけ「ずれた」かを測る

### 4.2 層ごとの感度

すべての層が量子化に対して同等に頑健ではない：

- **埋め込み層:** 高感度（語彙の区別に直結）
- **Attention層:** 中程度（冗長性がある場合が多い）
- **FFN層:** 低〜中（モデル/方式により異なる）
- **出力層:** 高感度（最終予測に直結）

**Mixed-Precision Quantization:**

- 感度の高い層は高精度、低い層は低精度
- **幾何学的解釈（比喩）:** 多様体の「曲率が高い」部分は細かく、「平坦な」部分は粗く

### 4.3 量子化は「重み」だけでは終わらない（KV cache / 活性）

LLM推論では、実務上しばしば以下が支配的になる：

- **KV cache（key/value のキャッシュ）**: 長文ではメモリ帯域・容量を圧迫し、速度/同時実行数に直結
- **アクティベーション（中間表現）**: 入力に依存して分布が変わるため、量子化の難しさが増す（キャリブレーションの重要性が上がる）

**幾何学的な見方:**

- 重み量子化は「モデル側の格子化」
- 活性量子化やKV量子化は「入力により変動する軌道（表現多様体）の分解能低下」
- したがって、同じW4でも **AやKVの扱い**で体感品質が大きく変わる

## 5. 実装ノート

### 5.1 基本的な量子化（PyTorch・概念例）

```python
import torch

def quantize_tensor_symmetric(x, bits=8, eps=1e-8):
    """対称量子化の概念例（スカラーscale）。
    実務では per-channel / group-wise がよく使われる。
    """
    qmin, qmax = -(2**(bits-1)), 2**(bits-1) - 1
    scale = x.abs().max().clamp_min(eps) / qmax
    x_q = torch.round(x / scale).clamp(qmin, qmax)
    return x_q, scale

def dequantize_tensor(x_q, scale):
    return x_q * scale

# 使用例
weight = torch.randn(768, 768)
weight_q, scale = quantize_tensor_symmetric(weight, bits=4)
weight_approx = dequantize_tensor(weight_q, scale)
print(f"平均絶対誤差: {(weight - weight_approx).abs().mean():.6f}")
```

### 5.2 実用的なライブラリ（例）

```python
# bitsandbytes（4-bit/8-bit量子化）
import bitsandbytes as bnb
linear_4bit = bnb.nn.Linear4bit(in_features, out_features)

# AutoGPTQ（GPTQ量子化）
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized("model-gptq")

# llama.cpp（GGUF形式、CPU推論）
# コマンドライン: ./main -m model.gguf -p "prompt"
```

### 5.3 量子化前後の表現の可視化（注意付き）

> 注意: `get_hidden_states` はモデル実装に依存する（フックで取得する等が必要）。
> また t-SNE は局所構造を強調しやすく、距離の解釈には注意。

```python
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_quantization_effect(hidden_orig, hidden_quant):
    """量子化前後の中間表現（テンソル）を比較する簡易例"""
    combined = torch.cat([hidden_orig, hidden_quant], dim=0)
    tsne = TSNE(n_components=2, perplexity=30)
    embedded = tsne.fit_transform(combined.cpu().numpy())

    n = len(hidden_orig)
    plt.scatter(embedded[:n, 0], embedded[:n, 1], label="Original", alpha=0.5)
    plt.scatter(embedded[n:, 0], embedded[n:, 1], label="Quantized", alpha=0.5)
    plt.legend()
    plt.title("Representation Space: Original vs Quantized")
    plt.show()
```

## 6. 未解決問題と研究フロンティア

### 6.1 理論的問い

1. **量子化の限界はどこか？**

   - 何bitまで下げても性能を維持できるか？
   - 「平均性能」だけでなく「稀なケース」の破綻をどう扱うか？

2. **最適な量子化格子とは？**

   - 一様格子 vs 非一様格子
   - データ分布に適応した格子の設計
   - 幾何（球面/双曲/混合曲率）に整合した格子は作れるか？

3. **量子化とスパース性の関係**

   - MoEの各Expertを異なる精度で量子化することは有効か？
   - スパース活性化と量子化の相乗効果

### 6.2 実用的課題

1. **キャリブレーションデータの選択**

   - どのデータで量子化パラメータを決定するか
   - ドメイン外データへの汎化

2. **推論時の動的量子化**

   - 入力に応じて精度を変える
   - 「簡単な」入力は低精度、「難しい」入力は高精度
   - ただしスループット/レイテンシの目標次第で有利不利が変わる

3. **学習との統合**

   - QATのさらなる改善
   - 量子化を前提としたアーキテクチャ設計
   - KV cache / 活性まで含めた最適化

## 7. 講義本編との接続まとめ

| 講義回 | 接続点 |
| --- | --- |
| 第2回（ノルムの呪い） | 量子化誤差とノルムの関係 |
| 第3回（球面多様体） | 正規化表現の量子化、再正規化の必要性 |
| 第4回補論（離散と連続） | STE、Gumbel-Softmaxとの類似性 |
| 第7回（不確実性） | 量子化による予測の不確実性増大（特に稀なケース） |
| 第13回（スパース性/MoE） | スパース活性化と量子化の組み合わせ |
| 第14回（TDA） | 量子化前後のトポロジー保存の分析 |

## 参考文献

### 基礎

- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018)
- Nagel et al., "A White Paper on Neural Network Quantization" (arXiv 2021)

### LLM量子化

- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (ICLR 2023)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (MLSys 2024)
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (ICML 2023)

### 極端な量子化

- Wang et al., "BitNet: Scaling 1-bit Transformers for Large Language Models" (arXiv 2023)
- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (arXiv 2024)

### 理論的分析

- Hubara et al., "Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations" (JMLR 2018)
- Banner et al., "Post Training 4-bit Quantization of Convolutional Networks for Rapid-Deployment" (NeurIPS 2019)
g
