import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# 擬似的なWord2Vecベクトル（実際のWord2Vecモデルがあれば置き換え可能）
# 高頻度語ほど更新回数が多く、ノルムが大きくなる傾向をシミュレート

n_words = 1000
dim = 100

# Zipf則に従う頻度
ranks = np.arange(1, n_words + 1)
frequencies = 1.0 / ranks  # Zipf則: f ∝ 1/rank
frequencies = frequencies / frequencies.sum()

# 頻度に比例した「更新回数」でノルムが成長すると仮定
# （これは単純化したモデルであり、実際のWord2Vecとは異なる）
norms = np.sqrt(frequencies * 10000)  # 単純化モデル：更新回数に比例してノルム成長
# 実際には有意性・情報量がノルムを左右する（本文参照）

# 散布図
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(np.log10(frequencies), norms, alpha=0.5, s=10)
plt.xlabel("log10(Frequency)")
plt.ylabel("Norm (simulated)")
plt.title("頻度とノルムの関係（シミュレーション）")

plt.subplot(1, 2, 2)
plt.scatter(np.log10(ranks), norms, alpha=0.5, s=10)
plt.xlabel("log10(Rank)")
plt.ylabel("Norm (simulated)")
plt.title("順位とノルムの関係（シミュレーション）")

plt.tight_layout()
plt.show()

# 注意：これは概念的なシミュレーションです。
# 実際のWord2Vecベクトルでの検証には、gensimなどのライブラリを使用してください。
