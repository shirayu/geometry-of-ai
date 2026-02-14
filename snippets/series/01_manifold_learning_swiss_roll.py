import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

# スイスロールデータ生成
n_samples = 1500
X, color = make_swiss_roll(n_samples, noise=0.1, random_state=42)

# 各手法で次元削減
pca = PCA(n_components=2)
isomap = Isomap(n_components=2, n_neighbors=10)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)

X_pca = pca.fit_transform(X)
X_isomap = isomap.fit_transform(X)
X_lle = lle.fit_transform(X)

# 可視化
fig = plt.figure(figsize=(16, 5))

# 元の3Dデータ
ax1 = fig.add_subplot(1, 4, 1, projection="3d")
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="viridis", s=10)
ax1.set_title("Original Swiss Roll (3D)")
ax1.view_init(10, -70)

# PCA
ax2 = fig.add_subplot(1, 4, 2)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap="viridis", s=10)
ax2.set_title("PCA (線形射影)\n→ 構造が潰れる")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")

# Isomap
ax3 = fig.add_subplot(1, 4, 3)
ax3.scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, cmap="viridis", s=10)
ax3.set_title("Isomap (測地距離保存)\n→ ロールがほどける")
ax3.set_xlabel("Component 1")
ax3.set_ylabel("Component 2")

# LLE
ax4 = fig.add_subplot(1, 4, 4)
ax4.scatter(X_lle[:, 0], X_lle[:, 1], c=color, cmap="viridis", s=10)
ax4.set_title("LLE (局所構造保存)\n→ 局所関係を維持")
ax4.set_xlabel("Component 1")
ax4.set_ylabel("Component 2")

plt.tight_layout()
plt.show()
