import matplotlib.pyplot as plt
import numpy as np
import torch


def kl_divergence(p, q):
    """KL(P||Q) を計算（0除算回避付き）"""
    epsilon = 1e-10
    return (p * torch.log((p + epsilon) / (q + epsilon))).sum()


# 2つの分布を定義
P = torch.tensor([0.7, 0.2, 0.1])
Q_list = []
kl_pq_list = []
kl_qp_list = []

# Qの最初の要素を変化させる
for q0 in np.linspace(0.1, 0.9, 50):
    Q = torch.tensor([q0, (1 - q0) * 0.6, (1 - q0) * 0.4])
    Q_list.append(q0)
    kl_pq_list.append(kl_divergence(P, Q).item())
    kl_qp_list.append(kl_divergence(Q, P).item())

# 可視化
plt.figure(figsize=(10, 6))
plt.plot(Q_list, kl_pq_list, label="KL(P||Q)", linewidth=2)
plt.plot(Q_list, kl_qp_list, label="KL(Q||P)", linewidth=2, linestyle="--")
plt.xlabel("Q[0]")
plt.ylabel("KL Divergence")
plt.title("KL Divergence Asymmetry")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("kl_asymmetry.png", dpi=150, bbox_inches="tight")
print("KLダイバージェンスの非対称性を kl_asymmetry.png に保存")
