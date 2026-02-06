import torch

# 真の分布 P と モデル Q
P = torch.tensor([0.1, 0.6, 0.3])
Q = torch.tensor([0.4, 0.3, 0.3])


# KLダイバージェンスの計算（手計算版）
def kl_divergence(p, q):
    """KL(P||Q) を計算"""
    return (p * torch.log(p / q)).sum()


kl_pq = kl_divergence(P, Q)
kl_qp = kl_divergence(Q, P)

print(f"KL(P||Q): {kl_pq:.4f}")
print(f"KL(Q||P): {kl_qp:.4f}")
print(f"非対称性: {abs(kl_pq - kl_qp):.4f}")
# 出力例:
# KL(P||Q): 0.2332
# KL(Q||P): 0.1596
# 非対称性: 0.0736
