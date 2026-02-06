import torch


def dot_product(u, v):
    """内積の計算（標準的なAttentionに相当）"""
    return torch.dot(u, v)


def cosine_similarity(u, v):
    """コサイン類似度の計算（正規化後の内積）"""
    u_norm = u / u.norm()
    v_norm = v / v.norm()
    return torch.dot(u_norm, v_norm)


# 例：同じ方向だが大きさが異なるベクトル
u = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([2.0, 4.0, 6.0])  # uの2倍

print(f"内積: {dot_product(u, v):.2f}")
print(f"コサイン類似度: {cosine_similarity(u, v):.4f}")
# 内積: 28.00  ← 大きさも反映される
# コサイン類似度: 1.0000  ← 完全に同じ方向（大きさは無視）
