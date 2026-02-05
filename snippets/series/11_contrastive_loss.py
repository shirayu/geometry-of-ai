import torch
import torch.nn.functional as F


def cosine_similarity_matrix(text_emb, image_emb):
    """テキストと画像の埋め込み間のコサイン類似度行列を計算

    Args:
        text_emb: テキスト埋め込み [batch_size, dim]
        image_emb: 画像埋め込み [batch_size, dim]

    Returns:
        類似度行列 [batch_size, batch_size]
    """
    # L2正規化
    text_emb = F.normalize(text_emb, dim=-1)
    image_emb = F.normalize(image_emb, dim=-1)

    # コサイン類似度行列（正規化後は内積＝コサイン類似度）
    return text_emb @ image_emb.T


def clip_loss(text_emb, image_emb, temperature=0.07):
    """CLIP形式の対照学習損失

    Args:
        text_emb: テキスト埋め込み [batch_size, dim]
        image_emb: 画像埋め込み [batch_size, dim]
        temperature: 温度パラメータ

    Returns:
        損失値（スカラー）
    """
    # 類似度行列
    logits = cosine_similarity_matrix(text_emb, image_emb) / temperature

    # ラベル（対角成分が正例）
    batch_size = text_emb.size(0)
    labels = torch.arange(batch_size, device=text_emb.device)

    # 対称な損失（text→image と image→text の両方向）
    loss_t2i = F.cross_entropy(logits, labels)
    loss_i2t = F.cross_entropy(logits.T, labels)

    return (loss_t2i + loss_i2t) / 2


# 使用例
batch_size, dim = 32, 512
text_emb = torch.randn(batch_size, dim)
image_emb = torch.randn(batch_size, dim)

loss = clip_loss(text_emb, image_emb)
print(f"CLIP Loss: {loss.item():.4f}")

# 類似度行列の可視化用
sim_matrix = cosine_similarity_matrix(text_emb, image_emb)
print(f"Similarity matrix shape: {sim_matrix.shape}")
print(f"Diagonal (positive pairs) mean: {sim_matrix.diag().mean().item():.4f}")
print(
    "Off-diagonal (negative pairs) mean:"
    f" {(sim_matrix.sum() - sim_matrix.diag().sum()).item() / (batch_size * (batch_size - 1)):.4f}"
)
