import torch
import torch.nn.functional as F


def compute_modality_gap(text_emb, image_emb):
    """モダリティギャップを計算

    Args:
        text_emb: テキスト埋め込み [n_samples, dim]
        image_emb: 画像埋め込み [n_samples, dim]

    Returns:
        ギャップベクトルとその大きさ
    """
    # L2正規化
    text_emb = F.normalize(text_emb, dim=-1)
    image_emb = F.normalize(image_emb, dim=-1)

    # 各モダリティの重心
    text_centroid = text_emb.mean(dim=0)
    image_centroid = image_emb.mean(dim=0)

    # ギャップベクトル（重心間の差）
    gap_vector = text_centroid - image_centroid
    gap_magnitude = gap_vector.norm().item()

    # 重心を正規化して角度も計算
    text_centroid_norm = F.normalize(text_centroid, dim=0)
    image_centroid_norm = F.normalize(image_centroid, dim=0)
    cos_angle = (text_centroid_norm @ image_centroid_norm).item()
    angle_deg = torch.acos(torch.tensor(cos_angle)).item() * 180 / 3.14159

    return {
        "gap_vector": gap_vector,
        "gap_magnitude": gap_magnitude,
        "centroid_angle_deg": angle_deg,
    }


def analyze_intra_inter_similarity(text_emb, image_emb):
    """モダリティ内・モダリティ間の類似度を分析

    Args:
        text_emb: テキスト埋め込み [n_samples, dim]
        image_emb: 画像埋め込み [n_samples, dim]

    Returns:
        各種統計量
    """
    text_emb = F.normalize(text_emb, dim=-1)
    image_emb = F.normalize(image_emb, dim=-1)

    # モダリティ内類似度
    text_sim = text_emb @ text_emb.T
    image_sim = image_emb @ image_emb.T

    # 対角成分を除いた平均（自分自身との類似度を除く）
    n = text_emb.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=text_emb.device)

    intra_text = text_sim[mask].mean().item()
    intra_image = image_sim[mask].mean().item()

    # モダリティ間類似度
    cross_sim = text_emb @ image_emb.T
    inter_matched = cross_sim.diag().mean().item()  # 対応するペア
    inter_unmatched = cross_sim[mask].mean().item()  # 対応しないペア

    return {
        "intra_text_similarity": intra_text,
        "intra_image_similarity": intra_image,
        "inter_matched_similarity": inter_matched,
        "inter_unmatched_similarity": inter_unmatched,
    }


# 使用例（擬似的なデータで）
n_samples, dim = 100, 512

# 実際のCLIPでは、テキストと画像は異なる領域に分布する
# ここでは、その様子を模擬的に再現
text_emb = torch.randn(n_samples, dim) + torch.tensor([1.0] + [0.0] * (dim - 1))
image_emb = torch.randn(n_samples, dim) + torch.tensor([-1.0] + [0.0] * (dim - 1))

gap_info = compute_modality_gap(text_emb, image_emb)
print(f"Modality gap magnitude: {gap_info['gap_magnitude']:.4f}")
print(f"Centroid angle: {gap_info['centroid_angle_deg']:.1f} degrees")

sim_info = analyze_intra_inter_similarity(text_emb, image_emb)
print(f"Intra-text similarity: {sim_info['intra_text_similarity']:.4f}")
print(f"Intra-image similarity: {sim_info['intra_image_similarity']:.4f}")
print(f"Inter-matched similarity: {sim_info['inter_matched_similarity']:.4f}")
print(f"Inter-unmatched similarity: {sim_info['inter_unmatched_similarity']:.4f}")
