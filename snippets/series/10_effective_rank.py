import numpy as np
import torch


def compute_effective_rank(embeddings, eps=1e-10):
    """埋め込み行列の有効ランクを計算

    有効ランク = exp(entropy of normalized singular values)

    Args:
        embeddings: 埋め込み行列 [n_samples, dim]
        eps: 数値安定性のための小さな値

    Returns:
        effective_rank: 有効ランク（スカラー）
        singular_values: 特異値の配列
    """
    # 中心化（平均を引く）
    embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)

    # SVD
    U, S, Vh = torch.linalg.svd(embeddings_centered, full_matrices=False)

    # 特異値を正規化（確率分布に）
    S_normalized = S / (S.sum() + eps)

    # エントロピー計算（0を避けるためにeps追加）
    entropy = -(S_normalized * torch.log(S_normalized + eps)).sum()

    # 有効ランク = exp(エントロピー)
    effective_rank = torch.exp(entropy)

    return effective_rank.item(), S.numpy()


def analyze_collapse(embeddings, threshold_ratio=0.9):
    """Dimensional Collapseの分析

    Args:
        embeddings: 埋め込み行列 [n_samples, dim]
        threshold_ratio: 累積寄与率の閾値

    Returns:
        analysis: 分析結果の辞書
    """
    eff_rank, singular_values = compute_effective_rank(embeddings)

    # 累積寄与率
    total_variance = (singular_values**2).sum()
    cumulative_ratio = np.cumsum(singular_values**2) / total_variance

    # 閾値を超える最小の次元数
    dims_for_threshold = np.searchsorted(cumulative_ratio, threshold_ratio) + 1

    # 最大特異値の支配度
    dominance_ratio = singular_values[0] / singular_values.sum()

    analysis = {
        "effective_rank": eff_rank,
        "nominal_rank": len(singular_values),
        "rank_ratio": eff_rank / len(singular_values),
        "dims_for_90_percent": dims_for_threshold,
        "top_singular_dominance": dominance_ratio,
        "singular_values": singular_values,
        "cumulative_ratio": cumulative_ratio,
    }

    return analysis


def diagnose_collapse(analysis):
    """Collapseの診断メッセージを生成

    Args:
        analysis: analyze_collapse の出力

    Returns:
        diagnosis: 診断メッセージ
    """
    rank_ratio = analysis["rank_ratio"]
    dominance = analysis["top_singular_dominance"]

    if rank_ratio < 0.1:
        severity = "重度"
    elif rank_ratio < 0.3:
        severity = "中程度"
    elif rank_ratio < 0.5:
        severity = "軽度"
    else:
        severity = "なし"

    diagnosis = f"""
Dimensional Collapse 診断結果:
================================
有効ランク: {analysis["effective_rank"]:.1f} / {analysis["nominal_rank"]} 次元
ランク比率: {rank_ratio:.2%}
90%分散に必要な次元: {analysis["dims_for_90_percent"]}
最大特異値の支配度: {dominance:.2%}

診断: Collapse {severity}
"""

    if severity != "なし":
        diagnosis += """
推奨対策:
- 正則化の見直し（weight decay, dropout）
- 対照学習の導入
- バッチ正規化の確認
"""

    return diagnosis


# 使用例
if __name__ == "__main__":
    # 健全な埋め込み（多次元を使用）
    torch.manual_seed(42)
    healthy_embeddings = torch.randn(1000, 256)

    # Collapseした埋め込み（低次元に集中）
    collapsed_base = torch.randn(1000, 10)
    padding = torch.zeros(1000, 246)
    collapsed_embeddings = torch.cat([collapsed_base, padding], dim=1)
    # 少しノイズを追加
    collapsed_embeddings += 0.01 * torch.randn_like(collapsed_embeddings)

    print("=== 健全な埋め込み ===")
    analysis_healthy = analyze_collapse(healthy_embeddings)
    print(diagnose_collapse(analysis_healthy))

    print("\n=== Collapseした埋め込み ===")
    analysis_collapsed = analyze_collapse(collapsed_embeddings)
    print(diagnose_collapse(analysis_collapsed))
