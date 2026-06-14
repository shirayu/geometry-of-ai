import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):
    """CosFace: コサインから直接マージンを減算"""

    def __init__(self, in_features, num_classes, scale=30.0, margin=0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cosine = F.linear(x_norm, w_norm)

        # 正解クラスのコサインからマージンを減算
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        logits = cosine - one_hot * self.margin

        return logits * self.scale


class SphereFace(nn.Module):
    """SphereFace: 角度を乗算（教育用簡略版）

    注意: 原論文（A-Softmax）は特徴を正規化しない。本実装は球面設計に揃えた
    教育用簡略版であり、実際のSphereFaceとは定式化が異なる。
    また実際のSphereFaceはより複雑なアニーリング戦略を使用する。
    """

    def __init__(self, in_features, num_classes, scale=30.0, margin=4):
        super().__init__()
        self.scale = scale
        self.margin = margin  # 整数の乗数
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cosine = F.linear(x_norm, w_norm)

        # cos(m * θ) の計算（チェビシェフ多項式を使うのが一般的）
        # ここでは簡略化のため m=2 の場合のみ示す
        # cos(2θ) = 2cos²(θ) - 1
        if self.margin == 2:
            cos_m_theta = 2 * cosine**2 - 1
        else:
            # 一般の場合は acos → *m → cos が必要（数値不安定）
            theta = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
            cos_m_theta = torch.cos(self.margin * theta)

        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        logits = one_hot * cos_m_theta + (1.0 - one_hot) * cosine

        return logits * self.scale
