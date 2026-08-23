import torch
import torch.nn as nn
import torch.nn.functional as F


class vMFLayer(nn.Module):
    """vMF分布のパラメータを出力する層

    入力から平均方向μと集中度κを推定する。
    """

    def __init__(self, input_dim, output_dim, kappa_min=1.0, kappa_max=100.0):
        super().__init__()
        self.output_dim = output_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

        # μを出力する層（正規化前）
        self.mu_layer = nn.Linear(input_dim, output_dim)

        # κの生の出力を出す層（スカラー）
        self.kappa_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Args:
            x: 入力テンソル [batch, input_dim]

        Returns:
            mu: 平均方向 [batch, output_dim]（単位ベクトル）
            kappa: 集中度 [batch, 1]
        """
        # 平均方向（正規化して単位ベクトルに）
        mu = self.mu_layer(x)
        mu = F.normalize(mu, dim=-1)

        # 集中度（正の値に制約）
        kappa_raw = self.kappa_layer(x)
        # Softplusで正の値に、さらに範囲を制限
        kappa = F.softplus(kappa_raw)
        kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * torch.sigmoid(kappa - 5)

        return mu, kappa
