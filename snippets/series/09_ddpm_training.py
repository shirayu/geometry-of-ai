import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """線形ノイズスケジュール

    Args:
        timesteps: 総ステップ数
        beta_start: 初期β
        beta_end: 最終β

    Returns:
        betas: [timesteps] のテンソル
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """コサインノイズスケジュール（Improved DDPM）

    Args:
        timesteps: 総ステップ数
        s: オフセット

    Returns:
        betas: [timesteps] のテンソル
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """拡散スケジュールの管理"""

    def __init__(self, timesteps=1000, schedule_type="linear"):
        self.timesteps = timesteps

        if schedule_type == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """Forward process: x_0 から x_t をサンプリング

        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε

        Args:
            x_0: 元データ [batch, ...]
            t: タイムステップ [batch]
            noise: ノイズ（Noneなら生成）

        Returns:
            x_t: ノイズが加わったデータ
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        # 形状を合わせる
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise


def ddpm_loss(model, x_0, schedule, t=None):
    """DDPMの損失関数（ノイズ予測）

    L = E_{t, x_0, ε}[||ε - ε_θ(x_t, t)||²]

    Args:
        model: ノイズ予測ネットワーク
        x_0: 元データ [batch, ...]
        schedule: DiffusionSchedule
        t: タイムステップ（Noneならランダム）

    Returns:
        loss: スカラー
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    # ランダムなタイムステップ
    if t is None:
        t = torch.randint(0, schedule.timesteps, (batch_size,), device=device)

    # ノイズを生成
    noise = torch.randn_like(x_0)

    # x_t を計算
    x_t = schedule.q_sample(x_0, t, noise)

    # ノイズを予測
    predicted_noise = model(x_t, t)

    # MSE損失
    loss = F.mse_loss(predicted_noise, noise)

    return loss


# 簡単なノイズ予測ネットワーク（教育目的）
class SimpleNoisePredictor(nn.Module):
    """簡単なノイズ予測ネットワーク

    実際の実装ではU-Netを使用
    """

    def __init__(self, dim, hidden_dim=256, time_emb_dim=64):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, t):
        """
        Args:
            x: 入力 [batch, dim]
            t: タイムステップ [batch]

        Returns:
            予測されたノイズ [batch, dim]
        """
        # 時刻を正規化して埋め込み
        t_emb = self.time_emb(t.float().unsqueeze(-1) / 1000.0)

        # 連結して予測
        x_t = torch.cat([x, t_emb], dim=-1)
        return self.net(x_t)


# 使用例
if __name__ == "__main__":
    dim = 64
    batch_size = 32
    timesteps = 1000

    schedule = DiffusionSchedule(timesteps, schedule_type="cosine")
    model = SimpleNoisePredictor(dim)

    # ダミーデータ
    x_0 = torch.randn(batch_size, dim)

    # 損失計算
    loss = ddpm_loss(model, x_0, schedule)
    print(f"DDPM Loss: {loss.item():.4f}")

    # Forward processの確認
    t = torch.tensor([0, 250, 500, 750, 999])
    for ti in t:
        x_t = schedule.q_sample(x_0[:1], ti.unsqueeze(0))
        print(f"t={ti.item():4d}: x_t norm = {x_t.norm().item():.4f}")
