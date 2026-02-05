import torch
import torch.nn as nn


class DiffusionSchedule:
    """拡散スケジュール（簡略版）"""

    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)


class SimpleNoisePredictor(nn.Module):
    """簡易ノイズ予測器"""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 64, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, dim)
        )
        self.time_emb = nn.Linear(1, 64)

    def forward(self, x, t):
        t_emb = self.time_emb(t.float().unsqueeze(-1) / 1000.0)
        return self.net(torch.cat([x, t_emb], dim=-1))


@torch.no_grad()
def ddim_sample(model, schedule, shape, steps=50, eta=0.0, device="cpu"):
    """DDIMサンプリング（概念実装）

    Args:
        model: ノイズ予測ネットワーク
        schedule: DiffusionSchedule
        shape: 出力形状 (batch, dim)
        steps: サンプリングステップ数
        eta: 確率性の制御（0=決定論的ODE、1=確率的SDE）
        device: デバイス

    Returns:
        x_0: 生成されたサンプル
    """
    # サンプリングする時刻のリスト
    timesteps = torch.linspace(schedule.timesteps - 1, 0, steps + 1).long()

    # 純粋なノイズから開始
    x = torch.randn(shape, device=device)

    alphas_cumprod = schedule.alphas_cumprod.to(device)

    for i in range(steps):
        t = timesteps[i].item()
        t_next = timesteps[i + 1].item()

        # 現在のα
        alpha_t = alphas_cumprod[int(t)]
        # t_next < 0 の場合は alpha = 1（完全にデノイズされた状態）
        if t_next >= 0:
            alpha_t_next = alphas_cumprod[int(t_next)]
        else:
            alpha_t_next = torch.tensor(1.0, device=device, dtype=alpha_t.dtype)

        # ノイズを予測
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_tensor)

        # x_0 を予測
        x0_pred = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)

        # ノイズの分散を計算（eta=0で決定論的）
        sigma_t = (
            eta
            * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t + 1e-8))
            * torch.sqrt(1 - alpha_t / (alpha_t_next + 1e-8))
        )

        # x_{t-1} を計算
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_t_next - sigma_t**2, min=0)) * eps_pred

        # sigma_t が実質ゼロかどうかを float で判定
        sigma_val = sigma_t.item() if sigma_t.numel() == 1 else sigma_t.mean().item()
        if sigma_val > 1e-8:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = torch.sqrt(alpha_t_next) * x0_pred + dir_xt + sigma_t * noise

    return x


# スコア関数としての解釈
def noise_to_score(eps_pred, sqrt_one_minus_alpha):
    """ノイズ予測からスコアへの変換

    ∇_x log p_t(x) ≈ -ε / sqrt(1 - α̅_t)

    Args:
        eps_pred: 予測されたノイズ
        sqrt_one_minus_alpha: sqrt(1 - α̅_t)

    Returns:
        score: スコア関数の近似
    """
    return -eps_pred / sqrt_one_minus_alpha


# 使用例
if __name__ == "__main__":
    dim = 64
    batch_size = 8
    timesteps = 1000

    schedule = DiffusionSchedule(timesteps)
    model = SimpleNoisePredictor(dim)

    # DDIMサンプリング（決定論的、eta=0）
    samples_deterministic = ddim_sample(model, schedule, (batch_size, dim), steps=50, eta=0.0)

    # DDIMサンプリング（確率的、eta=1）
    samples_stochastic = ddim_sample(model, schedule, (batch_size, dim), steps=50, eta=1.0)

    print(f"Deterministic samples shape: {samples_deterministic.shape}")
    print(f"Deterministic samples norm: {samples_deterministic.norm(dim=-1).mean():.4f}")

    print(f"Stochastic samples shape: {samples_stochastic.shape}")
    print(f"Stochastic samples norm: {samples_stochastic.norm(dim=-1).mean():.4f}")
