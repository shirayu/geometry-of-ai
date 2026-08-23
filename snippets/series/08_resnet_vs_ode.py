import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint

    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False


class ResidualBlock(nn.Module):
    """基本的な残差ブロック（08_residual_block.pyと同じ定義）"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualStack(nn.Module):
    def __init__(self, dim, num_steps):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(num_steps)])

    def forward(self, x, return_trajectory=False):
        trajectory = [x]
        for block in self.blocks:
            x = block(x)
            trajectory.append(x)
        if return_trajectory:
            return x, torch.stack(trajectory)
        return x


class ODEFunc(nn.Module):
    """ODEの右辺 f(h, t)（08_neural_ode.pyと同じ定義）"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t, h):
        return self.net(h)


class NeuralODE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.func = ODEFunc(dim)

    def forward(self, x, t_span, return_trajectory=False):
        trajectory = odeint(self.func, x, t_span, method="dopri5")
        if return_trajectory:
            return trajectory[-1], trajectory
        return trajectory[-1]


def compare_resnet_and_ode(dim=64, num_steps=10):
    """ResNetとNeural ODEの挙動を比較

    ResNet: 離散的なステップ
    Neural ODE: 連続的な流れ（離散化して比較）
    """
    # ResNet
    resnet = ResidualStack(dim, num_steps)

    # Neural ODE（利用可能な場合）
    if TORCHDIFFEQ_AVAILABLE:
        neural_ode = NeuralODE(dim)

    # テスト入力
    x = torch.randn(16, dim)

    # ResNetの軌跡
    _, resnet_traj = resnet(x, return_trajectory=True)

    results = {
        "resnet": {
            "trajectory_shape": resnet_traj.shape,
            "output_norm_mean": resnet_traj[-1].norm(dim=-1).mean().item(),
            "output_norm_std": resnet_traj[-1].norm(dim=-1).std().item(),
        }
    }

    # Neural ODEの軌跡（利用可能な場合）
    if TORCHDIFFEQ_AVAILABLE:
        t_span = torch.linspace(0, 1, num_steps + 1)
        _, ode_traj = neural_ode(x, t_span, return_trajectory=True)

        results["neural_ode"] = {
            "trajectory_shape": ode_traj.shape,
            "output_norm_mean": ode_traj[-1].norm(dim=-1).mean().item(),
            "output_norm_std": ode_traj[-1].norm(dim=-1).std().item(),
        }

    return results


# 比較実行
results = compare_resnet_and_ode()
print("Comparison Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
