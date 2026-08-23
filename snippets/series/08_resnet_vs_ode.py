import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint

    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False


class EulerResidualStack(nn.Module):
    """同じベクトル場をEuler法で積分する残差スタック"""

    def __init__(self, func, num_steps):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_stepsは1以上である必要があります")
        self.func = func
        self.num_steps = num_steps

    def forward(self, x, t0=0.0, t1=1.0, return_trajectory=False):
        dt = (t1 - t0) / self.num_steps
        trajectory = [x]
        for step in range(self.num_steps):
            t = torch.as_tensor(t0 + step * dt, dtype=x.dtype, device=x.device)
            x = x + dt * self.func(t, x)
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
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x, t_span, return_trajectory=False):
        if not TORCHDIFFEQ_AVAILABLE:
            raise RuntimeError("Neural ODEの実行にはtorchdiffeqが必要です")
        trajectory = odeint(self.func, x, t_span, method="dopri5")
        if return_trajectory:
            return trajectory[-1], trajectory
        return trajectory[-1]


def compare_resnet_and_ode(dim=64, num_steps=10):
    """同じベクトル場のEuler残差離散化とNeural ODEを比較

    比較対象を同じ初期値・同じベクトル場・同じ時刻区間に揃え、
    最終状態の差をEuler法の離散化誤差として測る。
    """
    torch.manual_seed(42)
    func = ODEFunc(dim)
    euler_residual = EulerResidualStack(func, num_steps)
    x = torch.randn(16, dim)

    _, euler_traj = euler_residual(x, return_trajectory=True)

    results = {
        "euler_residual": {
            "trajectory_shape": euler_traj.shape,
            "output_norm_mean": euler_traj[-1].norm(dim=-1).mean().item(),
        }
    }

    if TORCHDIFFEQ_AVAILABLE:
        neural_ode = NeuralODE(func)
        t_span = torch.linspace(0, 1, num_steps + 1, dtype=x.dtype, device=x.device)
        _, ode_traj = neural_ode(x, t_span, return_trajectory=True)
        final_l2_error = (euler_traj[-1] - ode_traj[-1]).norm(dim=-1)

        results["ode_reference"] = {
            "trajectory_shape": ode_traj.shape,
            "output_norm_mean": ode_traj[-1].norm(dim=-1).mean().item(),
        }
        results["discretization_error"] = {
            "final_l2_mean": final_l2_error.mean().item(),
            "final_l2_max": final_l2_error.max().item(),
        }
    else:
        results["note"] = {
            "message": "比較には `pip install torchdiffeq` が必要です",
        }

    return results


# 比較実行
results = compare_resnet_and_ode(num_steps=10)
print("Comparison Results:")
for section_name, metrics in results.items():
    print(f"\n{section_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
