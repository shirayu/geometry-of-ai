# pip install torchdiffeq
import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint

    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("torchdiffeq not installed. Neural ODE examples will be skipped.")


class ODEFunc(nn.Module):
    """ODEの右辺 f(h, t) を定義するニューラルネットワーク

    dh/dt = f(h, t)
    """

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),  # 有界な活性化関数（滑らかさの促進に寄与しうるが、
            # これだけでリプシッツ連続性が保証されるわけではない）
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t, h):
        """
        Args:
            t: 時刻（スカラー）
            h: 状態 [batch, dim]

        Returns:
            dh/dt: 時間微分 [batch, dim]
        """
        # この実装ではtを使わない（自律系）
        # 時間依存にするなら、tを特徴として連結する
        return self.net(h)


class NeuralODE(nn.Module):
    """Neural ODEモデル"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.func = ODEFunc(dim, hidden_dim)

    def forward(self, x, t_span=None, method="dopri5", return_trajectory=False):
        """
        Args:
            x: 初期状態 h(0) [batch, dim]
            t_span: 積分する時刻のリスト（デフォルト: [0, 1]）
            method: ODEソルバー（'dopri5', 'euler', 'rk4'など）
            return_trajectory: Trueなら全時刻の状態を返す

        Returns:
            output: 最終状態 h(T) [batch, dim]
            trajectory: (オプション) 全時刻の状態 [len(t_span), batch, dim]
        """
        if not TORCHDIFFEQ_AVAILABLE:
            raise RuntimeError("torchdiffeq is required for Neural ODE")

        if t_span is None:
            t_span = torch.tensor([0.0, 1.0])

        # ODEを解く
        trajectory = odeint(self.func, x, t_span, method=method)

        if return_trajectory:
            return trajectory[-1], trajectory
        return trajectory[-1]


# 使用例（torchdiffeqがインストールされている場合）
if TORCHDIFFEQ_AVAILABLE:
    torch.manual_seed(42)
    dim = 64
    model = NeuralODE(dim)

    x = torch.randn(32, dim)
    t_span = torch.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0

    output, trajectory = model(x, t_span, return_trajectory=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trajectory shape: {trajectory.shape}")  # [11, 32, 64]
