import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """基本的な残差ブロック

    h_{t+1} = h_t + f(h_t)

    ここで f は2層のMLPとする。
    """

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        # 出力を小さく初期化（学習初期の安定性のため）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # 残差接続: x + f(x)
        return x + self.net(x)


class ResidualStack(nn.Module):
    """残差ブロックを積み重ねたネットワーク

    「深さ」を「時間」として解釈できる。
    """

    def __init__(self, dim, num_blocks, hidden_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(dim, hidden_dim) for _ in range(num_blocks)])

    def forward(self, x, return_trajectory=False):
        """
        Args:
            x: 入力 [batch, dim]
            return_trajectory: Trueなら中間状態も返す

        Returns:
            output: 出力 [batch, dim]
            trajectory: (オプション) 軌跡 [num_blocks+1, batch, dim]
        """
        if return_trajectory:
            trajectory = [x]

        for block in self.blocks:
            x = block(x)
            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return x, torch.stack(trajectory)
        return x


# 使用例
torch.manual_seed(42)
dim, num_blocks = 64, 8
model = ResidualStack(dim, num_blocks)

x = torch.randn(32, dim)  # バッチサイズ32
output, trajectory = model(x, return_trajectory=True)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Trajectory shape: {trajectory.shape}")  # [9, 32, 64]
