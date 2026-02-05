"""Model Collapseのシミュレーション

注意: これは概念的なデモであり、実際のLLMでのModel Collapseとは
単純化された類似物である。
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class SimpleGenerator(nn.Module):
    """シンプルな生成モデル（デモ用）"""

    def __init__(self, latent_dim=2, output_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)

    def generate(self, n_samples, device="cpu"):
        z = torch.randn(n_samples, 2, device=device)
        return self.forward(z)


def train_generator(model, data, epochs=100, lr=0.01):
    """生成モデルを訓練（MSE損失で簡略化）"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _epoch in range(epochs):
        # ランダムな潜在ベクトル
        z = torch.randn(len(data), 2)
        generated = model(z)

        # 最近傍へのMSE（簡略化された損失）
        # 実際のGANやVAEとは異なる
        loss = ((generated.unsqueeze(1) - data.unsqueeze(0)) ** 2).sum(dim=-1).min(dim=1)[0].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def simulate_model_collapse(initial_data, n_generations=5, samples_per_gen=500):
    """Model Collapseをシミュレート

    Args:
        initial_data: 初期の実データ [n_samples, dim]
        n_generations: 世代数
        samples_per_gen: 各世代で生成するサンプル数

    Returns:
        history: 各世代のデータと統計
    """
    history = []
    current_data = initial_data.clone()

    for gen in range(n_generations):
        # 統計を記録
        mean = current_data.mean(dim=0)
        std = current_data.std(dim=0)
        cov = torch.cov(current_data.T)
        eigenvalues = torch.linalg.eigvalsh(cov)

        history.append(
            {
                "generation": gen,
                "data": current_data.clone(),
                "mean": mean.numpy(),
                "std": std.numpy(),
                "eigenvalues": eigenvalues.numpy(),
                "effective_dim": (eigenvalues > 0.01 * eigenvalues.max()).sum().item(),
            }
        )

        # 新しいモデルを作成して訓練
        model = SimpleGenerator()
        model = train_generator(model, current_data)

        # 次世代のデータを生成
        with torch.no_grad():
            current_data = model.generate(samples_per_gen)

    return history


def visualize_collapse(history):
    """Model Collapseの可視化"""
    n_generations = len(history)

    fig, axes = plt.subplots(2, n_generations, figsize=(4 * n_generations, 8))

    for i, gen_data in enumerate(history):
        # 上段：データ分布
        ax1 = axes[0, i]
        data = gen_data["data"].numpy()
        ax1.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(-4, 4)
        ax1.set_title(f"Generation {gen_data['generation']}")
        ax1.set_aspect("equal")

        # 下段：固有値分布
        ax2 = axes[1, i]
        eigenvalues = gen_data["eigenvalues"]
        ax2.bar(range(len(eigenvalues)), eigenvalues)
        ax2.set_xlabel("Component")
        ax2.set_ylabel("Eigenvalue")
        ax2.set_title(f"Eff. dim: {gen_data['effective_dim']}")

    plt.tight_layout()
    plt.savefig("model_collapse_simulation.png", dpi=150)
    plt.close()

    print("Saved: model_collapse_simulation.png")


# 使用例
if __name__ == "__main__":
    # 初期データ：2つのクラスタを持つ分布
    torch.manual_seed(42)
    n_samples = 500
    cluster1 = torch.randn(n_samples // 2, 2) * 0.5 + torch.tensor([-1.5, 0.0])
    cluster2 = torch.randn(n_samples // 2, 2) * 0.5 + torch.tensor([1.5, 0.0])
    initial_data = torch.cat([cluster1, cluster2], dim=0)

    # シミュレーション実行
    history = simulate_model_collapse(initial_data, n_generations=5)

    # 可視化
    visualize_collapse(history)

    # 統計の表示
    print("\nModel Collapse Simulation Results:")
    print("=" * 50)
    for gen_data in history:
        print(
            f"Generation {gen_data['generation']}: "
            f"std = {gen_data['std'].mean():.3f}, "
            f"effective_dim = {gen_data['effective_dim']}"
        )
