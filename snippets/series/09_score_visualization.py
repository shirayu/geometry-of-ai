import matplotlib.pyplot as plt
import numpy as np
import torch


def gaussian_score(x, mean=0.0, std=1.0):
    """ガウス分布のスコア（解析解）

    p(x) = N(mean, std²)
    ∇_x log p(x) = -(x - mean) / std²

    Args:
        x: 入力
        mean: 平均
        std: 標準偏差

    Returns:
        score: スコア関数の値
    """
    return -(x - mean) / (std**2)


def mixture_score(x, means, stds, weights):
    """混合ガウス分布のスコア

    注意: log_prob の計算では正規化定数（2π等）を一部省略している。
    スコア（勾配）自体は定数項の影響を受けないため問題ないが、
    表示される log_prob の絶対値は厳密な対数確率密度ではなく「比例」した値である。

    Args:
        x: 入力 [batch, dim]
        means: 各成分の平均 [K, dim]
        stds: 各成分の標準偏差 [K]
        weights: 混合係数 [K]

    Returns:
        score: スコア関数の値 [batch, dim]
    """
    K = len(means)

    # 各成分の確率密度
    log_probs = []
    for k in range(K):
        diff = x - means[k]
        log_prob = -0.5 * (diff**2).sum(dim=-1) / (stds[k] ** 2)
        log_prob -= x.shape[-1] * np.log(stds[k])
        log_prob += np.log(weights[k])
        log_probs.append(log_prob)

    log_probs = torch.stack(log_probs, dim=-1)  # [batch, K]

    # Softmax重み
    probs = torch.softmax(log_probs, dim=-1)  # [batch, K]

    # 各成分のスコアを重み付き平均
    score = torch.zeros_like(x)
    for k in range(K):
        score += probs[:, k : k + 1] * (-(x - means[k]) / (stds[k] ** 2))

    return score


def visualize_score_field_2d():
    """2次元でのスコア場の可視化"""
    # 2成分混合ガウス
    means = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    stds = torch.tensor([0.8, 0.8])
    weights = torch.tensor([0.5, 0.5])

    # グリッドを作成
    x_range = torch.linspace(-5, 5, 20)
    y_range = torch.linspace(-3, 3, 15)
    X, Y = torch.meshgrid(x_range, y_range, indexing="xy")

    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    # スコアを計算
    scores = mixture_score(points, means, stds, weights)
    U = scores[:, 0].reshape(X.shape)
    V = scores[:, 1].reshape(X.shape)

    # 確率密度も計算（可視化用）
    def mixture_density(x, means, stds, weights):
        density = torch.zeros(x.shape[0])
        for k in range(len(means)):
            diff = x - means[k]
            density += (
                weights[k] * torch.exp(-0.5 * (diff**2).sum(dim=-1) / (stds[k] ** 2)) / (2 * np.pi * stds[k] ** 2)
            )
        return density

    Z = mixture_density(points, means, stds, weights).reshape(X.shape)

    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 確率密度
    ax1 = axes[0]
    contour = ax1.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=20, cmap="viridis")
    plt.colorbar(contour, ax=ax1, label="p(x)")
    ax1.scatter(means[:, 0], means[:, 1], c="red", s=100, marker="x", label="Modes")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_title("Probability Density p(x)")
    ax1.legend()

    # 右: スコア場（ベクトル場）
    ax2 = axes[1]
    ax2.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=10, colors="gray", alpha=0.5)
    ax2.quiver(
        X.numpy(),
        Y.numpy(),
        U.numpy(),
        V.numpy(),
        color="blue",
        alpha=0.7,
        scale=50,
    )
    ax2.scatter(means[:, 0], means[:, 1], c="red", s=100, marker="x", label="Modes")
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.set_title("Score Field ∇ log p(x)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("score_field_2d.png", dpi=150)
    plt.close()

    print("Saved: score_field_2d.png")


def visualize_langevin_trajectory():
    """Langevin動力学の軌跡を可視化"""
    # 2成分混合ガウス
    means = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    stds = torch.tensor([0.8, 0.8])
    weights = torch.tensor([0.5, 0.5])

    # Langevinサンプリング
    def langevin_sample(x_init, score_fn, steps=1000, step_size=0.01):
        x = x_init.clone()
        trajectory = [x.clone()]

        for _ in range(steps):
            score = score_fn(x)
            noise = torch.randn_like(x)
            x = x + step_size * score + np.sqrt(2 * step_size) * noise
            trajectory.append(x.clone())

        return torch.stack(trajectory)

    # 複数の軌跡をサンプリング
    n_samples = 5
    x_init = torch.randn(n_samples, 2) * 3

    def score_fn(x):
        return mixture_score(x, means, stds, weights)

    trajectories = langevin_sample(x_init, score_fn, steps=500, step_size=0.05)

    # プロット
    fig, ax = plt.subplots(figsize=(8, 6))

    # 密度の等高線
    x_range = torch.linspace(-5, 5, 50)
    y_range = torch.linspace(-4, 4, 40)
    X, Y = torch.meshgrid(x_range, y_range, indexing="xy")
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    def mixture_density(x, means, stds, weights):
        density = torch.zeros(x.shape[0])
        for k in range(len(means)):
            diff = x - means[k]
            density += (
                weights[k] * torch.exp(-0.5 * (diff**2).sum(dim=-1) / (stds[k] ** 2)) / (2 * np.pi * stds[k] ** 2)
            )
        return density

    Z = mixture_density(points, means, stds, weights).reshape(X.shape)
    ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=10, colors="gray", alpha=0.5)

    # 軌跡をプロット
    colors = plt.cm.tab10(np.linspace(0, 1, n_samples))
    for i in range(n_samples):
        traj = trajectories[:, i].numpy()
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.7, linewidth=0.5)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker="o", s=50, label=f"Start {i + 1}")
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], marker="x", s=50)

    ax.scatter(means[:, 0], means[:, 1], c="red", s=200, marker="*", zorder=5, label="Modes")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("Langevin Dynamics Trajectories")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("langevin_trajectory.png", dpi=150)
    plt.close()

    print("Saved: langevin_trajectory.png")


# 実行
if __name__ == "__main__":
    visualize_score_field_2d()
    visualize_langevin_trajectory()
