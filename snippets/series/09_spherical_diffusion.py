"""球面上の拡散モデル（概念的なデモ）

!!!!! 重要な注意 !!!!!
このコードは教育目的の「直感的なデモ」であり、
**真の球面ブラウン運動の正しい離散化ではない**。

問題点:
1. 「接空間でノイズ→射影」は、真の球面上SDE離散化と一致しない
2. この近似は統計的バイアスを生む（特に大きな dt や高次元で顕著）
3. 正しい実装には、指数写像（exponential map）や測地線に沿った移動、
   または heat kernel の厳密な計算が必要

位置づけ:
- 射影ベースの近似も、非常に小さい dt かつ低次元であれば、
  「球面上でも拡散的な現象が起きる」という直感を得る目的には使える
- ただし、理論的な整合や定量的な精度は保証されない

研究・実用には:
- 専門文献（Riemannian Score-based Generative Models 等）を参照
- 専用ライブラリ（geomstats, geoopt 等）の使用を検討すること
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def project_to_sphere(x):
    """ベクトルを単位球面に射影

    Args:
        x: 入力ベクトル [batch, dim]

    Returns:
        x_proj: 単位球面上のベクトル [batch, dim]
    """
    return F.normalize(x, dim=-1)


def spherical_brownian_step(x, dt, temperature=1.0):
    """球面上のブラウン運動の1ステップ（非常に粗い近似）

    接空間でガウスノイズを加え、球面に射影し直す。
    ※ これは真の球面ブラウン運動の離散化ではなく、
       直感的なデモのための簡易実装である。

    Args:
        x: 現在の位置（単位球面上） [batch, dim]
        dt: 時間刻み
        temperature: ノイズの強度

    Returns:
        x_new: 次の位置（単位球面上）
    """
    # 接空間でのノイズ（xに直交する成分のみ）
    noise = torch.randn_like(x)
    # xに平行な成分を除去
    noise = noise - (noise * x).sum(dim=-1, keepdim=True) * x

    # 移動して射影
    x_new = x + np.sqrt(2 * temperature * dt) * noise
    return project_to_sphere(x_new)


def spherical_forward_process(x_0, timesteps, dt=0.01):
    """球面上のForward Process

    Args:
        x_0: 初期データ（単位球面上） [batch, dim]
        timesteps: ステップ数
        dt: 時間刻み

    Returns:
        trajectory: 軌跡 [timesteps+1, batch, dim]
    """
    x = x_0.clone()
    trajectory = [x.clone()]

    for _ in range(timesteps):
        x = spherical_brownian_step(x, dt)
        trajectory.append(x.clone())

    return torch.stack(trajectory)


def estimate_vMF_concentration(x_samples):
    """サンプルからvMF分布の集中度を推定（粗い近似）

    注意: この推定式は近似であり、次元・サンプル数・集中度によって
    精度が大きく揺れる。厳密な推定には最尤推定や Bessel 関数の逆関数が必要。

    Args:
        x_samples: サンプル [batch, dim]

    Returns:
        kappa: 推定された集中度（参考値）
        mean_dir: 推定された平均方向
    """
    mean_dir = x_samples.mean(dim=0)
    R = mean_dir.norm()
    mean_dir = F.normalize(mean_dir, dim=0)

    # 近似式（高次元での近似、精度は限定的）
    dim = x_samples.shape[-1]
    kappa = R * (dim - R**2) / (1 - R**2 + 1e-8)

    return kappa.item(), mean_dir


def visualize_spherical_diffusion_3d():
    """3次元球面上の拡散を可視化"""
    # 初期分布：北極付近に集中
    n_samples = 100
    kappa_init = 50  # 高い集中度

    # vMF分布からサンプリング（近似）
    mean_dir = torch.tensor([0.0, 0.0, 1.0])
    noise = torch.randn(n_samples, 3)
    x_0 = F.normalize(mean_dir + noise / np.sqrt(kappa_init), dim=-1)

    # Forward Process
    trajectory = spherical_forward_process(x_0, timesteps=200, dt=0.05)

    # 3Dプロット
    fig = plt.figure(figsize=(15, 5))

    # 球面を描画
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))

    timesteps_to_show = [0, 50, 100, 200]
    titles = ["t=0 (Concentrated)", "t=50", "t=100", "t=200 (Diffused)"]

    for idx, (t, title) in enumerate(zip(timesteps_to_show, titles, strict=True)):
        ax = fig.add_subplot(1, 4, idx + 1, projection="3d")

        # 球面
        ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.1, color="gray")

        # サンプル点
        points = trajectory[t].numpy()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="blue", s=10, alpha=0.6)

        # 集中度を推定
        kappa, _ = estimate_vMF_concentration(trajectory[t])

        ax.set_title(f"{title}\nκ≈{kappa:.1f}")
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    plt.tight_layout()
    plt.savefig("spherical_diffusion_3d.png", dpi=150)
    plt.close()

    print("Saved: spherical_diffusion_3d.png")


def compare_euclidean_vs_spherical():
    """ユークリッド空間と球面の拡散の比較"""
    n_samples = 500
    dim = 3
    timesteps = 100

    # 初期分布（同じ点から開始）
    x_0 = torch.zeros(n_samples, dim)
    x_0[:, 2] = 1.0  # 北極

    # ユークリッド空間での拡散
    def euclidean_diffusion(x_0, timesteps, dt=0.1):
        x = x_0.clone()
        trajectory = [x.clone()]
        for _ in range(timesteps):
            x = x + np.sqrt(2 * dt) * torch.randn_like(x)
            trajectory.append(x.clone())
        return torch.stack(trajectory)

    traj_euclidean = euclidean_diffusion(x_0.clone(), timesteps)

    # 球面での拡散
    x_0_sphere = F.normalize(x_0, dim=-1)
    traj_spherical = spherical_forward_process(x_0_sphere, timesteps, dt=0.1)

    # 統計量の比較
    print("=" * 60)
    print("Euclidean vs Spherical Diffusion Comparison")
    print("=" * 60)

    for t in [0, 25, 50, 100]:
        euc_norm = traj_euclidean[t].norm(dim=-1)
        sph_norm = traj_spherical[t].norm(dim=-1)

        print(f"\nt={t}:")
        print(f"  Euclidean: norm mean={euc_norm.mean():.3f}, std={euc_norm.std():.3f}")
        print(f"  Spherical: norm mean={sph_norm.mean():.3f}, std={sph_norm.std():.3f}")

        kappa, _ = estimate_vMF_concentration(traj_spherical[t])
        print(f"  Spherical κ (concentration): {kappa:.1f}")


# 実行
if __name__ == "__main__":
    visualize_spherical_diffusion_3d()
    compare_euclidean_vs_spherical()
