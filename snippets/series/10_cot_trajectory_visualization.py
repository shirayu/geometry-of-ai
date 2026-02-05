"""CoT軌跡の可視化（概念的なデモ）

注意: これは実際のLLMの内部状態を可視化しているわけではない。
CoTの「軌跡」という比喩を直感的に理解するためのデモである。
"""

import matplotlib.pyplot as plt
import numpy as np


def generate_cot_trajectory(start, end, n_steps, noise_level=0.1):
    """CoT軌跡を生成（概念的なシミュレーション）

    Args:
        start: 開始点（質問の埋め込み）
        end: 終了点（答えの埋め込み）
        n_steps: 中間ステップ数
        noise_level: ノイズの強度

    Returns:
        trajectory: 軌跡の点列 [n_steps+2, dim]
    """
    # 直線経路
    t = np.linspace(0, 1, n_steps + 2).reshape(-1, 1)
    linear_path = start + t * (end - start)

    # ノイズを追加（中間ステップのみ）
    noise = np.random.randn(n_steps + 2, len(start)) * noise_level
    noise[0] = 0  # 開始点はノイズなし
    noise[-1] = 0  # 終了点はノイズなし

    trajectory = linear_path + noise

    return trajectory


def generate_difficult_trajectory(start, end, n_steps, detour_strength=1.0):
    """難しい問題の軌跡（迂回が必要）

    Args:
        start: 開始点
        end: 終了点
        n_steps: 中間ステップ数
        detour_strength: 迂回の強さ

    Returns:
        trajectory: 軌跡の点列
    """
    dim = len(start)
    trajectory = [start]

    # 迂回点を生成
    mid_point = (start + end) / 2
    # 直線に垂直な方向に迂回
    direction = end - start
    perpendicular = np.array([-direction[1], direction[0], 0] if dim >= 3 else [-direction[1], direction[0]])
    perpendicular = perpendicular[:dim]
    if np.linalg.norm(perpendicular) > 0:
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
    detour_point = mid_point + detour_strength * perpendicular

    # スプライン的な経路
    for i in range(1, n_steps + 1):
        t = i / (n_steps + 1)
        # 二次ベジェ曲線的な補間
        p = (1 - t) ** 2 * start + 2 * (1 - t) * t * detour_point + t**2 * end
        # 少しノイズを追加
        p += np.random.randn(dim) * 0.05
        trajectory.append(p)

    trajectory.append(end)
    return np.array(trajectory)


def visualize_cot_comparison():
    """One-shot vs CoT の軌跡比較を可視化"""
    np.random.seed(42)

    # 3次元空間での可視化
    fig = plt.figure(figsize=(15, 5))

    # 開始点と終了点
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([3.0, 2.0, 1.0])

    # 1. One-shot（直接経路）
    ax1 = fig.add_subplot(131, projection="3d")
    oneshot = np.array([start, end])
    ax1.plot(oneshot[:, 0], oneshot[:, 1], oneshot[:, 2], "b-", linewidth=2, label="One-shot")
    ax1.scatter(*start, color="green", s=100, marker="o", label="Question")
    ax1.scatter(*end, color="red", s=100, marker="*", label="Answer")
    ax1.set_title("One-shot (Direct Path)")
    ax1.legend()
    ax1.set_xlabel("Dim 1")
    ax1.set_ylabel("Dim 2")
    ax1.set_zlabel("Dim 3")

    # 2. 簡単な問題のCoT（ほぼ直線）
    ax2 = fig.add_subplot(132, projection="3d")
    easy_cot = generate_cot_trajectory(start, end, n_steps=4, noise_level=0.1)
    ax2.plot(easy_cot[:, 0], easy_cot[:, 1], easy_cot[:, 2], "b-", linewidth=2)
    ax2.scatter(easy_cot[1:-1, 0], easy_cot[1:-1, 1], easy_cot[1:-1, 2], color="blue", s=50, marker="o", label="Steps")
    ax2.scatter(*start, color="green", s=100, marker="o", label="Question")
    ax2.scatter(*end, color="red", s=100, marker="*", label="Answer")
    ax2.set_title("Easy Problem CoT")
    ax2.legend()
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")
    ax2.set_zlabel("Dim 3")

    # 3. 難しい問題のCoT（迂回が必要）
    ax3 = fig.add_subplot(133, projection="3d")
    hard_cot = generate_difficult_trajectory(start, end, n_steps=6, detour_strength=1.5)
    ax3.plot(hard_cot[:, 0], hard_cot[:, 1], hard_cot[:, 2], "b-", linewidth=2)
    ax3.scatter(hard_cot[1:-1, 0], hard_cot[1:-1, 1], hard_cot[1:-1, 2], color="blue", s=50, marker="o", label="Steps")
    ax3.scatter(*start, color="green", s=100, marker="o", label="Question")
    ax3.scatter(*end, color="red", s=100, marker="*", label="Answer")
    # 直接経路も点線で表示
    ax3.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        "r--",
        linewidth=1,
        alpha=0.5,
        label="Direct (blocked)",
    )
    ax3.set_title("Hard Problem CoT (Detour)")
    ax3.legend()
    ax3.set_xlabel("Dim 1")
    ax3.set_ylabel("Dim 2")
    ax3.set_zlabel("Dim 3")

    plt.tight_layout()
    plt.savefig("cot_trajectory_comparison.png", dpi=150)
    plt.close()

    print("Saved: cot_trajectory_comparison.png")


def visualize_beam_search():
    """ビームサーチの軌跡を可視化"""
    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(10, 8))

    start = np.array([0.0, 0.5])
    end = np.array([5.0, 0.5])

    # 複数の候補経路
    n_beams = 4
    colors = plt.cm.viridis(np.linspace(0, 1, n_beams))

    all_trajectories = []
    scores = []

    for i in range(n_beams):
        # 各経路に異なる迂回を追加
        detour = 0.5 * (i - n_beams / 2)
        traj = []
        for t in np.linspace(0, 1, 10):
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + detour * np.sin(np.pi * t) + 0.1 * np.random.randn()
            traj.append([x, y])
        traj = np.array(traj)
        all_trajectories.append(traj)

        # スコアを計算（終点への近さ + 滑らかさ）
        final_dist = np.linalg.norm(traj[-1] - end)
        smoothness = np.mean(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        score = 1 / (final_dist + 0.5 * smoothness + 0.1)
        scores.append(score)

    # スコアでソート
    sorted_indices = np.argsort(scores)[::-1]

    # 描画
    for rank, idx in enumerate(sorted_indices):
        traj = all_trajectories[idx]
        alpha = 1.0 if rank == 0 else 0.3
        linewidth = 3 if rank == 0 else 1
        label = f"Beam {idx + 1} (score: {scores[idx]:.2f})"
        if rank == 0:
            label += " ✓ Selected"
        ax.plot(traj[:, 0], traj[:, 1], color=colors[idx], alpha=alpha, linewidth=linewidth, label=label)
        ax.scatter(traj[1:-1, 0], traj[1:-1, 1], color=colors[idx], alpha=alpha, s=20)

    ax.scatter(*start, color="green", s=200, marker="o", zorder=5, label="Start")
    ax.scatter(*end, color="red", s=200, marker="*", zorder=5, label="Goal")

    ax.set_xlabel("Position")
    ax.set_ylabel("State")
    ax.set_title("Beam Search: Multiple Path Exploration")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("beam_search_visualization.png", dpi=150)
    plt.close()

    print("Saved: beam_search_visualization.png")


# 実行
if __name__ == "__main__":
    visualize_cot_comparison()
    visualize_beam_search()
