import torch

# 【重要】以下は「自然勾配法の形式」を示す教育的疑似コードであり、
# Fisher情報行列を単位行列で代用しているため、自然勾配の性質を何も示さない。
# 実際には単なる学習率の再スケール（(1+damping)倍）に過ぎない。


def natural_gradient_step(model, loss, lr=0.01, damping=1e-4):
    """自然勾配法の"形式"を示す疑似コード（実用不可）

    警告: 本実装はFisher情報を単位行列で代用しており、
    自然勾配の本質（モデル依存の計量）が完全に失われている。
    fisher_approx = (1+damping)*I なので、これは単なるスカラー倍であり、
    自然勾配が提供する「空間の歪みの補正」は一切行われない。

    真の実装には、モデルの出力分布からFisherを計算する必要がある。
    """
    # 通常の勾配
    loss.backward()

    # パラメータと勾配を1次元ベクトルに
    params = torch.cat([p.flatten() for p in model.parameters()])
    grads = torch.cat([p.grad.flatten() for p in model.parameters()])

    # 【問題点】Fisherを単位行列で代用 → これでは自然勾配にならない
    # fisher_approx = I + damping*I = (1+damping)*I
    # つまり、恒等変換のスカラー倍 = 学習率を (1+damping) 倍するだけ
    # 実際には、出力のlog probabilityの勾配の外積の期待値を計算する必要がある
    fisher_approx = torch.eye(len(params)) + damping * torch.eye(len(params))

    # "自然勾配のような形" = Fisher^{-1} @ grad
    # （ただしFisher = (1+damping)*I なので、実質 grad / (1+damping)）
    natural_grad = torch.linalg.solve(fisher_approx, grads)

    # パラメータ更新
    idx = 0
    for p in model.parameters():
        p_len = p.numel()
        p.data -= lr * natural_grad[idx : idx + p_len].view_as(p)
        idx += p_len
        p.grad.zero_()


# 【推奨】実用にはK-FAC等の専用ライブラリ、または対角Fisher近似を用いるべき
