import torch


def fisher_information_gaussian(mu, sigma):
    """正規分布のフィッシャー情報行列（解析的）

    N(mu, sigma^2) のフィッシャー情報行列:
    [[1/sigma^2, 0],
     [0, 2/sigma^4]]
    """
    return torch.tensor([[1 / sigma**2, 0], [0, 2 / sigma**4]])


# 異なる分散での比較
sigmas = [0.5, 1.0, 2.0]

for sigma in sigmas:
    fisher = fisher_information_gaussian(mu=0.0, sigma=sigma)
    print(f"σ={sigma}:")
    print(f"  F_μμ = {fisher[0, 0]:.4f} (平均への感度)")
    print(f"  F_σσ = {fisher[1, 1]:.4f} (分散への感度)")
    print()

# 出力例:
# σ=0.5:
#   F_μμ = 4.0000 (平均への感度)  ← 分散が小さいほど感度が高い
#   F_σσ = 32.0000 (分散への感度)
#
# σ=1.0:
#   F_μμ = 1.0000
#   F_σσ = 2.0000
#
# σ=2.0:
#   F_μμ = 0.2500  ← 分散が大きいと感度が低い
#   F_σσ = 0.1250
