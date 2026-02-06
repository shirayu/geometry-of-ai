import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# 簡単な2パラメータのロジスティック回帰
class SimpleLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.0, 0.0]))

    def forward(self, x):
        return torch.sigmoid(x @ self.w)


# データ生成（線形分離可能）
torch.manual_seed(42)
X = torch.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).float()

# 通常の勾配降下
model_sgd = SimpleLogistic()
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.1)
losses_sgd = []

for _ in range(100):
    optimizer_sgd.zero_grad()
    pred = model_sgd(X)
    loss = F.binary_cross_entropy(pred, y)
    loss.backward()
    optimizer_sgd.step()
    losses_sgd.append(loss.item())

# 適応的前処理（Adamで近似的に）
model_adam = SimpleLogistic()
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.1)
losses_adam = []

for _ in range(100):
    optimizer_adam.zero_grad()
    pred = model_adam(X)
    loss = F.binary_cross_entropy(pred, y)
    loss.backward()
    optimizer_adam.step()
    losses_adam.append(loss.item())

# 可視化
plt.figure(figsize=(10, 6))
plt.plot(losses_sgd, label="SGD (Standard Gradient)", linewidth=2)
plt.plot(losses_adam, label="Adam (Adaptive Preconditioning)", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Convergence: Standard Gradient vs Adaptive Preconditioning")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale("log")
plt.savefig("gradient_comparison.png", dpi=150, bbox_inches="tight")
print("収束の比較を gradient_comparison.png に保存")
