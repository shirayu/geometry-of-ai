import torch
import torch.nn.functional as F


def quantize(tensor, num_bits=4):
    """[-1, 1]の範囲を仮定した対称一様量子化（最小構成）。"""
    if num_bits < 2:
        raise ValueError("num_bitsは2以上である必要があります")
    qmax = 2 ** (num_bits - 1) - 1
    return torch.round(tensor.clamp(-1, 1) * qmax) / qmax


# 概念的なコード
torch.manual_seed(42)
x = torch.randn(2, 3)
x_normalized = F.normalize(x, dim=-1)  # ノルム1に正規化
x_quantized = quantize(x_normalized)  # 量子化
# この時点で ||x_quantized|| ≠ 1 の可能性
x_renormalized = F.normalize(x_quantized, dim=-1)  # 再正規化
