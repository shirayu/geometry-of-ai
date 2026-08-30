import warnings

import torch


class OODDetector:
    """κに基づくOOD検知

    WARNING: 閾値はデータセット・モデルに依存する。
    必ず検証セットで校正すること。
    """

    def __init__(self, threshold=10.0):
        self.threshold = threshold
        self._is_calibrated = False

    def calibrate(self, kappa_in, kappa_out, target_fpr=0.05):
        """検証セットで閾値を校正

        Args:
            kappa_in: 分布内データのκ。1D tensor [n_in] を想定。
                      2D [n_in, 1] の場合は自動的にsqueezeされる。
            kappa_out: 分布外データのκ。1D tensor [n_out] を想定。
                       2D [n_out, 1] の場合は自動的にsqueezeされる。
            target_fpr: 目標偽陽性率（分布内をOODと誤判定する率）

                        NOTE: 本実装では「分布内サンプルを誤ってOODと判定する率」を
                        FPR（False Positive Rate）と定義している。
                        文献によっては「OODサンプルを誤って分布内と判定する率」を
                        FPRとする場合もあるため、比較時は定義に注意すること。

        Returns:
            calibrated_threshold: 校正された閾値
        """
        # 形状を1Dに揃える
        kappa_in = kappa_in.squeeze()
        kappa_out = kappa_out.squeeze()

        # 分布内データのκ分位点から閾値を決定
        # target_fpr = 0.05 なら、分布内の5%がOODと判定される閾値
        self.threshold = torch.quantile(kappa_in, target_fpr).item()
        self._is_calibrated = True

        # 検出率（分布外をOODと正しく判定する率）を計算
        tpr = (kappa_out < self.threshold).float().mean().item()

        print(f"Calibrated threshold: {self.threshold:.2f}")
        print(f"  FPR (in-dist as OOD): {target_fpr:.1%}")
        print(f"  TPR (out-dist as OOD): {tpr:.1%}")

        return self.threshold

    def is_ood(self, kappa):
        """κが閾値以下ならOODと判定

        Args:
            kappa: 推定された集中度 [batch, 1] または [batch]

        Returns:
            ood_mask: OODフラグ [batch]
        """
        if not self._is_calibrated:
            warnings.warn(
                "OODDetector is not calibrated. Results may be unreliable.",
                stacklevel=2,
            )
        return kappa.squeeze(-1) < self.threshold

    def get_confidence(self, kappa):
        """κを[0, 1]の確信度スコアに変換

        Args:
            kappa: 集中度 [batch, 1] または [batch]

        Returns:
            confidence: 確信度スコア [batch]
        """
        # シグモイドで[0, 1]に変換
        # threshold周りで0.5になるように調整
        return torch.sigmoid((kappa.squeeze(-1) - self.threshold) / 5.0)


# 使用例
detector = OODDetector(threshold=10.0)

# 仮のκ値
kappa_in_distribution = torch.tensor([[50.0], [30.0], [25.0]])
kappa_out_of_distribution = torch.tensor([[5.0], [3.0], [1.0]])

print("In-distribution:")
print(f"  OOD: {detector.is_ood(kappa_in_distribution)}")
print(f"  Confidence: {detector.get_confidence(kappa_in_distribution)}")

print("Out-of-distribution:")
print(f"  OOD: {detector.is_ood(kappa_out_of_distribution)}")
print(f"  Confidence: {detector.get_confidence(kappa_out_of_distribution)}")
