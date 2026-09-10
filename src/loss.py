"""复合损失函数：MSE + 梯度 L1 + 深监督。

- 主损失：MSE，与评估指标对齐。
- 梯度损失：拉普拉斯算子（1次差分）L1 距离，保护文字边缘结构。
- 深监督损失：每个解码层辅助预测都贡献一项 MSE（权重衰减）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobel(nn.Module):
    """Sobel 算子：计算 dx 与 dy 方向梯度。"""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, x):
        # 确保 buffer 与输入同 device/dtype
        kx = self.kx.to(device=x.device, dtype=x.dtype)
        ky = self.ky.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)


class CombinedLoss(nn.Module):
    """主损失 + 边缘梯度损失 + 深监督损失。"""

    def __init__(self, w_grad: float = 0.2, w_aux: float = 0.1):
        super().__init__()
        self.sobel = Sobel()
        self.w_grad = w_grad
        self.w_aux = w_aux

    def forward(self, pred, target, aux_preds=None):
        # 1) 主 MSE
        mse = F.mse_loss(pred, target)
        # 2) 梯度 L1
        g_pred = self.sobel(pred)
        g_target = self.sobel(target)
        grad_l1 = F.l1_loss(g_pred, g_target)
        loss = mse + self.w_grad * grad_l1

        # 3) 深监督
        if aux_preds is not None:
            aux_total = 0.0
            n = len(aux_preds)
            for i, ap in enumerate(aux_preds):
                # 深监督权重随深度衰减（越深权重越小）
                w = (0.5 ** (n - 1 - i))
                # 辅助预测的尺寸可能比 target 小，做对齐
                if ap.shape[2:] != target.shape[2:]:
                    ap = F.interpolate(
                        ap, size=target.shape[2:], mode="bilinear", align_corners=False
                    )
                aux_total = aux_total + w * F.mse_loss(ap, target)
            loss = loss + self.w_aux * aux_total
        return loss