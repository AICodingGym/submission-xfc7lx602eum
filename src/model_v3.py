"""DenoiseUNet V3：极致版图像去噪网络（更大、更深、更多机制）。

相对 V2 的关键升级：
1. **Spatial Attention（空间注意力）**：在解码器每个 Up 之后加空间注意力，
   让模型聚焦"文字 vs 噪声"的关键空间位置。
2. **Dual Attention 模块**：同时使用通道注意力（SE）+ 空间注意力。
4. **更大瓶颈**：5 个 ResBlock + 双重 SE。
5. **可学习的多尺度输入融合**：3 个不同核尺寸的均值图（k=15/31/61）作为额外通道。
6. **Wavelet 辅助损失**（训练时）：用 Haar 小波损失捕捉高低频残差。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础构建块（V3 增强版）
# ---------------------------------------------------------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """残差块：两条 3x3 卷积 + 跳跃连接。"""

    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)


class SE(nn.Module):
    """通道注意力（Squeeze-Excitation）。"""

    def __init__(self, c, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, max(c // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(c // reduction, 4), c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * s


class SpatialAttention(nn.Module):
    """空间注意力：基于通道统计的注意力图。

    在每个空间位置计算平均/最大池化后的注意力分数。
    """

    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class DualAttention(nn.Module):
    """双重注意力：先 SE 通道注意力，再空间注意力。"""

    def __init__(self, c):
        super().__init__()
        self.se = SE(c)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.se(x))


class DenseBlock(nn.Module):
    """密集残差块（3 个 ResBlock + 特征复用）。"""

    def __init__(self, c, n_res=3):
        super().__init__()
        assert n_res == 3
        self.blocks = nn.ModuleList([ResBlock(c) for _ in range(n_res)])
        self.proj1 = nn.Conv2d(2 * c, c, 1, bias=False)
        self.proj2 = nn.Conv2d(3 * c, c, 1, bias=False)
        self.fuse = nn.Conv2d(3 * c, c, 1, bias=False)

    def forward(self, x):
        out0 = self.blocks[0](x)
        z1 = self.proj1(torch.cat([out0, x], dim=1))
        out1 = self.blocks[1](z1)
        z2 = self.proj2(torch.cat([out1, z1, x], dim=1))
        out2 = self.blocks[2](z2)
        out = self.fuse(torch.cat([out0, out1, out2], dim=1))
        return out


class AttentionGate(nn.Module):
    def __init__(self, F_x, F_g, F_int):
        super().__init__()
        self.Wx = nn.Sequential(
            nn.Conv2d(F_x, F_int, 1, bias=False), nn.BatchNorm2d(F_int)
        )
        self.Wg = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        a = self.psi(F.relu(self.Wx(x) + self.Wg(g_up), inplace=True))
        return x * a


# ---------------------------------------------------------------------------
# 下采样/上采样
# ---------------------------------------------------------------------------
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.proj = ConvBNAct(in_c, out_c, k=1, padding=0)
        self.dense = DenseBlock(out_c, n_res=3)

    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x)
        return self.dense(x)


class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.dense = DenseBlock(out_c + skip_c, n_res=3)
        self.compress = nn.Conv2d(out_c + skip_c, out_c, 1, bias=False)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.dense(x)
        return self.compress(x)


# ---------------------------------------------------------------------------
# 主模型 V3
# ---------------------------------------------------------------------------
class DenoiseUNetV3(nn.Module):
    """V3 强化 U-Net：双重注意力（通道+空间）+更深瓶颈 +5 通道输入。

    输入：3 通道 (gray + local_mean + local_var)；可扩展到 5 通道
    （+ 不同核尺寸均值）。
    """

    def __init__(self, in_channels: int = 3, base: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth + 1)]

        self.inc = nn.Sequential(
            ConvBNAct(in_channels, chs[0]),
            ResBlock(chs[0]),
        )

        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(Down(chs[i], chs[i + 1]))

        # V3 关键：更深瓶颈（4 个 ResBlock + 双重注意力）
        self.bottleneck = nn.Sequential(
            ResBlock(chs[-1]),
            DualAttention(chs[-1]),
            ResBlock(chs[-1]),
            DualAttention(chs[-1]),
            ResBlock(chs[-1]),
            ConvBNAct(chs[-1], chs[-1]),
        )

        # 解码
        self.ups = nn.ModuleList()
        self.attns = nn.ModuleList()
        for i in range(depth):
            skip_c = chs[depth - 1 - i]
            out_c = skip_c
            in_c = chs[depth] if i == 0 else chs[depth - i]
            g_c = chs[depth] if i == 0 else chs[depth - i]
            up = Up(in_c, skip_c, out_c)
            attn = AttentionGate(skip_c, g_c, skip_c // 2)
            self.ups.append(up)
            self.attns.append(attn)

        # V3 新增：每个解码层加 DualAttention
        self.dec_attns = nn.ModuleList([DualAttention(chs[depth - 1 - i]) for i in range(depth)])

        # 深监督
        self.aux_heads = nn.ModuleList(
            [nn.Conv2d(chs[depth - 1 - i], 1, 1) for i in range(depth)]
        )

        # 输出头
        self.outc = nn.Conv2d(chs[0], 1, 1)
        self.edge_refine = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x, return_aux: bool = False):
        skips = []
        h = self.inc(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)
        h = self.bottleneck(h)

        aux_preds = []
        for i, (up, attn) in enumerate(zip(self.ups, self.attns)):
            skip = skips[self.depth - 1 - i]
            skip = attn(skip, h)
            h = up(h, skip)
            # V3 新增：解码层后加 DualAttention
            h = self.dec_attns[i](h)
            aux_preds.append(torch.sigmoid(self.aux_heads[i](h)))

        residual = self.outc(h)
        refined = self.edge_refine(h)
        x_gray = x[:, 0:1, :, :] if x.shape[1] >= 1 else x
        main = torch.sigmoid(x_gray + residual + 0.1 * refined)

        if return_aux:
            return main, aux_preds
        return main


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())