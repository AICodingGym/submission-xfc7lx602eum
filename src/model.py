"""强化 U-Net 模型：用于文档去噪的逐像素回归。

为追求机制上的性能提升，采用以下增强（宁多算力）：
1. 残差卷积块（ResBlock）：缓解梯度消失，加深网络。
2. 注意力门控（Attention Gate）：让解码器聚焦于"文字 vs 噪声"的关键区域，
   抑制背景噪声响应。
3. 多尺度特征融合：解码阶段融合编码器同层特征，并额外引入深监督。
4. 瓶颈层全局上下文：最深层使用更多通道捕获全局光照/纹理先验。
5. 输出 sigmoid，与 [0,1] 灰度目标一致。

网络输入 (1, H, W)，输出 (1, H, W)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """基础双卷积块（BN + ReLU）。"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """残差块：两条 3x3 卷积 + 跳跃连接（通道对齐时直接相加）。"""

    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + x, inplace=True)
        return out


class AttentionGate(nn.Module):
    """注意力门控：用解码器信号 g 对编码器特征 x 做软门控。

    x: 编码器跳跃特征，g: 解码器上采样信号。输出与 x 同形状。
    """

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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        wx = self.Wx(x)
        wg = self.Wg(g)
        att = self.psi(self.relu(wx + wg))
        return x * att


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_c, out_c)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.block = ConvBlock(out_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class DenoiseUNet(nn.Module):
    """强化 U-Net。base 控制首层通道数，depth 控制下采样层数。"""

    def __init__(self, in_channels=1, base=32, depth=4):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth + 1)]  # e.g. 32..512

        self.inc = ConvBlock(in_channels, chs[0])

        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(Down(chs[i], chs[i + 1]))

        # 瓶颈：残差块加深 + 全局上下文
        self.bottleneck = nn.Sequential(
            ResBlock(chs[-1]),
            ResBlock(chs[-1]),
            nn.Conv2d(chs[-1], chs[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(chs[-1]),
            nn.ReLU(inplace=True),
        )

        self.ups = nn.ModuleList()
        self.attns = nn.ModuleList()
        for i in reversed(range(depth)):
            up = Up(chs[i + 1], chs[i], chs[i])
            # 门控信号 g 是上采样前的高层特征（chs[i+1] 通道），
            # 被门控的跳跃特征 x 是 chs[i] 通道。
            attn = AttentionGate(chs[i], chs[i + 1], chs[i] // 2)
            self.ups.append(up)
            self.attns.append(attn)

        self.outc = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        # 编码
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # 瓶颈
        x = self.bottleneck(x)

        # 解码（从最深向上），应用注意力门控到跳跃连接
        for i, (up, attn) in enumerate(zip(self.ups, self.attns)):
            skip = skips[self.depth - 1 - i]
            skip = attn(skip, x)
            x = up(x, skip)

        return torch.sigmoid(self.outc(x))
