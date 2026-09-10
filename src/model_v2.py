"""DenoiseUNet V2：强化版图像去噪网络。

相对 V1 的关键改进：
1. 多尺度输入特征融合：编码器首层额外接收局部均值/方差通道（3 输入）。
2. 密集跳跃连接（DSConv 风格）：同一 level 内多个 ResBlock 串联，特征复用。
3. 更深的瓶颈：3 个 ResBlock + 通道注意力（SE-like）。
4. 多尺度特征融合（Deep Supervision）：每个解码层都输出辅助预测，
   训练时对中间层预测施加额外 MSE 损失。
5. 边缘感知输出头：在最终 sigmoid前附加一个 1x1 卷积产生的梯度分支，输出
   "边缘细化残差"，帮助恢复文字笔画的精细结构。
6. 残差学习：模型预测残差 `clean = sigmoid(x_input + f(x))`，加速收敛并稳定。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础构建块
# ---------------------------------------------------------------------------
class ConvBNAct(nn.Module):
    """Conv -> BN -> ReLU。"""

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
            nn.Linear(c, c // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // reduction, c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * s


class DenseBlock(nn.Module):
    """密集残差块：3 个 ResBlock，逐层特征复用。

    设计：每个 ResBlock 接收 "之前所有 ResBlock 输出 + 原始输入" 的拼接。
    - 块 0：输入 x (c 通道) -> out0 (c)
    - 块 1：拼接 (out0, x) = 2c -> proj -> z1 (c) -> blocks[1] -> out1 (c)
    - 块 2：拼接 (out1, z1, x) = 3c -> proj -> z2 (c) -> blocks[2] -> out2 (c)
    - 融合 (out0, out1, out2) = 3c -> fuse -> 输出 (c)
    """

    def __init__(self, c, n_res=3):
        super().__init__()
        assert n_res == 3, "当前实现仅支持 n_res=3"
        self.blocks = nn.ModuleList([ResBlock(c) for _ in range(n_res)])
        self.proj1 = nn.Conv2d(2 * c, c, 1, bias=False)   # (out0, x) -> c
        self.proj2 = nn.Conv2d(3 * c, c, 1, bias=False)   # (out1, z1, x) -> c
        self.fuse = nn.Conv2d(3 * c, c, 1, bias=False)    # (out0, out1, out2) -> c

    def forward(self, x):
        out0 = self.blocks[0](x)
        z1 = self.proj1(torch.cat([out0, x], dim=1))
        out1 = self.blocks[1](z1)
        z2 = self.proj2(torch.cat([out1, z1, x], dim=1))
        out2 = self.blocks[2](z2)
        out = self.fuse(torch.cat([out0, out1, out2], dim=1))
        return out


class AttentionGate(nn.Module):
    """注意力门控：用解码器信号 g 对编码器特征 x 做软门控。"""

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
# 下采样/上采样模块
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
        # DenseBlock 的输入是 (out_c + skip_c) 通道，输出也应为 out_c
        self.dense = DenseBlock(out_c + skip_c, n_res=3)
        # 融合投影：把 out_c + skip_c 投影回 out_c
        self.proj = nn.Conv2d(3 * (out_c + skip_c), out_c, 1, bias=False)
        # 但 DenseBlock 内部 fuse 是 3c -> c (c = out_c + skip_c)，所以输出仍是 out_c + skip_c
        # 加一个 1x1 把通道压回 out_c
        self.compress = nn.Conv2d(out_c + skip_c, out_c, 1, bias=False)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.dense(x)
        return self.compress(x)


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------
class DenoiseUNetV2(nn.Module):
    """强化 U-Net V2。

    多尺度输入（3 通道：gray + local mean + local var），深监督，残差学习。

    Args:
        in_channels: 输入通道数（默认 3：灰度 + 局部均值 + 局部方差）
        base: 首层通道数
        depth: 下采样层数
    """

    def __init__(self, in_channels: int = 3, base: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth + 1)]

        # 编码首层：处理原始输入（含局部特征）
        self.inc = nn.Sequential(
            ConvBNAct(in_channels, chs[0]),
            ResBlock(chs[0]),
        )

        # 编码下采样
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(Down(chs[i], chs[i + 1]))

        # 瓶颈：SE 注意力 + 多残差块
        self.bottleneck = nn.Sequential(
            ResBlock(chs[-1]),
            SE(chs[-1]),
            ResBlock(chs[-1]),
            ConvBNAct(chs[-1], chs[-1]),
        )

        # 解码上采样 + 注意力门控
        # 解码层 i（0..depth-1）从最浅到最深循环，i=0 是最深（接收 bottleneck）
        # 接收来自编码器 level (depth-1-i) 的 skip
        self.ups = nn.ModuleList()
        self.attns = nn.ModuleList()
        for i in range(depth):
            skip_c = chs[depth - 1 - i]  # 编码器对应层的通道
            out_c = chs[depth - 1 - i]   # 解码层输出通道（与 skip 一致）
            in_c = chs[depth - i] if i == 0 else chs[depth - i]  # 上一层通道
            # 第一次：in = chs[depth]，后续：in = 上层输出（等于当前 skip_c）
            if i == 0:
                in_c = chs[depth]  # bottleneck 输出
            else:
                in_c = chs[depth - i]  # = chs[depth - i] = 上层 out
            up = Up(in_c, skip_c, out_c)
            # g 是更深层特征（上一层 h），第一次是 bottleneck (chs[depth])
            if i == 0:
                g_c = chs[depth]
            else:
                g_c = chs[depth - i]  # 上层 h 的通道
            attn = AttentionGate(skip_c, g_c, skip_c // 2)
            self.ups.append(up)
            self.attns.append(attn)

        # 深监督：每个解码层都辅助输出 1 通道预测
        # 按解码层实际输出通道建 head（depth 个，顺序对应解码层 i=0..depth-1）
        self.aux_heads = nn.ModuleList(
            [nn.Conv2d(chs[depth - 1 - i], 1, 1) for i in range(depth)]
        )

        # 输出头：1x1 卷积 + sigmoid
        self.outc = nn.Conv2d(chs[0], 1, 1)

        # 边缘细化残差：1x1 卷积产生细节残差，与主输出相加
        self.edge_refine = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x, return_aux: bool = False):
        """返回主预测，或同时返回所有解码层的辅助预测（用于深监督训练）。"""
        # 多尺度输入已经由 Dataset 提供（3 通道）
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
            # 解码层 i 输出通道 = chs[depth-1-i]
            # aux_heads[i] 的 in_channels = chs[depth-1-i]，直接对应
            aux_preds.append(torch.sigmoid(self.aux_heads[i](h)))

        # 主输出：残差学习（在 sigmoid 内做加法近似，保留数值稳定）
        residual = self.outc(h)
        refined = self.edge_refine(h)
        # 输入 x 的第 0 通道作为残差基准
        x_gray = x[:, 0:1, :, :] if x.shape[1] >= 1 else x
        main = torch.sigmoid(x_gray + residual + 0.1 * refined)

        if return_aux:
            return main, aux_preds
        return main


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())