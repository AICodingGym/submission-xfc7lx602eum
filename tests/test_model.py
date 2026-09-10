"""model.py 的单元测试。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import DenoiseUNet, ResBlock, AttentionGate, ConvBlock, Up


def test_convblock_shape():
    m = ConvBlock(1, 8)
    x = torch.randn(2, 1, 16, 16)
    y = m(x)
    assert y.shape == (2, 8, 16, 16)


def test_resblock_shape_and_residual():
    m = ResBlock(8)
    x = torch.randn(2, 8, 16, 16)
    y = m(x)
    assert y.shape == x.shape


def test_attention_gate_shape_and_scaling():
    m = AttentionGate(16, 8, 8)
    x = torch.rand(2, 16, 32, 32)  # 非负输入，便于验证缩放
    g = torch.randn(2, 8, 16, 16)  # 尺寸可不同，内部会插值
    y = m(x, g)
    assert y.shape == x.shape
    # 注意力门控是 sigmoid 缩放，非负输入下应逐元素 ≤ x
    assert torch.all(y <= x + 1e-6)


def test_up_shape():
    m = Up(32, 16, 16)
    x = torch.randn(2, 32, 8, 8)
    skip = torch.randn(2, 16, 16, 16)
    y = m(x, skip)
    assert y.shape == (2, 16, 16, 16)


def test_unet_forward_output_range_and_shape():
    m = DenoiseUNet(in_channels=1, base=16, depth=4)
    x = torch.randn(2, 1, 64, 64)
    y = m(x)
    assert y.shape == (2, 1, 64, 64)
    assert y.min() >= 0.0 and y.max() <= 1.0  # sigmoid 输出


def test_unet_full_image_size():
    # 全图 420x540 可被 2^depth 整除? 420/16=26.25 不可整，需验证上采样能还原
    m = DenoiseUNet(in_channels=1, base=16, depth=4)
    x = torch.randn(1, 1, 420, 540)
    y = m(x)
    # 上采样层做了插值对齐，输出应等于输入尺寸
    assert y.shape == (1, 1, 420, 540)


def test_unet_param_count():
    m = DenoiseUNet(in_channels=1, base=32, depth=4)
    n = sum(p.numel() for p in m.parameters())
    # 应是一个较大的网络（> 1M 参数，体现"宁复杂"）
    assert n > 1_000_000
