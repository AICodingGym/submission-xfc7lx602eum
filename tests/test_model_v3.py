"""model_v3 / V3 推理的单元测试。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import common as C
from src.dataset import DocDataset
from src.model_v3 import DenoiseUNetV3, count_params, DualAttention, SpatialAttention, ResBlock


def test_resblock_shape():
    m = ResBlock(16)
    x = torch.randn(2, 16, 32, 32)
    y = m(x)
    assert y.shape == x.shape


def test_spatial_attention_shape():
    m = SpatialAttention()
    x = torch.rand(2, 16, 32, 32)
    y = m(x)
    assert y.shape == x.shape
    # 注意力应限制在非负（sigmoid）
    assert y.min() >= 0


def test_dual_attention_shape():
    m = DualAttention(16)
    x = torch.rand(2, 16, 32, 32)
    y = m(x)
    assert y.shape == x.shape


def test_v3_forward_output_range_and_shape():
    m = DenoiseUNetV3(in_channels=3, base=16, depth=4)
    x = torch.rand(2, 3, 64, 64)
    y = m(x)
    assert y.shape == (2, 1, 64, 64)
    assert y.min() >= 0 and y.max() <= 1


def test_v3_forward_with_aux():
    m = DenoiseUNetV3(in_channels=3, base=16, depth=4)
    x = torch.rand(1, 3, 64, 64)
    main, aux = m(x, return_aux=True)
    assert main.shape == (1, 1, 64, 64)
    assert len(aux) == 4
    for a in aux:
        assert a.shape[1] == 1


def test_v3_supports_mixed_sizes():
    """V3 支持 420x540 与 258x540 两种尺寸。"""
    m = DenoiseUNetV3(in_channels=3, base=16, depth=4).eval()
    for h, w in [(420, 540), (258, 540)]:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 1, h, w)


def test_v3_param_count():
    """V3 应有更大参数量（> V2 的 56.8M）。"""
    m = DenoiseUNetV3(in_channels=3, base=32, depth=4)
    n = count_params(m)
    assert n > 55_000_000, f"V3 参数量 {n} 太小"
    # 上界：避免过度膨胀
    assert n < 200_000_000, f"V3 参数量 {n} 异常大"


def test_v3_does_not_break_v2_dataset():
    """V3 应能用 V2 的 DocDataset（3 通道）。"""
    train_stems, _ = C.make_split(seed=42)
    ds = DocDataset(train_stems[:2], train=True, patch=64, seed=0, with_features=True)
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    m = DenoiseUNetV3(in_channels=3, base=8, depth=3)
    pred = m(x[None])
    assert pred.shape == (1, 1, 64, 64)