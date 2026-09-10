"""dataset.py 的单元测试。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import common as C
from src.dataset import DocDataset


def test_len_and_load():
    train, val = C.make_split(seed=42)
    ds = DocDataset(train[:5], train=True, patch=128, seed=0)
    assert len(ds) == 5
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    # V2 默认 3 通道（灰度 + 局部均值 + 局部方差）
    assert x.shape == (3, 128, 128)
    assert y.shape == (1, 128, 128)
    assert x.dtype == torch.float32
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_train_crop_patch_size():
    ds = DocDataset(["101"], train=True, patch=64, seed=0)
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and y.shape == (1, 64, 64)


def test_eval_full_size():
    # 验证模式不裁剪，输出全图（3 通道输入 + 1 通道标签）
    ds = DocDataset(["101"], train=False)
    x, y = ds[0]
    assert x.shape == (3, C.IMG_H, C.IMG_W)
    assert y.shape == (1, C.IMG_H, C.IMG_W)


def test_augment_x_y_geometrically_consistent():
    # 增强必须同时作用于 x 和 y：裁剪/翻转/旋转一致。
    # 直接调用内部方法，验证同一几何变换下，两者形状一致且 y 值域合法。
    ds = DocDataset(["101"], train=True, patch=128, seed=7)
    x, y = ds._load("101")
    xc, yc = ds._crop(x, y)
    xa, ya = ds._augment(xc, yc)
    assert xa.shape == ya.shape
    assert xa.shape == (128, 128)
    # y 是干净文档，值域仍在 [0,1]
    assert ya.min() >= 0.0 and ya.max() <= 1.0


def test_augment_preserves_label_structure():
    # 标签是二值化文档（0/1 为主），增强不应破坏其值域
    ds = DocDataset(["101"], train=True, patch=128, seed=3)
    _, y = ds[0]
    assert y.min() >= 0.0 and y.max() <= 1.0
