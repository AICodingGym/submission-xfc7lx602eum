"""PyTorch Dataset：文档去噪的训练/验证数据加载与增强。

设计要点（对应 PLAN.md）：
- 输入/标签均为 [0,1] float32 灰度，形状 (1, H, W)。
- 随机裁剪 patch、水平/垂直翻转、90° 旋转、轻度亮度/对比度抖动。
- 不做弹性形变，保护文字几何先验。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import common as C


class DocDataset(Dataset):
    def __init__(
        self,
        stems: list[str],
        train: bool = True,
        patch: int = 256,
        seed: int = 0,
    ):
        self.stems = stems
        self.train = train
        self.patch = patch
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.stems)

    def _load(self, stem: str):
        x = C.load_gray(C.TRAIN_DIR / f"{stem}.png")
        y = C.load_gray(C.CLEAN_DIR / f"{stem}.png")
        return x, y

    def __getitem__(self, idx):
        stem = self.stems[idx]
        x, y = self._load(stem)

        if self.train:
            x, y = self._crop(x, y)
            x, y = self._augment(x, y)
            x = self._jitter(x)

        # 转 tensor: (1, H, W)
        xt = torch.from_numpy(x[None, ...].astype(np.float32))
        yt = torch.from_numpy(y[None, ...].astype(np.float32))
        return xt, yt

    # -- 内部方法 ------------------------------------------------------------
    def _crop(self, x, y):
        h, w = x.shape
        if self.patch >= min(h, w):
            return x, y
        rng = self.rng
        top = int(rng.integers(0, h - self.patch + 1))
        left = int(rng.integers(0, w - self.patch + 1))
        return (
            x[top : top + self.patch, left : left + self.patch],
            y[top : top + self.patch, left : left + self.patch],
        )

    def _augment(self, x, y):
        rng = self.rng
        if rng.random() < 0.5:  # 水平翻转
            x = x[:, ::-1]
            y = y[:, ::-1]
        if rng.random() < 0.5:  # 垂直翻转
            x = x[::-1, :]
            y = y[::-1, :]
        k = int(rng.integers(0, 4))  # 90° 旋转
        if k:
            x = np.rot90(x, k)
            y = np.rot90(y, k)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def _jitter(self, x):
        # 亮度 + 对比度轻度抖动，模拟光照差异
        rng = self.rng
        a = 1.0 + rng.uniform(-0.1, 0.1)  # 对比度
        b = rng.uniform(-0.05, 0.05)  # 亮度
        x = a * x + b
        return np.clip(x, 0.0, 1.0)
