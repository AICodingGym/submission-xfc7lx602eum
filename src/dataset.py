"""PyTorch Dataset：文档去噪的训练/验证数据加载与增强（V2 增强版）。

设计要点（相对 V1 升级）：
- 输入/标签均为 [0,1] float32 灰度，形状 (C, H, W)，C=3（gray + 局部均值 + 局部方差）。
- 局部均值/方差通道：帮助模型理解背景光照场与噪声强度，显著加速收敛。
- 短图（258 高）过采样：补足样本不均衡，平衡两种尺寸的损失贡献。
- 随机裁剪 patch、水平/垂直翻转、90° 旋转、轻度亮度/对比度抖动、噪声注入。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import common as C


def _local_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算局部均值和方差（k=31 高斯估计），输出 [0,1] 归一化。"""
    if x.ndim != 2:
        raise ValueError(f"expected 2D, got {x.shape}")
    k = 31
    mu = cv2.GaussianBlur(x, (k, k), 0).astype(np.float32)
    mu2 = cv2.GaussianBlur((x * x).astype(np.float32), (k, k), 0)
    var = np.clip(mu2 - mu * mu, 0.0, 1.0)
    return mu, var


class DocDataset(Dataset):
    def __init__(
        self,
        stems: list[str],
        train: bool = True,
        patch: int = 256,
        seed: int = 0,
        with_features: bool = True,
        noise_aug: bool = False,
    ):
        self.stems = stems
        self.train = train
        self.patch = patch
        self.rng = np.random.default_rng(seed)
        self.with_features = with_features
        self.noise_aug = noise_aug
        self.channels = 3 if with_features else 1

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
            if self.noise_aug:
                x = self._inject_noise(x)

        if self.with_features:
            mu, var = _local_features(x)
            t = np.stack([x, mu, var], axis=0).astype(np.float32)
        else:
            t = x[None].astype(np.float32)
        xt = torch.from_numpy(t)
        yt = torch.from_numpy(y[None].astype(np.float32))
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
        if rng.random() < 0.5:
            x = x[:, ::-1]
            y = y[:, ::-1]
        if rng.random() < 0.5:
            x = x[::-1, :]
            y = y[::-1, :]
        k = int(rng.integers(0, 4))
        if k:
            x = np.rot90(x, k)
            y = np.rot90(y, k)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def _jitter(self, x):
        rng = self.rng
        a = 1.0 + rng.uniform(-0.1, 0.1)
        b = rng.uniform(-0.05, 0.05)
        x = a * x + b
        return np.clip(x, 0.0, 1.0)

    def _inject_noise(self, x):
        """合成额外噪声：模拟咖啡渍/暗斑/高频纹理。"""
        rng = self.rng
        h, w = x.shape
        if rng.random() < 0.5:
            cy = rng.integers(0, h)
            cx = rng.integers(0, w)
            sigma = max(rng.uniform(20, 60), 1)
            yy, xx = np.mgrid[0:h, 0:w]
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
            intensity = rng.uniform(0.05, 0.2)
            x = x - intensity * blob
        if rng.random() < 0.5:
            tex = rng.normal(0, rng.uniform(0.01, 0.04), size=(h, w)).astype(np.float32)
            x = x + tex
        return np.clip(x, 0.0, 1.0)


def is_short_stem(stem: str) -> bool:
    """判断是否为 258x540 短图 stem。"""
    p = C.TRAIN_DIR / f"{stem}.png"
    if not p.exists():
        p = C.TEST_DIR / f"{stem}.png"
    if not p.exists():
        return False
    img = C.load_gray(p)
    return img.shape[0] == 258


def oversample_short(stems: list[str], factor: int = 2) -> list[str]:
    """将短图（258x540）按 factor 倍过采样。"""
    out = []
    for s in stems:
        out.append(s)
        if is_short_stem(s):
            out.extend([s] * (factor - 1))
    return out