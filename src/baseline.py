"""Stage 0 / M1 — 传统方法基线。

思路：
1. 背景估计：对原始图做大核中值滤波 / 形态学闭运算，得到低频背景（含阴影、
   褶皱的大尺度变化）。
2. 光照归一化：用 (x - bg) / (1 - bg) 的形式把背景推向 1（白），文字保持暗。
   这里用更稳健的 `clip((x - bg) / (1 - bg + eps))`，背景像素 x≈bg 时输出≈0。
3. 残差阈值：归一化后文字应显著低于背景，可对小于某阈值的像素做中值去噪，
   但为控制 RMSE 保持软输出（不做硬二值化）。

该基线只用于验证全链路与保底提交，不代表最终方案。
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_background(x: np.ndarray, ksize: int = 31) -> np.ndarray:
    """用大核中值滤波估计背景（灰度低频分量）。ksize 需为奇数。"""
    if ksize % 2 == 0:
        ksize += 1
    # 大核中值在 opencv 上对灰度图较快
    return cv2.medianBlur((x * 255.0).astype(np.uint8), ksize).astype(np.float32) / 255.0


def denoise_baseline(x: np.ndarray, ksize: int = 31, eps: float = 1e-3) -> np.ndarray:
    """基线去噪：背景估计 + 光照归一化，输出软灰度 [0,1]。

    目标：把背景（≈bg）推向 1（白），文字（暗于背景）保持暗。
    采用白平衡式归一化：out = x / bg，背景处 x≈bg -> 1，文字处 x<<bg -> 接近 0。
    clip 到 [0,1] 并保持软输出。
    """
    bg = estimate_background(x, ksize=ksize)
    out = x / (bg + eps)
    out = np.clip(out, 0.0, 1.0)
    return out


def denoise_baseline_batch(
    images: dict[str, np.ndarray], ksize: int = 31, eps: float = 1e-3
) -> dict[str, np.ndarray]:
    """对 {stem: array} 批量执行基线去噪。"""
    return {s: denoise_baseline(a, ksize=ksize, eps=eps) for s, a in images.items()}
