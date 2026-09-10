"""后处理：温和极值化（soft thresholding），把置信像素推向 0/1 但不硬截断。

公式：p' = clip((p - 0.5) * k + 0.5, 0, 1)，k >= 1。
- k = 1 时恒等；k 越大越接近硬二值化。
- k 在验证集上扫描确定，无收益则不使用（k=1）。
"""
from __future__ import annotations

import numpy as np


def soft_threshold(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    """温和极值化。"""
    if k < 1.0:
        raise ValueError(f"k 必须 >= 1，收到 {k}")
    out = (x - 0.5) * k + 0.5
    return np.clip(out, 0.0, 1.0)


def tune_k(
    preds: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    ks=(1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0),
    metric=None,
) -> tuple[float, float]:
    """在验证集上扫描 k，返回 (best_k, best_score)。

    metric 默认 RMSE；targets 与 preds 键对齐。
    """
    from . import common as C

    if metric is None:
        metric = C.rmse

    best_k, best_score = 1.0, float("inf")
    for k in ks:
        score = np.mean(
            [metric(soft_threshold(preds[s], k), targets[s]) for s in preds]
        )
        if score < best_score:
            best_k, best_score = k, score
    return best_k, best_score
