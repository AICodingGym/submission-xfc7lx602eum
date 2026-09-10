"""信息泄漏利用模块：复现 Kaggle 冠军 Colin 的完整方案核心。

两个信息泄漏：
1. 背景泄漏：只有 8 种背景，同背景图可互相估计真实背景。
   方法：中值滤波(25)估计背景 -> RMSE 相似度分组 -> 组内 max 背景
   -> 除法归一化 F = (I - B) / B（Colin 的对比度修正公式）。
2. 文字泄漏：同文字/同字体的图重复出现，逐像素 median 可消除噪声，
   使 RMSE 减半（这是把分数从 0.01 级降到 0.005 级的关键）。

本模块提供纯函数，供 dataset 输入通道构造与最终提交后处理使用。
"""
from __future__ import annotations

import cv2
import numpy as np

from . import common as C


# ---------------------------------------------------------------------------
# 背景估计
# ---------------------------------------------------------------------------
def estimate_background(x: np.ndarray, ksize: int = 25) -> np.ndarray:
    """大核中值滤波估计背景（去除文字，保留低频污渍/光照）。

    ksize=25 是 Colin 原文推荐的核大小，可捕获背景宽泛模式并移除文字笔画。
    """
    if ksize % 2 == 0:
        ksize += 1
    return (
        cv2.medianBlur((np.clip(x, 0, 1) * 255.0).astype(np.uint8), ksize).astype(np.float32)
        / 255.0
    )


def background_group_rmse(bg_a: np.ndarray, bg_b: np.ndarray) -> float:
    """两张背景估计图的对齐 RMSE 相似度（不同尺寸先 resize 到较小尺寸）。"""
    a, b = _align(bg_a, bg_b)
    return C.rmse(a, b)


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if a.shape != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    if b.shape != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def group_by_background(
    stems: list[str],
    images: dict[str, np.ndarray],
    threshold: float = 0.03,
) -> list[list[str]]:
    """按背景相似度对 stems 贪心聚类，返回若干组（每组同背景）。

    仅在同尺寸桶内比较（混合尺寸无法直接比 RMSE）。
    """
    from collections import defaultdict

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for s in stems:
        buckets[images[s].shape].append(s)

    groups: list[list[str]] = []
    for shape, members in buckets.items():
        remaining = list(members)
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            to_remove = []
            for other in remaining:
                if background_group_rmse(
                    estimate_background(images[seed]),
                    estimate_background(images[other]),
                ) < threshold:
                    group.append(other)
                    to_remove.append(other)
            for o in to_remove:
                remaining.remove(o)
            groups.append(group)
    return groups


def group_background(images: list[np.ndarray]) -> np.ndarray:
    """组内 max 背景：每个像素取组内最亮值作为真实背景估计。

    原理：噪声通过 min(背景, 文字) 合成，最亮处背景最接近真实背景。
    要求组内所有图同尺寸。
    """
    stacked = np.stack(images, axis=0)
    return stacked.max(axis=0)


def remove_background(x: np.ndarray, bg: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Colin 的对比度修正公式：F = (I - B) / B。

    与简单 x/bg 的区别：先减再除，把暗污渍区域的文字对比度重新拉回一致。
    """
    f = (x - bg) / (bg + eps)
    return np.clip(f, -1.0, 1.0)


def remove_background_self(x: np.ndarray, ksize: int = 25, eps: float = 1e-3) -> np.ndarray:
    """单图背景去除（无分组，用自身中值滤波背景）。"""
    bg = estimate_background(x, ksize=ksize)
    return remove_background(x, bg, eps)


# ---------------------------------------------------------------------------
# 文字泄漏利用
# ---------------------------------------------------------------------------
def median_across_group(images: list[np.ndarray]) -> np.ndarray:
    """对同文字的一组预测/干净图做逐像素 median，消除独立噪声。

    要求组内所有图同尺寸。这是把 RMSE 减半的关键后处理。
    """
    stacked = np.stack(images, axis=0)
    return np.median(stacked, axis=0)


def find_duplicate_groups(
    stems: list[str],
    images: dict[str, np.ndarray],
    threshold: float = 0.01,
) -> list[list[str]]:
    """找出「文字内容几乎相同」的图分组（用于文字泄漏 median 合并）。

    用二值化 mask（文字=1）比较，diff < threshold 视为同文字。
    仅同尺寸比较。
    """
    from collections import defaultdict

    bins = defaultdict(list)
    for s in stems:
        bins[images[s].shape].append(s)

    def binarize(y):
        return (y < 0.5).astype(np.float32)

    # 用并查集合并「相同文字」的图
    parent = {s: s for s in stems}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for shape, members in bins.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                si, sj = members[i], members[j]
                diff = np.mean(np.abs(binarize(images[si]) - binarize(images[sj])))
                if diff < threshold:
                    union(si, sj)

    groups_map: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        groups_map[find(s)].append(s)

    groups = [sorted(g, key=int) for g in groups_map.values()]
    groups.sort(key=lambda g: int(g[0]))
    return groups
