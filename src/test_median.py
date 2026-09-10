"""测试集文字泄漏后处理：同文字组 median 合并。

思路（测试时合法，仅用测试集自身信息）：
1. 对每张测试图做背景去除（x/bg），得到「文字前景」估计
2. 用「去背景图 + 模型预测」的相似度，找出同文字的图对
3. 同文字组内对「模型预测结果」做逐像素 median，消除独立残差

这是 Kaggle 冠军 Colin「第二个信息泄漏」的现代复现：同文字图重复，
median 平均掉独立噪声，使 RMSE 显著下降。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src import common as C
from src import leakage as L


def bg_removed(x: np.ndarray, ksize: int = 25) -> np.ndarray:
    """单图背景去除 x/bg。"""
    bg = L.estimate_background(x, ksize=ksize)
    return np.clip(x / (bg + 1e-3), 0.0, 1.0)


def group_similar(stems, feats, threshold=0.02):
    """按特征图相似度贪心分组（同尺寸桶内比较）。"""
    from collections import defaultdict

    buckets = defaultdict(list)
    for s in stems:
        buckets[feats[s].shape].append(s)

    groups = []
    for shape, members in buckets.items():
        remaining = list(members)
        while remaining:
            seed = remaining.pop(0)
            g = [seed]
            rm = []
            for o in remaining:
                r = C.rmse(feats[seed], feats[o])
                if r < threshold:
                    g.append(o)
                    rm.append(o)
            for o in rm:
                remaining.remove(o)
            groups.append(g)
    return groups


def median_merge_groups(stems, preds, groups):
    """对每组内预测做 median 合并。单张组保持原样。"""
    final = {}
    for s in stems:
        final[s] = preds[s]
    for g in groups:
        g_same = [s for s in g if preds[s].shape == preds[g[0]].shape]
        if len(g_same) < 2:
            continue
        med = L.median_across_group([preds[s] for s in g_same])
        for s in g_same:
            final[s] = med
    return final


def find_text_groups(stems, xs, preds, bg_thresh=0.02, pred_thresh=0.02):
    """综合「去背景图」和「模型预测」两种信号找同文字组。

    策略：取两种分组的并集（更保守，避免漏掉同文字对）。
    同文字要求：去背景图相似 AND/OR 预测图相似。
    """
    # 信号1：去背景图
    divs = {s: bg_removed(xs[s]) for s in stems}
    # 信号2：模型预测
    # 综合特征：两种图的平均
    feats = {s: 0.5 * divs[s] + 0.5 * preds[s] for s in stems}
    return group_similar(stems, feats, threshold=pred_thresh)


def apply_test_median(stems, xs, preds, threshold=0.02, verbose=True):
    """对测试集应用文字泄漏 median 后处理。

    Args:
        stems: 测试集 stem 列表
        xs: {stem: 脏图}
        preds: {stem: 模型预测}
        threshold: 同文字分组相似度阈值
    Returns:
        {stem: 最终预测}
    """
    groups = find_text_groups(stems, xs, preds, pred_thresh=threshold)
    if verbose:
        multi = [g for g in groups if len(g) > 1]
        print(f"[median] 测试集同文字组 {len(multi)} 组:")
        for g in multi:
            print(f"  {[int(s) for s in g]}")
    return median_merge_groups(stems, preds, groups)
