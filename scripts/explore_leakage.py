"""验证「背景泄漏」假设：
1. 训练集是否只有 8 种背景（每 8 张循环）
2. 中值滤波背景估计 + RMSE 分组能否还原分组
3. 除法归一化 (x/bg) vs 减法 (x-bg) 哪个更接近干净图
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src import common as C


def estimate_background(x, ksize=25):
    """大核中值滤波估计背景（去除文字，保留低频污渍）。"""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur((x * 255.0).astype(np.uint8), ksize).astype(np.float32) / 255.0


def _align(a, b):
    """把两张图对齐到相同形状（resize 到较小尺寸）用于 RMSE 比较。"""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if a.shape != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    if b.shape != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def main():
    stems = C.list_image_stems(C.TRAIN_DIR)
    print(f"训练集共 {len(stems)} 张")

    # 1. 计算每张图的背景估计（中值滤波）
    bgs = {}
    xs = {}
    ys = {}
    for s in stems:
        xs[s] = C.load_gray(C.TRAIN_DIR / f"{s}.png")
        ys[s] = C.load_gray(C.CLEAN_DIR / f"{s}.png")
        bgs[s] = estimate_background(xs[s], ksize=25)
    print("背景估计完成")

    # 2. 验证「每 8 张背景重复」假设
    stems_sorted = sorted(stems, key=int)
    n = len(stems_sorted)
    print("\n=== 验证「每 8 张背景重复」假设（对齐后比较）===")
    for gap in [4, 8, 12, 16]:
        rmses = []
        for i in range(n - gap):
            a, b = _align(bgs[stems_sorted[i]], bgs[stems_sorted[i + gap]])
            rmses.append(C.rmse(a, b))
        print(f"gap={gap}: 背景RMSE mean={np.mean(rmses):.5f} min={np.min(rmses):.5f} max={np.max(rmses):.5f}")

    # 3. 对比：单图除法归一化 vs 减法，哪个更接近干净图
    print("\n=== 对比归一化方法（单图背景估计，ksize=25）===")
    rmse_div = []
    rmse_sub = []
    rmse_raw = []
    for s in stems:
        x = xs[s]
        y = ys[s]
        bg = bgs[s]
        eps = 1e-3
        # 除法归一化
        out_div = np.clip(x / (bg + eps), 0, 1)
        # 减法归一化（白平衡）
        out_sub = np.clip((x - bg) / (1 - bg + eps), 0, 1)
        rmse_div.append(C.rmse(out_div, y))
        rmse_sub.append(C.rmse(out_sub, y))
        rmse_raw.append(C.rmse(x, y))
    print(f"原始脏图 RMSE:        {np.mean(rmse_raw):.5f}")
    print(f"除法归一化 (x/bg):    {np.mean(rmse_div):.5f}")
    print(f"减法归一化 (x-bg):    {np.mean(rmse_sub):.5f}")

    # 4. 尝试组内 max 背景（需要先按相似度聚类，而不是假设每8张一组）
    # 先看：按数值顺序相邻图，到底哪几张背景相同。打印完整相似度，做简单贪心聚类
    print("\n=== 背景聚类分析（贪心，阈值 0.03）===")
    # 构建相似度：同尺寸才能比，这里只对同尺寸分组比较
    def same_shape(s):
        return xs[s].shape

    # 按形状分桶
    from collections import defaultdict
    buckets = defaultdict(list)
    for s in stems_sorted:
        buckets[same_shape(s)].append(s)

    # 在每个尺寸桶内做贪心聚类
    groups = []
    for shape, members in buckets.items():
        remaining = list(members)
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            # 找与 seed 背景 RMSE < 阈值的
            to_remove = []
            for other in remaining:
                r = C.rmse(bgs[seed], bgs[other])
                if r < 0.03:
                    group.append(other)
                    to_remove.append(other)
            for o in to_remove:
                remaining.remove(o)
            groups.append(group)

    print(f"聚类得到 {len(groups)} 组背景")
    for i, g in enumerate(groups):
        print(f"  组{i}: {len(g)} 张  stems={[int(s) for s in g]}")

    # 组内 max 背景 + 除法
    print("\n=== 组内 max 背景 + 除法 ===")
    rmse_group_div = []
    for g in groups:
        if len(g) < 2:
            continue
        stacked = np.stack([xs[s] for s in g], axis=0)
        bg_group = stacked.max(axis=0)
        for s in g:
            out = np.clip(xs[s] / (bg_group + 1e-3), 0, 1)
            rmse_group_div.append(C.rmse(out, ys[s]))
    print(f"组内max背景 + 除法: {np.mean(rmse_group_div):.5f}  (vs 单图除法 0.05069)")


if __name__ == "__main__":
    main()
