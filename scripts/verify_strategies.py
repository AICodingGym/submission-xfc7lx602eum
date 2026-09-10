"""聚焦实验：找到真正有效的泄漏利用方式。

在训练集上完整模拟「测试时只有脏图」的场景，评估不同策略的 RMSE。

策略对比（都在 val 集评估）：
A. 单图背景去除 (x/bg)
B. 同背景组内 median 直接作为最终预测（背景+文字一起 median）
C. 同背景组：先各自去背景，再 median
D. 同背景组：组内 max 背景，再除法，再 median
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


def est_bg(x, k=25):
    return cv2.medianBlur((np.clip(x,0,1)*255).astype(np.uint8), k).astype(np.float32)/255.0


def main():
    train_stems, val_stems = C.make_split(seed=42, val_fraction=0.13)
    xs = {s: C.load_gray(C.TRAIN_DIR / f"{s}.png") for s in val_stems}
    ys = {s: C.load_gray(C.CLEAN_DIR / f"{s}.png") for s in val_stems}

    # 背景分组（仅用脏图）
    groups = L.group_by_background(val_stems, xs, threshold=0.03)
    print(f"val 背景分组: {len(groups)} 组")

    # A. 单图 x/bg
    a = {s: np.clip(xs[s]/(est_bg(xs[s])+1e-3), 0, 1) for s in val_stems}
    ra = np.mean([C.rmse(a[s], ys[s]) for s in val_stems])
    print(f"A 单图 x/bg: {ra:.5f}")

    # B. 同背景组直接 median（背景+文字一起）
    b = {}
    for g in groups:
        g_same = [s for s in g if xs[s].shape == xs[g[0]].shape]
        if len(g_same) >= 2:
            med = L.median_across_group([xs[s] for s in g_same])
            for s in g_same:
                b[s] = med
        else:
            for s in g_same:
                b[s] = xs[s]
    rb = np.mean([C.rmse(b[s], ys[s]) for s in val_stems])
    print(f"B 组内 median(脏图): {rb:.5f}")

    # C. 各自去背景 x/bg 后 median
    c = {}
    for g in groups:
        g_same = [s for s in g if xs[s].shape == xs[g[0]].shape]
        if len(g_same) >= 2:
            med = L.median_across_group([a[s] for s in g_same])
            for s in g_same:
                c[s] = med
        else:
            for s in g_same:
                c[s] = a[s]
    rc = np.mean([C.rmse(c[s], ys[s]) for s in val_stems])
    print(f"C 去背景后 median: {rc:.5f}")

    # D. 组内 max 背景 + 除法 + median
    d = {}
    for g in groups:
        g_same = [s for s in g if xs[s].shape == xs[g[0]].shape]
        if len(g_same) >= 2:
            bg = L.group_background([xs[s] for s in g_same])
            divs = [np.clip(xs[s]/(bg+1e-3),0,1) for s in g_same]
            med = L.median_across_group(divs)
            for s in g_same:
                d[s] = med
        else:
            for s in g_same:
                d[s] = a[s]
    rd = np.mean([C.rmse(d[s], ys[s]) for s in val_stems])
    print(f"D max背景除法+median: {rd:.5f}")

    print(f"\n=== 结论：原始脏图 RMSE = {np.mean([C.rmse(xs[s],ys[s]) for s in val_stems]):.5f} ===")


if __name__ == "__main__":
    main()
