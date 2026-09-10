"""纠正方向：median 应该用在「同文字」组，不是「同背景」组。

关键：数据集是 8背景 × N文字 交叉组合。
- 同背景图：文字不同 -> median 会糊文字（错误）
- 同文字图：背景不同 -> median 能去掉背景差异（正确！）

但问题：测试时没有干净图，怎么找「同文字」组？
答案：用「去背景后的图」找同文字组（去背景后只剩文字，文字相同的图去背景结果相似）。
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

    # 单图去背景
    a = {s: np.clip(xs[s]/(est_bg(xs[s])+1e-3), 0, 1) for s in val_stems}
    ra = np.mean([C.rmse(a[s], ys[s]) for s in val_stems])
    print(f"单图 x/bg: {ra:.5f}")

    # 用「去背景后的图」找同文字组
    # 去背景后文字对比度恢复，同文字应该相似
    dup = L.find_duplicate_groups(val_stems, a, threshold=0.05)
    multi = [g for g in dup if len(g) > 1]
    print(f"用去背景图找同文字组: {len(multi)} 组(>1张)")
    for g in multi:
        print(f"  {[int(s) for s in g]}")

    # 对这些组做 median 合并
    final = {s: a[s] for s in val_stems}
    for g in multi:
        g_same = [s for s in g if a[s].shape == a[g[0]].shape]
        if len(g_same) < 2:
            continue
        med = L.median_across_group([a[s] for s in g_same])
        for s in g_same:
            final[s] = med
    rf = np.mean([C.rmse(final[s], ys[s]) for s in val_stems])
    print(f"\n单图 {ra:.5f} -> 同文字median {rf:.5f} (提升 {ra-rf:.5f})")


if __name__ == "__main__":
    main()
