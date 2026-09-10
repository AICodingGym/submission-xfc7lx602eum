"""验证第二个信息泄漏：测试集（或训练集）里是否有「文字内容重复」的图。

方法：对每张图做中值滤波去除文字后，用剩余「文字 mask」比对；
或者更直接：比较干净图（train_cleaned）之间的相似度，找出几乎相同的图。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src import common as C


def _align(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if a.shape != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    if b.shape != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def main():
    # 1. 训练集干净图之间的相似度（找出「同文字」的图）
    stems = C.list_image_stems(C.CLEAN_DIR)
    ys = {}
    for s in stems:
        ys[s] = C.load_gray(C.CLEAN_DIR / f"{s}.png")

    stems_sorted = sorted(stems, key=int)
    n = len(stems_sorted)

    print("=== 训练集干净图相似度（找出同文字图）===")
    # 二值化后比较，突出「文字内容」而非灰度
    def binarize(y, thresh=0.5):
        return (y < thresh).astype(np.float32)  # 文字=1

    yb = {s: binarize(ys[s]) for s in stems}

    # 两两比较（只比较同尺寸），找几乎相同的对
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = stems_sorted[i], stems_sorted[j]
            if yb[si].shape != yb[sj].shape:
                continue
            diff = np.mean(np.abs(yb[si] - yb[sj]))
            if diff < 0.01:  # 文字 mask 几乎一致
                pairs.append((int(si), int(sj), diff))

    print(f"找到 {len(pairs)} 对「几乎相同文字」的训练图:")
    for a, b, d in pairs:
        print(f"  {a} <-> {b}  (diff={d:.6f})")

    # 2. 测试集内部：中值滤波后比对，找同背景/同文字
    print("\n=== 测试集背景相似度 ===")
    tstems = C.list_image_stems(C.TEST_DIR)
    txs = {s: C.load_gray(C.TEST_DIR / f"{s}.png") for s in tstems}
    tstems_sorted = sorted(tstems, key=int)
    m = len(tstems_sorted)

    def bg_est(x, k=25):
        return cv2.medianBlur((x * 255).astype(np.uint8), k).astype(np.float32) / 255.0

    tbgs = {s: bg_est(txs[s]) for s in tstems}

    # 打印测试集两两背景 RMSE（找重复背景组）
    print("测试集背景两两 RMSE < 0.03 的对：")
    cnt = 0
    for i in range(m):
        for j in range(i + 1, m):
            si, sj = tstems_sorted[i], tstems_sorted[j]
            if tbgs[si].shape != tbgs[sj].shape:
                continue
            r = C.rmse(tbgs[si], tbgs[sj])
            if r < 0.03:
                print(f"  {int(si)} <-> {int(sj)}  (bg_rmse={r:.5f})")
                cnt += 1
    print(f"共 {cnt} 对")


if __name__ == "__main__":
    main()
