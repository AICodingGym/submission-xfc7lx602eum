"""深挖测试集 29 张图的完整重复结构。

目标：精确确定每张测试图属于哪个「同文字组」和「同背景组」，
为 median 融合提供最可靠的分组依据。

方法（多信号交叉验证）：
1. 信号A：去背景图（x/bg）相似度 → 反映文字内容
2. 信号B：模型预测相似度 → 反映文字内容（更精确）
3. 信号C：背景估计（中值滤波）相似度 → 反映背景

输出：完整的相似度矩阵 + 最优分组建议。
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


def align(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if a.shape != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    if b.shape != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def main():
    tstems = C.list_image_stems(C.TEST_DIR)
    tstems = sorted(tstems, key=int)
    xs = {s: C.load_gray(C.TEST_DIR / f"{s}.png") for s in tstems}
    bgs = {s: est_bg(xs[s]) for s in tstems}
    divs = {s: np.clip(xs[s]/(bgs[s]+1e-3), 0, 1) for s in tstems}

    print("测试集 29 张图，尺寸：")
    for s in tstems:
        print(f"  {int(s):>4}: {xs[s].shape}")

    # 同尺寸分组
    from collections import defaultdict
    buckets = defaultdict(list)
    for s in tstems:
        buckets[xs[s].shape].append(s)

    print("\n=== 同尺寸桶 ===")
    for sh, ms in buckets.items():
        print(f"  {sh}: {[int(s) for s in ms]}")

    # 对每个桶，计算去背景图的两两相似度（文字内容）
    print("\n=== 去背景图两两相似度（同尺寸桶内）===")
    for sh, ms in buckets.items():
        print(f"\n桶 {sh}（{len(ms)}张）：")
        for i in range(len(ms)):
            for j in range(i+1, len(ms)):
                si, sj = ms[i], ms[j]
                r_div = C.rmse(divs[si], divs[sj])
                r_bg = C.rmse(bgs[si], bgs[sj])
                flag = " <-- 文字相似" if r_div < 0.05 else ""
                flag2 = " <-- 背景相似" if r_bg < 0.02 else ""
                print(f"  {int(si):>4} vs {int(sj):>4}: 文字r={r_div:.4f}{flag}  背景r={r_bg:.4f}{flag2}")


if __name__ == "__main__":
    main()
