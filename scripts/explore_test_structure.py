"""深入测试集结构：29张测试图，找出「同文字」和「同背景」的完整关系。

这是最终预测目标，必须彻底搞清楚测试集的重复结构。
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
    tstems = C.list_image_stems(C.TEST_DIR)
    txs = {s: C.load_gray(C.TEST_DIR / f"{s}.png") for s in tstems}
    tstems = sorted(tstems, key=int)
    print(f"测试集 {len(tstems)} 张，尺寸分布：")
    shapes = {}
    for s in tstems:
        shapes[txs[s].shape] = shapes.get(txs[s].shape, 0) + 1
    for sh, cnt in shapes.items():
        print(f"  {sh}: {cnt} 张")

    # 测试集内「同背景」分组（背景相似度）
    tbgs = {s: est_bg(txs[s]) for s in tstems}
    print("\n=== 测试集背景分组（贪心，阈值0.02）===")
    # 按尺寸分桶
    from collections import defaultdict
    buckets = defaultdict(list)
    for s in tstems:
        buckets[txs[s].shape].append(s)
    groups = []
    for shape, members in buckets.items():
        remaining = list(members)
        while remaining:
            seed = remaining.pop(0)
            g = [seed]
            rm = []
            for o in remaining:
                if tbgs[seed].shape == tbgs[o].shape:
                    r = C.rmse(tbgs[seed], tbgs[o])
                    if r < 0.02:
                        g.append(o)
                        rm.append(o)
            for o in rm:
                remaining.remove(o)
            groups.append(g)

    print(f"共 {len(groups)} 组：")
    for i, g in enumerate(groups):
        print(f"  组{i} ({len(g)}张): {[int(s) for s in g]}")

    # 测试集内「同文字」：用去背景后的图找
    print("\n=== 测试集内同文字组（去背景后相似）===")
    divs = {s: np.clip(txs[s]/(tbgs[s]+1e-3),0,1) for s in tstems}
    dup = L.find_duplicate_groups(tstems, divs, threshold=0.03)
    multi = [g for g in dup if len(g) > 1]
    print(f"同文字组 {len(multi)} 组：")
    for g in multi:
        print(f"  {[int(s) for s in g]}")


if __name__ == "__main__":
    main()
