"""重新理解泄漏的威力：分开验证每个环节的正确作用。

关键澄清：
1. (I-B)/B 是「特征」，不是最终灰度输出。它把文字对比度归一化，但值域是 [-1,1] 附近。
2. 真正的最终输出是灰度 [0,1]（0=黑字 1=白纸）。
3. Stage 4 median 泄漏：作用于「同文字的干净图」，直接降噪。

本实验验证：
A. 同文字图（干净图）median 合并 vs 单图，噪声差异有多大
B. 正确的背景去除输出应该是什么形态
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src import common as C
from src import leakage as L


def main():
    train_stems, val_stems = C.make_split(seed=42, val_fraction=0.13)

    # ============ A. 文字泄漏的威力 ============
    # 在训练集全部115张里找同文字组（因为val只有15张，同文字组太少）
    all_stems = C.list_image_stems(C.TRAIN_DIR)
    ys_all = {s: C.load_gray(C.CLEAN_DIR / f"{s}.png") for s in all_stems}
    xs_all = {s: C.load_gray(C.TRAIN_DIR / f"{s}.png") for s in all_stems}

    # 找同文字组（用干净图）
    dup_groups = L.find_duplicate_groups(all_stems, ys_all, threshold=0.01)
    multi = [g for g in dup_groups if len(g) > 1]
    print(f"训练集同文字组(>1张): {len(multi)} 组")

    # 对每组：脏图 median vs 干净图 median
    # 关键实验：同文字组，脏图的中值滤波/median 合并后，是否接近干净图
    # 因为噪声是独立加的，median(脏图) 应该接近 median(干净图) = 干净图本身
    print("\n=== 验证：median(脏图) 能否逼近 干净图 ===")
    gains = []
    for g in multi[:10]:
        g = [s for s in g if xs_all[s].shape == ys_all[g[0]].shape]
        if len(g) < 2:
            continue
        med_dirty = L.median_across_group([xs_all[s] for s in g])
        # 该组的干净图（都相同文字，取一张即可）
        clean = ys_all[g[0]]
        # 单图脏图 RMSE（取第一张）
        single_rmse = C.rmse(xs_all[g[0]], clean)
        # median 后 RMSE
        med_rmse = C.rmse(med_dirty, clean)
        gains.append((single_rmse, med_rmse))
        print(f"  组[{g}] 单图脏RMSE={single_rmse:.5f} median后={med_rmse:.5f} 提升={single_rmse-med_rmse:.5f}")

    if gains:
        avg_single = np.mean([g[0] for g in gains])
        avg_med = np.mean([g[1] for g in gains])
        print(f"\n平均: 单图脏图 {avg_single:.5f} -> median合并 {avg_med:.5f} (提升 {avg_single-avg_med:.5f})")


if __name__ == "__main__":
    main()
