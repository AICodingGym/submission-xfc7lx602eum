"""端到端验证完整泄漏方案在验证集上的 RMSE。

核心实验：
1. 用 make_split 划分 train/val（与训练口径一致）
2. 在 val 集上，模拟测试时的情形：只给脏图（不给干净图）
3. 用「背景泄漏 + 文字泄漏」重建干净图，看 RMSE 能否到 0.007 级别

关键：验证「同文字图 median 合并」在真实数据上的减半效果。
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
    # 用与训练一致的划分（seed=42, val 13%）
    train_stems, val_stems = C.make_split(seed=42, val_fraction=0.13)
    print(f"train={len(train_stems)} val={len(val_stems)}")

    # 载入 val 的脏图和干净图
    xs = {s: C.load_gray(C.TRAIN_DIR / f"{s}.png") for s in val_stems}
    ys = {s: C.load_gray(C.CLEAN_DIR / f"{s}.png") for s in val_stems}

    # 基线：原始脏图 RMSE
    base = np.mean([C.rmse(xs[s], ys[s]) for s in val_stems])
    print(f"\n原始脏图 RMSE: {base:.5f}")

    # Stage 1: 单图背景去除 (I-B)/B
    div = {}
    for s in val_stems:
        div[s] = L.remove_background_self(xs[s], ksize=25)
    r1 = np.mean([C.rmse(div[s], ys[s]) for s in val_stems])
    print(f"单图背景去除 (I-B)/B RMSE: {r1:.5f}")

    # Stage 1+: 分组背景去除（在 val 内部按背景分组）
    groups = L.group_by_background(val_stems, xs, threshold=0.03)
    print(f"\n背景分组: {len(groups)} 组")
    div_grp = {}
    for g in groups:
        if len(g) < 2:
            # 单张组用自身背景
            for s in g:
                div_grp[s] = div[s]
            continue
        bg = L.group_background([xs[s] for s in g])
        for s in g:
            div_grp[s] = L.remove_background(xs[s], bg)
    r2 = np.mean([C.rmse(div_grp[s], ys[s]) for s in val_stems])
    print(f"分组背景去除 RMSE: {r2:.5f}")

    # Stage 4: 文字泄漏 median 合并
    # 关键：用「去背景后的图」找同文字组，再 median 合并
    dup_groups = L.find_duplicate_groups(val_stems, ys, threshold=0.01)
    print(f"\n文字重复分组: {len(dup_groups)} 组 (含单张组)")
    # 只看 >1 张的组
    multi = [g for g in dup_groups if len(g) > 1]
    print(f"  其中 >1 张的组: {len(multi)} 组")

    # 对同文字组：median 合并去背景图，然后和干净图比
    # 注意：median 是「最终预测」，我们对每组用 median(去背景图) 作为统一预测
    final = {}
    for g in dup_groups:
        if len(g) == 1:
            final[g[0]] = div_grp[g[0]]
        else:
            med = L.median_across_group([div_grp[s] for s in g])
            for s in g:
                final[s] = med

    r4 = np.mean([C.rmse(final[s], ys[s]) for s in val_stems])
    print(f"\n文字泄漏 median 合并后 RMSE: {r4:.5f}")
    print(f"  vs 分组背景去除 {r2:.5f} -> 提升 {r2 - r4:.5f}")

    # 但注意：median 用的是 div_grp（去背景图），还需要 clip 回 [0,1] 和可能的后处理
    # 先看 (I-B)/B 的数值范围，决定怎么映射回 [0,1] 灰度
    all_vals = np.concatenate([div_grp[s].ravel() for s in val_stems[:5]])
    print(f"\n(I-B)/B 数值范围: min={all_vals.min():.3f} max={all_vals.max():.3f} "
          f"mean={all_vals.mean():.3f}")


if __name__ == "__main__":
    main()
