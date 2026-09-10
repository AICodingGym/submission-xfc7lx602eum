"""验证完整链条：背景去除 -> median 文字合并 的组合威力。

正确顺序：
1. 每组（同背景）先做背景去除，得到「文字前景」
2. 同文字组 median 合并，消除随机噪声
3. 映射回灰度 [0,1]

关键：背景污渍是确定性的（同背景组共享），median 去不掉；
但文字内容相同的图，噪声是独立的，median 能平均掉。
所以正确顺序是「先去背景（去确定性污渍），再 median（去随机噪声）」。
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


def bg_removed_to_gray(f, bg):
    """把 (I-B)/B 的前景映射回灰度空间。

    Colin 的做法是：前景 F = (I-B)/B 表示「文字相对背景的对比度」，
    干净的灰度图 = 1 + F（文字处 F<0 变暗，背景处 F≈0 变白）。
    更准确：干净图 clean ≈ clip(1 + F) 但文字处需要更暗。
    实际应该用 bg 恢复：clean = clip(bg * (1 + F)) = clip(bg + (I-B)) = clip(I)
    这回到原图了，不对。
    
    正确理解：Colin 的 (I-B)/B 是「特征」给后续模型用的。
    但我们可以直接用更简单的物理模型：
    干净图 = 文字mask（0或1）× 背景 + 白纸。
    实际上干净图只有两个值域：白纸(≈1) 和 黑字(≈0附近)。
    """
    # 简化：文字前景 = clip(1 + F, 0, 1)，即对比度归一化后的"增强版"图
    return np.clip(1.0 + f, 0.0, 1.0)


def main():
    all_stems = C.list_image_stems(C.TRAIN_DIR)
    ys_all = {s: C.load_gray(C.CLEAN_DIR / f"{s}.png") for s in all_stems}
    xs_all = {s: C.load_gray(C.TRAIN_DIR / f"{s}.png") for s in all_stems}

    # 1. 背景分组（8种背景）
    groups = L.group_by_background(all_stems, xs_all, threshold=0.03)
    print(f"背景分组: {len(groups)} 组")

    # 2. 每组做背景去除
    div = {}
    for g in groups:
        if len(g) < 2:
            for s in g:
                div[s] = L.remove_background_self(xs_all[s], ksize=25)
            continue
        bg = L.group_background([xs_all[s] for s in g])
        for s in g:
            div[s] = L.remove_background(xs_all[s], bg)

    # 3. 映射回灰度，评估单图去背景的 RMSE
    gray = {s: bg_removed_to_gray(div[s], None) for s in all_stems}
    r_bg = np.mean([C.rmse(gray[s], ys_all[s]) for s in all_stems])
    print(f"背景去除后(单图) RMSE: {r_bg:.5f}")

    # 4. 文字泄漏：同文字组 median 合并「去背景后的灰度」
    dup_groups = L.find_duplicate_groups(all_stems, ys_all, threshold=0.01)
    multi = [g for g in dup_groups if len(g) > 1]
    print(f"同文字组(>1张): {len(multi)} 组")

    final = {s: gray[s] for s in all_stems}
    for g in multi:
        # 同尺寸才可 median
        g_same = [s for s in g if gray[s].shape == gray[g[0]].shape]
        if len(g_same) < 2:
            continue
        med = L.median_across_group([gray[s] for s in g_same])
        for s in g_same:
            final[s] = med

    r_final = np.mean([C.rmse(final[s], ys_all[s]) for s in all_stems])
    print(f"背景去除 + median 合并 RMSE: {r_final:.5f}")
    print(f"  单图背景去除 {r_bg:.5f} -> 加median {r_final:.5f} (提升 {r_bg - r_final:.5f})")


if __name__ == "__main__":
    main()
