"""寻找理论上限：完美背景估计能带来多大收益。

关键实验：如果背景估计是「完美的」（用同背景组的 ground truth 反推），
背景去除能到什么 RMSE？这决定了背景泄漏这条路的理论上限。

方法：
1. 找同背景组（用背景相似度）
2. 完美背景 = 组内每张图「干净图 + 已知噪声生成方式」反推
   实际上：脏图 = min(干净图, 背景)，所以背景 >= 脏图，且背景在「文字处」等于干净图的反推
   更简单：同背景组的图，其「背景」是共享的，可以用组内所有图的 max 近似
3. 但真正的理论上限是：如果能拿到真实背景 B，则 干净 = 脏图/B 的完美反演
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


def main():
    all_stems = C.list_image_stems(C.TRAIN_DIR)
    ys = {s: C.load_gray(C.CLEAN_DIR / f"{s}.png") for s in all_stems}
    xs = {s: C.load_gray(C.TRAIN_DIR / f"{s}.png") for s in all_stems}

    # 噪声生成方式：脏图 = min(干净图, 背景)  =>  背景 = 脏图，除了文字处（干净图<背景处，脏图=干净图）
    # 所以：背景 B >= 脏图 X 处处成立，且 B >= 干净图 Y
    # 实际背景 = 在"非文字"处 B = X = Y(接近1)，在"文字"处 B = X（因为 X=min(Y,B)=Y < B，但 B 未知）
    # 关键：B 无法从单图唯一确定，但可以从"同背景组"的多图确定：
    #   同背景组内，每张图的文字不同，所以在每个像素位置，至少有一张图该位置是"非文字"（B 暴露出来）
    #   所以 B = 组内所有图的 max（逐像素）—— 这就是组内 max 背景的理论依据

    # 验证：组内 max 背景 vs 真实背景的差距
    groups = L.group_by_background(all_stems, xs, threshold=0.03)
    print(f"背景分组 {len(groups)} 组")

    # 对每组，组内 max 背景，然后 x/bg，评估 RMSE
    rmses = []
    for g in groups:
        g_same = [s for s in g if xs[s].shape == xs[g[0]].shape]
        if len(g_same) < 2:
            # 单张组，用自身中值滤波背景
            for s in g_same:
                bg = L.estimate_background(xs[s], 25)
                out = np.clip(xs[s] / (bg + 1e-3), 0, 1)
                rmses.append(C.rmse(out, ys[s]))
            continue
        bg = L.group_background([xs[s] for s in g_same])
        for s in g_same:
            out = np.clip(xs[s] / (bg + 1e-3), 0, 1)
            rmses.append(C.rmse(out, ys[s]))

    print(f"\n组内max背景 + x/bg: RMSE = {np.mean(rmses):.5f}")

    # 理论上限：如果背景完美已知（用干净图反推 B = max(组内所有干净图) 因为背景处干净图=1）
    # 实际上，背景在文字处也是暗的，干净图无法直接给背景。
    # 真正的完美背景：用「同背景组 + 文字mask」迭代估计
    # 简化验证：组内 max 背景已经接近完美，看看它和"更激进"方法的差距

    # 对比：单图 x/bg（中值滤波25）
    r_single = []
    for s in all_stems:
        bg = L.estimate_background(xs[s], 25)
        out = np.clip(xs[s] / (bg + 1e-3), 0, 1)
        r_single.append(C.rmse(out, ys[s]))
    print(f"单图 x/bg (k=25): RMSE = {np.mean(r_single):.5f}")

    # 更大的中值核
    for k in [35, 45, 55, 75]:
        r = []
        for s in all_stems[:40]:
            bg = L.estimate_background(xs[s], k)
            out = np.clip(xs[s] / (bg + 1e-3), 0, 1)
            r.append(C.rmse(out, ys[s]))
        print(f"单图 x/bg (k={k}): RMSE = {np.mean(r):.5f}")


if __name__ == "__main__":
    main()
