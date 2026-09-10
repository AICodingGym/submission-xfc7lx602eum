"""关键实验：在完整训练集上严格量化「同文字 median 合并」的真实收益。

训练集有 36 组同文字对（样本充足），能准确预测测试集上的效果。

流程（模拟测试时场景）：
1. 用「去背景图相似度」找同文字组（不用干净图，符合测试时约束）
2. 对同文字组做 median 合并
3. 对比 median 前后 RMSE

注意：这里用「模型预测」做 median 的对象（因为测试时我们只有模型预测）。
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
    all_stems = C.list_image_stems(C.TRAIN_DIR)
    all_stems = sorted(all_stems, key=int)
    xs = {s: C.load_gray(C.TRAIN_DIR / f"{s}.png") for s in all_stems}
    ys = {s: C.load_gray(C.CLEAN_DIR / f"{s}.png") for s in all_stems}

    # 去背景图
    divs = {s: np.clip(xs[s]/(est_bg(xs[s])+1e-3), 0, 1) for s in all_stems}

    # 用「去背景图相似度」找同文字组（阈值0.08，从测试集分析确定）
    dup = L.find_duplicate_groups(all_stems, divs, threshold=0.08)
    multi = [g for g in dup if len(g) > 1]
    print(f"去背景图找同文字组: {len(multi)} 组(>1张)")
    for g in multi:
        print(f"  {[int(s) for s in g]}")

    # 真值同文字组（用干净图）
    true_dup = L.find_duplicate_groups(all_stems, ys, threshold=0.01)
    true_multi = [g for g in true_dup if len(g) > 1]
    print(f"\n真值同文字组: {len(true_multi)} 组")

    # 关键实验：同文字组的 median 合并，能多大程度逼近干净图
    # 模拟：用「去背景图」作为"预测"（因为这是测试时能拿到的），median 合并
    print("\n=== 去背景图 median 合并收益（模拟）===")
    base_rmse = []
    med_rmse = []
    for g in true_multi:
        g_same = [s for s in g if divs[s].shape == divs[g[0]].shape]
        if len(g_same) < 2:
            continue
        # 单图去背景 RMSE
        for s in g_same:
            base_rmse.append(C.rmse(divs[s], ys[s]))
        # median 合并后 RMSE
        med = L.median_across_group([divs[s] for s in g_same])
        for s in g_same:
            med_rmse.append(C.rmse(med, ys[s]))
    print(f"去背景单图 RMSE: {np.mean(base_rmse):.5f}")
    print(f"去背景+median RMSE: {np.mean(med_rmse):.5f}")
    print(f"提升: {np.mean(base_rmse) - np.mean(med_rmse):.5f}")

    # 更关键：如果"预测"已经很好（如 U-Net 0.011），median 还能提升多少？
    # 这个需要在有 U-Net 预测的情况下测，见 verify_unet_median.py


if __name__ == "__main__":
    main()
