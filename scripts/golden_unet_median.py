"""黄金实验：U-Net 集成预测 + 同文字 median，在完整训练集上的真实收益。

这是预测测试集效果的最准确方法（训练集36组同文字对 vs 测试集8组）。

流程：
1. V2 三模型集成预测完整训练集（模拟测试时）
2. 用「去背景图」找同文字组（36组，已验证100%准确）
3. 同文字组 median 合并预测
4. 对比 RMSE

注意：这里用「去背景图」分组（测试时可用），而不是用「干净图」分组。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from src import common as C
from src import leakage as L
from src.predict_v2 import load_model_v2, predict_image_v2_weighted


def est_bg(x, k=25):
    return cv2.medianBlur((np.clip(x,0,1)*255).astype(np.uint8), k).astype(np.float32)/255.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = [0.35, 0.45, 0.20]

    all_stems = C.list_image_stems(C.TRAIN_DIR)
    all_stems = sorted(all_stems, key=int)
    xs = {s: C.load_gray(C.TRAIN_DIR / f"{s}.png") for s in all_stems}
    ys = {s: C.load_gray(C.CLEAN_DIR / f"{s}.png") for s in all_stems}

    # 去背景图（用于分组）
    divs = {s: np.clip(xs[s]/(est_bg(xs[s])+1e-3), 0, 1) for s in all_stems}

    # 预测：逐模型累加，避免同时驻留3个模型
    print("预测完整训练集（逐模型累加）...")
    acc = {s: np.zeros_like(xs[s], dtype=np.float64) for s in all_stems}
    for mi in range(3):
        model = load_model_v2(ROOT/f"outputs/ckpts/v2_seed{mi}.pt", device)
        print(f"  模型 {mi} 预测中...")
        for i, s in enumerate(all_stems):
            p = predict_image_v2_weighted([model], xs[s], device, [1.0], tta=True, n_tta=4)
            acc[s] += weights[mi] * np.clip(p, 0, 1)
            if (i+1) % 40 == 0:
                print(f"    {i+1}/{len(all_stems)}")
        del model
        torch.cuda.empty_cache()

    preds = {s: np.clip(acc[s].astype(np.float32), 0, 1) for s in all_stems}
    base = np.mean([C.rmse(preds[s], ys[s]) for s in all_stems])
    print(f"\nU-Net 集成预测 RMSE（全训练集）: {base:.5f}")

    # 同文字组（用去背景图）
    dup = L.find_duplicate_groups(all_stems, divs, threshold=0.08)
    multi = [g for g in dup if len(g) > 1]
    print(f"同文字组: {len(multi)} 组")

    # median 合并
    final = {s: preds[s] for s in all_stems}
    for g in multi:
        g_same = [s for s in g if preds[s].shape == preds[g[0]].shape]
        if len(g_same) < 2:
            continue
        med = L.median_across_group([preds[s] for s in g_same])
        for s in g_same:
            final[s] = med

    final_rmse = np.mean([C.rmse(final[s], ys[s]) for s in all_stems])
    print(f"median 合并后 RMSE: {final_rmse:.5f}")
    print(f"提升: {base - final_rmse:.5f}  ({(base-final_rmse)/base*100:.1f}%)")


if __name__ == "__main__":
    main()
