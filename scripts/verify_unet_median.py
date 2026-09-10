"""决定性实验：V2 三模型集成预测 + 文字泄漏 median 合并。

用现有最优 ckpt（v2_seed0/1/2）对验证集预测，然后：
1. 用「预测结果」找同文字组（预测已经接近干净图，可用于分组）
2. 同文字组 median 合并
3. 对比前后 RMSE

这是判断「文字泄漏后处理」是否值得投入的决定性实验。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src import common as C
from src import leakage as L
from src.predict_v2 import load_model_v2, predict_image_v2_weighted


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # 载入 V2 三模型
    ckpts = [
        ROOT / "outputs/ckpts/v2_seed0.pt",
        ROOT / "outputs/ckpts/v2_seed1.pt",
        ROOT / "outputs/ckpts/v2_seed2.pt",
    ]
    models = []
    for c in ckpts:
        if c.exists():
            models.append(load_model_v2(c, device))
            print(f"载入 {c.name}")
    if not models:
        print("!! 未找到 V2 ckpt")
        return

    # 验证集划分（与训练一致）
    train_stems, val_stems = C.make_split(seed=42, val_fraction=0.13)
    xs = {s: C.load_gray(C.TRAIN_DIR / f"{s}.png") for s in val_stems}
    ys = {s: C.load_gray(C.CLEAN_DIR / f"{s}.png") for s in val_stems}

    # 预测
    weights = [0.35, 0.45, 0.20][: len(models)]
    preds = {}
    for s in val_stems:
        p = predict_image_v2_weighted(models, xs[s], device, weights, tta=True, n_tta=4)
        preds[s] = np.clip(p, 0, 1)

    base_rmse = np.mean([C.rmse(preds[s], ys[s]) for s in val_stems])
    print(f"\nV2 集成预测 RMSE: {base_rmse:.5f}")

    # 用预测结果找同文字组
    dup = L.find_duplicate_groups(val_stems, preds, threshold=0.03)
    multi = [g for g in dup if len(g) > 1]
    print(f"预测结果找同文字组: {len(multi)} 组")
    for g in multi:
        print(f"  {[int(s) for s in g]}")

    # median 合并
    final = {s: preds[s] for s in val_stems}
    for g in multi:
        g_same = [s for s in g if preds[s].shape == preds[g[0]].shape]
        if len(g_same) < 2:
            continue
        med = L.median_across_group([preds[s] for s in g_same])
        for s in g_same:
            final[s] = med

    final_rmse = np.mean([C.rmse(final[s], ys[s]) for s in val_stems])
    print(f"\nmedian 合并后 RMSE: {final_rmse:.5f}  (vs 合并前 {base_rmse:.5f})")
    print(f"提升: {base_rmse - final_rmse:.5f}")


if __name__ == "__main__":
    main()
