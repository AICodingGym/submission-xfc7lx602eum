"""验证 test_median 后处理在验证集上的效果 + 分组准确性。

关键：验证「去背景图 + 预测」综合信号能否正确找到同文字组。
用验证集的 ground truth（干净图）作为「真值分组」对照。
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
from src import test_median as TM
from src.predict_v2 import load_model_v2, predict_image_v2_weighted


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = [
        ROOT / "outputs/ckpts/v2_seed0.pt",
        ROOT / "outputs/ckpts/v2_seed1.pt",
        ROOT / "outputs/ckpts/v2_seed2.pt",
    ]
    models = [load_model_v2(c, device) for c in ckpts if c.exists()]
    weights = [0.35, 0.45, 0.20][: len(models)]

    train_stems, val_stems = C.make_split(seed=42, val_fraction=0.13)
    xs = {s: C.load_gray(C.TRAIN_DIR / f"{s}.png") for s in val_stems}
    ys = {s: C.load_gray(C.CLEAN_DIR / f"{s}.png") for s in val_stems}

    preds = {}
    for s in val_stems:
        preds[s] = np.clip(
            predict_image_v2_weighted(models, xs[s], device, weights, tta=True, n_tta=4), 0, 1
        )

    base = np.mean([C.rmse(preds[s], ys[s]) for s in val_stems])
    print(f"V2 集成 RMSE: {base:.5f}")

    # 真值同文字组（用干净图）
    true_groups = L.find_duplicate_groups(val_stems, ys, threshold=0.01)
    true_multi = [g for g in true_groups if len(g) > 1]
    print(f"\n真值同文字组: {[[int(s) for s in g] for g in true_multi]}")

    # 用不同阈值测试分组效果
    for thresh in [0.01, 0.02, 0.03, 0.05]:
        final = TM.apply_test_median(val_stems, xs, preds, threshold=thresh, verbose=False)
        r = np.mean([C.rmse(final[s], ys[s]) for s in val_stems])
        groups = TM.find_text_groups(val_stems, xs, preds, pred_thresh=thresh)
        multi = [g for g in groups if len(g) > 1]
        print(f"\nthresh={thresh}: RMSE={r:.5f} (base {base:.5f}, Δ={base-r:+.5f}), 组数={len(multi)}")
        for g in multi:
            print(f"  {[int(s) for s in g]}")


if __name__ == "__main__":
    main()
