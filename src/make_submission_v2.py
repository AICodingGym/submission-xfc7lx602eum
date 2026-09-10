"""生成最终提交：V2 模型集成 + 多 TTA + 后处理。

支持：
  - 多模型集成（等权或加权）
  - 4 或 8 变换 TTA
  - 验证集上搜索最优权重
  - 后处理 soft_threshold
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import common as C
from src.postprocess import soft_threshold
from src.predict_v2 import load_model_v2, predict_image_v2


def search_weights(models, val_stems, device, n_tta=4):
    """在验证集上网格搜索最优集成权重。

    Args:
        models: V2 模型列表
        val_stems: 验证集 stem 列表
        device: torch.device
        n_tta: TTA 变换数

    Returns:
        (weights, val_rmse)
    """
    # 收集每个模型每张图的预测
    n = len(models)
    preds_by_model = {i: {} for i in range(n)}
    targets = {}
    for s in val_stems:
        x = C.load_gray(C.TRAIN_DIR / f"{s}.png")
        y = C.load_gray(C.CLEAN_DIR / f"{s}.png")
        targets[s] = y
        for i, m in enumerate(models):
            preds_by_model[i][s] = predict_image_v2(m, x, device, tta=True, n_tta=n_tta)

    # 单模型 RMSE
    print("Single-model val RMSE:")
    for i in range(n):
        r = np.mean([C.rmse(preds_by_model[i][s], targets[s]) for s in val_stems])
        print(f"  model[{i}]: {r:.5f}")

    # 等权 RMSE
    rmse_eq = np.mean([
        C.rmse(np.mean([preds_by_model[i][s] for i in range(n)], axis=0), targets[s])
        for s in val_stems
    ])
    print(f"Equal-weight val RMSE: {rmse_eq:.5f}")

    # 网格搜索权重（n 个模型的网格）
    best = (1e9, np.ones(n) / n)
    if n == 2:
        for w0 in np.arange(0.0, 1.01, 0.05):
            w = np.array([w0, 1 - w0])
            r = np.mean([
                C.rmse(sum(wi * preds_by_model[i][s] for i, wi in enumerate(w)), targets[s])
                for s in val_stems
            ])
            if r < best[0]:
                best = (r, w)
    elif n == 3:
        for w0 in np.arange(0.0, 1.01, 0.05):
            for w1 in np.arange(0.0, 1.0 - w0 + 0.001, 0.05):
                w2 = 1 - w0 - w1
                if w2 < -1e-9:
                    continue
                w2 = max(0.0, w2)
                w = np.array([w0, w1, w2])
                r = np.mean([
                    C.rmse(sum(wi * preds_by_model[i][s] for i, wi in enumerate(w)), targets[s])
                    for s in val_stems
                ])
                if r < best[0]:
                    best = (r, w)
    else:
        # 一般情况：随机采样权重
        for _ in range(2000):
            w = np.random.dirichlet(np.ones(n))
            r = np.mean([
                C.rmse(sum(wi * preds_by_model[i][s] for i, wi in enumerate(w)), targets[s])
                for s in val_stems
            ])
            if r < best[0]:
                best = (r, w)

    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=None,
                    help="V2 模型权重列表；默认用 outputs/ckpts/v2_seed*.pt")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="集成权重；不提供则在验证集搜索")
    ap.add_argument("--out", type=str, default="outputs/submission.csv")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--n-tta", type=int, default=4, choices=[4, 8])
    ap.add_argument("--no-search", action="store_true",
                    help="不搜索权重，等权集成")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpts is None:
        ckpt_paths = sorted(C.CKPT_DIR.glob("v2_seed*.pt"))
    else:
        ckpt_paths = [C.CKPT_DIR / c if not Path(c).is_absolute() else Path(c) for c in args.ckpts]

    models = [load_model_v2(c, device) for c in ckpt_paths]
    n = len(models)
    print(f"[submit_v2] loaded {n} models: {[c.name for c in ckpt_paths]}")

    # 权重
    if args.weights is not None:
        weights = np.asarray(args.weights, dtype=np.float32)
        weights = weights / weights.sum()
        print(f"[submit_v2] weights (手动): {weights.tolist()}")
    elif args.no_search or n == 1:
        weights = np.ones(n, dtype=np.float32) / n
        print(f"[submit_v2] weights (等权): {weights.tolist()}")
    else:
        print("[submit_v2] 验证集上搜索最优权重...")
        train_stems, val_stems = C.make_split(seed=42, val_fraction=0.13)
        weights, val_rmse = search_weights(models, val_stems, device, n_tta=args.n_tta)
        print(f"[submit_v2] best weights: {weights.tolist()}, val RMSE: {val_rmse:.5f}")

    # 测试集推理
    test_stems = C.list_image_stems(C.TEST_DIR)
    preds = {}
    for i, s in enumerate(test_stems):
        x = C.load_gray(C.TEST_DIR / f"{s}.png")
        ps = [predict_image_v2(m, x, device, tta=True, n_tta=args.n_tta) for m in models]
        p = sum(w * pi for w, pi in zip(weights, ps))
        p = soft_threshold(p, args.k)
        preds[s] = np.clip(p, 0.0, 1.0)
        if (i + 1) % 5 == 0 or i == len(test_stems) - 1:
            print(f"[submit_v2] predicted {i + 1}/{len(test_stems)}")

    out = C.write_submission(preds, C.OUTPUTS / "submission.csv")
    n_pixels = sum(p.size for p in preds.values())
    print(f"[submit_v2] DONE -> {out} ({len(preds)} images, {n_pixels:,} pixels)")


if __name__ == "__main__":
    main()