"""端到端流水线：多 seed 训练 → 集成预测 → TTA → 后处理 → 生成提交。

用法：
  python -m src.run_pipeline --seeds 0 1 2 --out outputs/submission.csv

步骤：
1. 对每个 seed 训练一个 U-Net（若权重已存在则跳过）。
2. 用所有模型对测试集做 TTA 集成预测。
3. 在验证集上扫描后处理 k。
4. 生成 melted 提交文件。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from . import common as C
from .model import DenoiseUNet
from .postprocess import tune_k, soft_threshold
from .predict import load_model, predict_dir
from .train import train


def val_predictions(models, val_stems, device):
    """用集成模型对验证集预测，返回 {stem: pred} 与 {stem: target}。"""
    from .predict import predict_image_ensemble

    preds, targets = {}, {}
    for s in val_stems:
        x = C.load_gray(C.TRAIN_DIR / f"{s}.png")
        y = C.load_gray(C.CLEAN_DIR / f"{s}.png")
        preds[s] = predict_image_ensemble(models, x, device, tta=True)
        targets[s] = y
    return preds, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="outputs/submission.csv")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 训练（或复用）
    ckpts = []
    for seed in args.seeds:
        ckpt = C.CKPT_DIR / f"model_seed{seed}.pt"
        ckpts.append(ckpt)
        if ckpt.exists() and args.skip_train:
            print(f"[pipeline] skip training seed {seed} (exists)")
            continue
        print(f"[pipeline] training seed {seed} ...")
        # 直接调用 train，参数透传
        from types import SimpleNamespace

        targs = SimpleNamespace(
            epochs=args.epochs, base=args.base, depth=args.depth,
            patch=args.patch, batch=args.batch, lr=args.lr, seed=seed,
            split_seed=42, val_frac=0.13, patience=20, workers=2,
            out=str(ckpt),
        )
        train(targs)

    # 2. 集成模型
    models = [load_model(c, device) for c in ckpts]
    print(f"[pipeline] loaded {len(models)} models")

    # 3. 验证集评估 + k 调参
    train_stems, val_stems = C.make_split(seed=42, val_fraction=0.13)
    vp, vt = val_predictions(models, val_stems, device)
    base_rmse = float(np.mean([C.rmse(vp[s], vt[s]) for s in val_stems]))
    print(f"[pipeline] ensemble val RMSE (k=1): {base_rmse:.5f}")

    best_k, best_rmse = tune_k(vp, vt)
    print(f"[pipeline] tuned k={best_k:.2f} -> val RMSE {best_rmse:.5f}")
    k = best_k if best_rmse < base_rmse else 1.0
    print(f"[pipeline] final k={k}")

    # 4. 测试集预测
    preds = predict_dir(models, C.TEST_DIR, C.PRED_DIR, device, tta=True, save_png=True)

    # 5. 后处理
    if k != 1.0:
        preds = {s: soft_threshold(p, k) for s, p in preds.items()}

    # 6. 生成提交
    out = C.write_submission(preds, Path(args.out))
    print(f"[pipeline] submission written to {out} (val RMSE {best_rmse:.5f})")


if __name__ == "__main__":
    main()
