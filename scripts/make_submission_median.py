"""生成「带文字泄漏 median 后处理」的最终提交。

流程：
1. V2 三模型加权集成 + 8-TTA 预测测试集
2. test_median 同文字组 median 合并（消除独立残差）
3. soft_threshold 后处理
4. 写提交

用法：
  python scripts/make_submission_median.py --ckpts outputs/ckpts/v2_seed0.pt ... \
      --weights 0.35 0.45 0.20 --threshold 0.03 --out outputs/submission_median.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import common as C
from src import test_median as TM
from src.postprocess import soft_threshold
from src.predict_v2 import load_model_v2, predict_image_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--n-tta", type=int, default=8, choices=[4, 8])
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="outputs/submission_median.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_paths = [Path(c) if Path(c).is_absolute() else ROOT / c for c in args.ckpts]
    models = [load_model_v2(c, device) for c in ckpt_paths]
    print(f"[median_submit] loaded {len(models)} models")

    if args.weights is not None:
        weights = np.asarray(args.weights, dtype=np.float32)
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(models), dtype=np.float32) / len(models)
    print(f"[median_submit] weights: {weights.tolist()}")

    # 测试集预测
    test_stems = C.list_image_stems(C.TEST_DIR)
    xs = {}
    preds = {}
    for i, s in enumerate(test_stems):
        x = C.load_gray(C.TEST_DIR / f"{s}.png")
        xs[s] = x
        ps = [predict_image_v2(m, x, device, tta=True, n_tta=args.n_tta) for m in models]
        p = sum(w * pi for w, pi in zip(weights, ps))
        preds[s] = np.clip(p, 0.0, 1.0)
        if (i + 1) % 5 == 0:
            print(f"[median_submit] predicted {i+1}/{len(test_stems)}")

    # median 后处理
    final = TM.apply_test_median(test_stems, xs, preds, threshold=args.threshold, verbose=True)

    # soft_threshold
    for s in test_stems:
        final[s] = soft_threshold(np.clip(final[s], 0, 1), args.k)

    out = C.write_submission(final, Path(args.out))
    n_pixels = sum(p.size for p in final.values())
    print(f"[median_submit] DONE -> {out} ({len(final)} images, {n_pixels:,} pixels)")


if __name__ == "__main__":
    main()
