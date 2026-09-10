"""生成最终提交：V2 模型加权集成 + TTA。

用法：
  python -m src.make_submission_v2
  python -m src.make_submission_v2 --ckpts v2_seed0.pt v2_seed1.pt v2_seed2.pt
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=None,
                    help="V2 模型权重列表；默认用 outputs/ckpts/v2_seed*.pt")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="集成权重；默认等权")
    ap.add_argument("--out", type=str, default="outputs/submission.csv")
    ap.add_argument("--k", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpts is None:
        ckpt_paths = sorted(C.CKPT_DIR.glob("v2_seed*.pt"))
    else:
        ckpt_paths = [C.CKPT_DIR / c if not Path(c).is_absolute() else Path(c) for c in args.ckpts]

    models = [load_model_v2(c, device) for c in ckpt_paths]
    n = len(models)

    if args.weights is None:
        weights = np.ones(n, dtype=np.float32) / n
    else:
        weights = np.asarray(args.weights, dtype=np.float32)
        weights = weights / weights.sum()

    print(f"[submit_v2] loaded {n} models: {[c.name for c in ckpt_paths]}")
    print(f"[submit_v2] weights: {weights.tolist()}  k={args.k}")

    test_stems = C.list_image_stems(C.TEST_DIR)
    preds = {}
    for i, s in enumerate(test_stems):
        x = C.load_gray(C.TEST_DIR / f"{s}.png")
        # 加权集成 + TTA
        ps = [predict_image_v2(m, x, device, tta=True) for m in models]
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