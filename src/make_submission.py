"""生成最终提交：三模型加权集成 + TTA + 后处理。

用法：
  python -m src.make_submission
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import common as C
from src.postprocess import soft_threshold
from src.predict import load_model, predict_image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = [0.30, 0.05, 0.65]  # 验证集搜索得到的最优权重
    k = 1.0  # 后处理系数（验证集上 k=1 最优，即不做极值化）

    models = [load_model(C.CKPT_DIR / f"model_seed{i}.pt", device) for i in range(3)]
    print(f"[submit] loaded {len(models)} models, weights={weights}, k={k}")

    test_stems = C.list_image_stems(C.TEST_DIR)
    preds = {}
    for s in test_stems:
        x = C.load_gray(C.TEST_DIR / f"{s}.png")
        # 加权集成 + TTA
        ps = [predict_image(m, x, device, tta=True) for m in models]
        w = np.asarray(weights, dtype=np.float32)
        w = w / w.sum()
        p = sum(wi * pi for wi, pi in zip(w, ps))
        p = soft_threshold(p, k)
        preds[s] = np.clip(p, 0.0, 1.0)
        # 保存预测图
        C.save_gray(C.PRED_DIR / f"{s}.png", preds[s])
        if len(preds) % 8 == 0:
            print(f"[submit] predicted {len(preds)}/{len(test_stems)}")

    out = C.write_submission(preds, C.OUTPUTS / "submission.csv")
    n_pixels = sum(p.size for p in preds.values())
    print(f"[submit] DONE -> {out} ({len(preds)} images, {n_pixels:,} pixels)")


if __name__ == "__main__":
    main()
