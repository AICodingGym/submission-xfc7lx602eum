"""V2 模型推理：3 通道输入（gray + 局部均值 + 局部方差）。

用法：
  python -m src.predict_v2 --ckpt outputs/ckpts/v2_model_seed0.pt \
      --out outputs/preds_v2 --test-dir data/test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from . import common as C
from .dataset import _local_features
from .model_v2 import DenoiseUNetV2


def load_model_v2(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = DenoiseUNetV2(
        in_channels=3, base=args.get("base", 32), depth=args.get("depth", 4)
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_image_v2(model, x, device, tta=True):
    """对单张图预测，TTA = 翻转/旋转增强平均。

    x: (H,W) float32 [0,1] 灰度图。
    """
    model.eval()
    mu, var = _local_features(x)
    t_np = np.stack([x, mu, var], axis=0).astype(np.float32)
    t = torch.from_numpy(t_np[None]).to(device)  # (1, 3, H, W)

    transforms = ["id", "h", "v", "r90"]
    outs = []
    for kind in transforms:
        if kind == "id":
            v = t
        elif kind == "h":
            v = torch.flip(t, dims=[2])
        elif kind == "v":
            v = torch.flip(t, dims=[3])
        else:
            v = torch.rot90(t, k=1, dims=[2, 3])
        p = model(v)
        if kind == "h":
            p = torch.flip(p, dims=[2])
        elif kind == "v":
            p = torch.flip(p, dims=[3])
        elif kind == "r90":
            p = torch.rot90(p, k=-1, dims=[2, 3])
        outs.append(p)
    pred = torch.stack(outs).mean(dim=0)
    return pred[0, 0].cpu().numpy()


@torch.no_grad()
def predict_image_v2_ensemble(models, x, device, tta=True):
    preds = [predict_image_v2(m, x, device, tta=tta) for m in models]
    return np.mean(preds, axis=0)


def predict_dir_v2(models, test_dir: Path, out_dir: Path, device, save_png=True):
    stems = C.list_image_stems(test_dir)
    preds = {}
    for s in stems:
        x = C.load_gray(test_dir / f"{s}.png")
        p = predict_image_v2_ensemble(models, x, device, tta=True)
        preds[s] = np.clip(p, 0.0, 1.0)
        if save_png:
            C.save_gray(out_dir / f"{s}.png", preds[s])
    return preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", type=str, default="outputs/preds_v2")
    ap.add_argument("--test-dir", type=str, default="data/test")
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model_v2(c, device) for c in args.ckpt]
    preds = predict_dir_v2(models, Path(args.test_dir), Path(args.out), device)
    print(f"[predict_v2] predicted {len(preds)} images")