"""V2 模型推理：3 通道输入（gray + 局部均值 + 局部方差）。

支持 8 变换 TTA 与多模型加权集成。

用法：
  python -m src.predict_v2 --ckpt outputs/ckpts/v2_seed0.pt \
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


# 8 个 D4 群变换：恒等、3 个 90° 旋转、4 个翻转（水平、垂直、对角、逆对角）
_TTA_TRANSFORMS = [
    ("id",    lambda t: t,                          lambda p: p),
    ("r90",   lambda t: torch.rot90(t, 1, [2, 3]),  lambda p: torch.rot90(p, -1, [2, 3])),
    ("r180",  lambda t: torch.rot90(t, 2, [2, 3]),  lambda p: torch.rot90(p, -2, [2, 3])),
    ("r270",  lambda t: torch.rot90(t, 3, [2, 3]),  lambda p: torch.rot90(p, -3, [2, 3])),
    ("flip_h",  lambda t: torch.flip(t, [2]),       lambda p: torch.flip(p, [2])),
    ("flip_v",  lambda t: torch.flip(t, [3]),       lambda p: torch.flip(p, [3])),
    ("flip_d",  lambda t: torch.flip(t, [2, 3]),    lambda p: torch.flip(p, [2, 3])),
    ("flip_ad", lambda t: torch.flip(t, [2, 3]).rot90(1, [2, 3]),  # 对角翻转
                  lambda p: torch.rot90(torch.flip(p, [2, 3]), -1, [2, 3])),
]


@torch.no_grad()
def predict_image_v2(model, x, device, tta: bool = True, n_tta: int = 4):
    """对单张图预测，支持 4 或 8 变换 TTA 取平均。

    Args:
        model: V2 模型
        x: (H,W) float32 [0,1]
        device: torch.device
        tta: 是否使用 TTA
        n_tta: TTA 变换数（4 或 8）
    Returns:
        (H,W) float32 [0,1]
    """
    model.eval()
    mu, var = _local_features(x)
    t_np = np.stack([x, mu, var], axis=0).astype(np.float32)
    t = torch.from_numpy(t_np[None]).to(device)

    transforms = _TTA_TRANSFORMS[:n_tta] if tta else _TTA_TRANSFORMS[:1]
    outs = []
    for _, fwd, inv in transforms:
        v = fwd(t)
        p = inv(model(v))
        outs.append(p)
    pred = torch.stack(outs).mean(dim=0)
    return pred[0, 0].cpu().numpy()


@torch.no_grad()
def predict_image_v2_ensemble(models, x, device, tta: bool = True, n_tta: int = 4):
    """多模型等权集成 + TTA。"""
    preds = [predict_image_v2(m, x, device, tta=tta, n_tta=n_tta) for m in models]
    return np.mean(preds, axis=0)


@torch.no_grad()
def predict_image_v2_weighted(models, x, device, weights, tta: bool = True, n_tta: int = 4):
    """多模型加权集成 + TTA。"""
    preds = [predict_image_v2(m, x, device, tta=tta, n_tta=n_tta) for m in models]
    w = np.asarray(weights, dtype=np.float32)
    w = w / w.sum()
    return sum(wi * p for wi, p in zip(w, preds))


def predict_dir_v2(models, test_dir: Path, out_dir: Path, device,
                   weights=None, tta: bool = True, n_tta: int = 4, save_png: bool = True):
    """对目录下所有图像去噪，可选加权集成。"""
    stems = C.list_image_stems(test_dir)
    preds = {}
    for s in stems:
        x = C.load_gray(test_dir / f"{s}.png")
        if weights is None:
            p = predict_image_v2_ensemble(models, x, device, tta=tta, n_tta=n_tta)
        else:
            p = predict_image_v2_weighted(models, x, device, weights, tta=tta, n_tta=n_tta)
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
    ap.add_argument("--n-tta", type=int, default=4, choices=[4, 8])
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model_v2(c, device) for c in args.ckpt]
    preds = predict_dir_v2(
        models, Path(args.test_dir), Path(args.out), device,
        weights=args.weights, tta=not args.no_tta, n_tta=args.n_tta,
    )
    print(f"[predict_v2] predicted {len(preds)} images")