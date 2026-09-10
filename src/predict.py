"""推理：加载模型权重，对测试集（或任意图像）做去噪预测。

支持 TTA（翻转/旋转）与多模型集成（M3 使用）。
用法：
  python -m src.predict --ckpt outputs/ckpts/model_seed0.pt \
      --out outputs/preds --test-dir data/test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from . import common as C
from .model import DenoiseUNet


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = DenoiseUNet(
        in_channels=1, base=args.get("base", 32), depth=args.get("depth", 4)
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_image(model, x, device, tta=True):
    """对单张 [0,1] 灰度图 (H,W) 预测干净图。tta=True 时做翻转增强取平均。

    支持任意尺寸（含非正方图）。翻转/旋转后做逆变换再平均。
    """
    model.eval()
    t = torch.from_numpy(x[None, None].astype(np.float32)).to(device)

    def _inv(v, kind):
        if kind == "h":
            return torch.flip(v, dims=[2])
        if kind == "v":
            return torch.flip(v, dims=[3])
        if kind == "r90":
            return torch.rot90(v, k=-1, dims=[2, 3])
        return v

    # 每个 (变换, 逆变换) 对
    transforms = [("id", None), ("h", "h"), ("v", "v"), ("r90", "r90")]
    outs = []
    for kind, _ in transforms:
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
def predict_image_ensemble(models, x, device, tta=True):
    """多模型集成：对每个模型做 TTA 预测后取平均。"""
    preds = [predict_image(m, x, device, tta=tta) for m in models]
    return np.mean(preds, axis=0)


@torch.no_grad()
def predict_image_weighted(models, x, device, weights, tta=True):
    """加权集成：weights 与 models 对齐。"""
    preds = [predict_image(m, x, device, tta=tta) for m in models]
    w = np.asarray(weights, dtype=np.float32)
    w = w / w.sum()
    out = sum(wi * p for wi, p in zip(w, preds))
    return out


def predict_dir(
    models, test_dir: Path, out_dir: Path, device, tta: bool = True, save_png: bool = True
) -> dict[str, np.ndarray]:
    """对目录下所有图像去噪，返回 {stem: array}，可选保存 PNG。"""
    stems = C.list_image_stems(test_dir)
    preds = {}
    for s in stems:
        x = C.load_gray(test_dir / f"{s}.png")
        pred = predict_image_ensemble(models, x, device, tta=tta)
        preds[s] = np.clip(pred, 0.0, 1.0)
        if save_png:
            C.save_gray(out_dir / f"{s}.png", preds[s])
    return preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="一个或多个模型权重")
    ap.add_argument("--out", type=str, default="outputs/preds")
    ap.add_argument("--test-dir", type=str, default="data/test")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(c, device) for c in args.ckpt]

    preds = predict_dir(
        models,
        Path(args.test_dir),
        Path(args.out),
        device,
        tta=not args.no_tta,
        save_png=not args.no_png,
    )
    print(f"[predict] predicted {len(preds)} images -> {args.out}")
