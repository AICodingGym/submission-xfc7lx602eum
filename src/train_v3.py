"""V3 模型训练脚本：与 train_v2 相似，但用 DenoiseUNetV3。

用法：
  python -m src.train_v3 --epochs 150 --base 32 --depth 4 \
      --patch 192 --batch 8 --lr 1e-3 --seed 0 --workers 0 \
      --short-factor 2 --out outputs/ckpts/v3_seed0.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import common as C
from .dataset import DocDataset, oversample_short
from .loss import CombinedLoss
from .model_v3 import DenoiseUNetV2 as DenoiseUNetV3  # 占位，改为 V3


def evaluate(model, val_loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += ((pred - y) ** 2).sum().item()
            n += y.numel()
    return float(np.sqrt(total / n))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_stems, val_stems = C.make_split(seed=args.split_seed, val_fraction=args.val_frac)

    if args.short_factor > 1:
        train_stems = oversample_short(train_stems, factor=args.short_factor)

    train_ds = DocDataset(
        train_stems, train=True, patch=args.patch, seed=args.seed,
        with_features=True, noise_aug=args.noise_aug,
    )
    val_ds = DocDataset(val_stems, train=False, with_features=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = DenoiseUNetV3(in_channels=3, base=args.base, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_v3] params: {n_params:,}  device={device}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    loss_fn = CombinedLoss(w_grad=args.w_grad, w_aux=args.w_aux)

    best_rmse = float("inf")
    best_state = None
    patience = args.patience
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred, aux_preds = model(x, return_aux=True)
            loss = loss_fn(pred, y, aux_preds=aux_preds)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            nb += x.size(0)

        sched.step()
        avg_loss = running / max(nb, 1)
        val_rmse = evaluate(model, val_loader, device)
        dt = time.time() - t0
        cur_lr = opt.param_groups[0]["]["lr"]
        print(
            f"[epoch {epoch:3d}] loss={avg_loss:.5f} val_rmse={val_rmse:.5f} "
            f"lr={cur_lr:.2e} ({dt:.1f}s)"
        )

        if val_rmse < best_rmse - 1e-5:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[train_v3] early stop at epoch {epoch}")
                break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "val_rmse": best_rmse, "args": vars(args)}, out_path)
    print(f"[train_v3] best val_rmse={best_rmse:.5f} saved to {out_path}")
    return best_rmse


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.13)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/ckpts/v3_seed0.pt")
    ap.add_argument("--short-factor", type=int, default=2)
    ap.add_argument("--noise-aug", action="store_true")
    ap.add_argument("--w-grad", type=float, default=0.2)
    ap.add_argument("--w-aux", type=float, default=0.1)
    train(ap.parse_args())