"""方案 C：大模型轻量（base=48, depth=4，128M 参数）
适用：RTX 3090 / A100 40GB
比方案 B 快约 1.5x，精度相近
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train_v2 import train
from types import SimpleNamespace


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/ckpts/cloudC_v2_seed{seed}.pt")
    args = ap.parse_args()

    train_args = SimpleNamespace(
        epochs=200, base=48, depth=4, patch=256, batch=32, lr=1e-3,
        seed=args.seed, split_seed=42, val_frac=0.13,
        patience=30, workers=4,
        short_factor=2, noise_aug=True, w_grad=0.2, w_aux=0.1,
        out=args.out.format(seed=args.seed),
    )
    train(train_args)


if __name__ == "__main__":
    main()