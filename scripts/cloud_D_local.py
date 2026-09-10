"""方案 D：中等模型（base=32, depth=4，57M 参数）
适用：RTX 3080 / RTX 4060 8GB
与本地训练方案一致（patch=192 节省显存），便于本地复现
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
    ap.add_argument("--out", type=str, default="outputs/ckpts/cloudD_v2_seed{seed}.pt")
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    train_args = SimpleNamespace(
        epochs=200, base=32, depth=4, patch=args.patch, batch=args.batch, lr=1e-3,
        seed=args.seed, split_seed=42, val_frac=0.13,
        patience=30, workers=4,
        short_factor=2, noise_aug=True, w_grad=0.2, w_aux=0.1,
        out=args.out.format(seed=args.seed),
    )
    train(train_args)


if __name__ == "__main__":
    main()