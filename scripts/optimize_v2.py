"""端到端优化流水线：训练 seed1/seed2 → 集成 → 提交。

用法：
  python scripts/optimize_v2.py --start-from-seed 0 --train-seeds 1 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train_v2 import train
from src.make_submission_v2 import main as run_make_submission


def run_training(seed, epochs, patch, batch, base, depth, short_factor, noise_aug):
    """训练单个 V2 seed。"""
    print(f"\n=== Training V2 seed {seed} ===")
    args = SimpleNamespace(
        epochs=epochs, base=base, depth=depth, patch=patch, batch=batch,
        lr=1e-3, seed=seed, split_seed=42, val_frac=0.13,
        patience=30, workers=0,
        short_factor=short_factor, noise_aug=noise_aug, w_grad=0.2, w_aux=0.1,
        out=f"outputs/ckpts/v2_seed{seed}.pt",
    )
    return train(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-from-seed", type=int, default=0,
                    help="已训练的最大 seed（不会重复训练）")
    ap.add_argument("--train-seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--short-factor", type=int, default=2)
    ap.add_argument("--noise-aug", action="store_true")
    ap.add_argument("--n-tta", type=int, default=4, choices=[4, 8])
    ap.add_argument("--submit", action="store_true",
                    help="训练完直接调用 make_submission_v2")
    args = ap.parse_args()

    for s in args.train_seeds:
        if s <= args.start_from_seed:
            print(f"[optimize_v2] skip seed {s} (already trained)")
            continue
        best = run_training(
            seed=s, epochs=args.epochs, patch=args.patch, batch=args.batch,
            base=args.base, depth=args.depth,
            short_factor=args.short_factor, noise_aug=args.noise_aug,
        )
        print(f"[optimize_v2] seed {s} best val_rmse: {best:.5f}")

    if args.submit:
        print("\n=== Generating submission ===")
        # 调用 make_submission_v2 子进程
        cmd = [
            sys.executable, "-u", "-m", "src.make_submission_v2",
            "--n-tta", str(args.n_tta),
            "--out", "outputs/submission.csv",
        ]
        subprocess.run(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()