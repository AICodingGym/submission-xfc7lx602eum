"""租卡训练入口：大模型 V2 配置，针对云端 GPU（如 A100 40GB/80GB）优化。

本地 RTX 4060 8GB 跑不动 base=64 depth=4（227M），但 A100 40GB+ 可轻松。
预计：base=64 depth=4, batch=32, patch=256, ~30 分钟完成 200 epoch。

用法（在租卡环境中）:
  python scripts/train_on_cloud.py --seed 0 \
      --base 64 --depth 4 --batch 32 --patch 256 --epochs 200 \
      --out outputs/ckpts/cloud_v2_seed0.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 让脚本能 import src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train_v2 import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=64, help="base channels (推荐 64)")
    ap.add_argument("--depth", type=int, default=4, help="unet depth (推荐 4)")
    ap.add_argument("--batch", type=int, default=32, help="batch size (按显存调)")
    ap.add_argument("--patch", type=int, default=256, help="crop patch")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--short-factor", type=int, default=2)
    ap.add_argument("--noise-aug", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    train(args)