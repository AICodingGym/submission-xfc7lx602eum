"""生成最终提交：跨集文字泄漏方案。

测试图 -> 训练集同文字图 -> 用训练干净图直接预测。

用法：
  python scripts/make_submission_cross.py --out outputs/submission_cross.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import common as C
from src import cross_leakage as CL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksize", type=int, default=25)
    ap.add_argument("--out", type=str, default="outputs/submission_cross.csv")
    args = ap.parse_args()

    print("[cross_submit] 执行跨集文字泄漏预测...")
    preds = CL.predict_test(ksize=args.ksize, verbose=True)

    # 写提交
    out = C.write_submission(preds, Path(args.out))
    n_pixels = sum(p.size for p in preds.values())
    print(f"\n[cross_submit] DONE -> {out} ({len(preds)} images, {n_pixels:,} pixels)")


if __name__ == "__main__":
    main()
