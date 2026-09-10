"""验证最关键的假设：测试集和训练集是否共享「相同文字」的图。

如果是，那么测试集的干净图可以直接从训练集的「同文字干净图」得到，
这是比「测试集内部 median」大得多的泄漏。

方法：
1. 对训练集和测试集都做「去背景」处理
2. 跨集比较去背景图的相似度
3. 找出测试集图对应的「同文字训练图」
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src import common as C
from src import leakage as L


def est_bg(x, k=25):
    return cv2.medianBlur((np.clip(x,0,1)*255).astype(np.uint8), k).astype(np.float32)/255.0


def align(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if a.shape != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    if b.shape != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def main():
    # 训练集
    train_stems = sorted(C.list_image_stems(C.TRAIN_DIR), key=int)
    # 测试集
    test_stems = sorted(C.list_image_stems(C.TEST_DIR), key=int)

    print(f"训练集 {len(train_stems)} 张, 测试集 {len(test_stems)} 张")

    # 训练集去背景图
    train_divs = {}
    train_clean = {}
    for s in train_stems:
        x = C.load_gray(C.TRAIN_DIR / f"{s}.png")
        train_divs[s] = np.clip(x/(est_bg(x)+1e-3), 0, 1)
        train_clean[s] = C.load_gray(C.CLEAN_DIR / f"{s}.png")

    # 测试集去背景图
    test_divs = {}
    for s in test_stems:
        x = C.load_gray(C.TEST_DIR / f"{s}.png")
        test_divs[s] = np.clip(x/(est_bg(x)+1e-3), 0, 1)

    # 跨集匹配：每个测试图找最相似的训练图（同尺寸）
    print("\n=== 跨集同文字匹配（测试图 -> 最相似训练图）===")
    for ts in test_stems:
        best = None
        best_r = 1e9
        for trs in train_stems:
            if train_divs[trs].shape != test_divs[ts].shape:
                continue
            r = C.rmse(train_divs[trs], test_divs[ts])
            if r < best_r:
                best_r = r
                best = trs
        flag = " <-- 可能同文字!" if best_r < 0.08 else ""
        print(f"  测试 {int(ts):>4} -> 训练 {int(best):>4}  (文字r={best_r:.4f}){flag}")


if __name__ == "__main__":
    main()
