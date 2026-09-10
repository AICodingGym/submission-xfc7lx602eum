"""Denoising Dirty Documents — 公共基础设施模块。

提供：路径管理、图像 IO、验证集划分、RMSE 指标、melted 提交文件生成。
所有函数保持无状态、可测试。
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRAIN_DIR = DATA / "train"
CLEAN_DIR = DATA / "train_cleaned"
TEST_DIR = DATA / "test"
OUTPUTS = ROOT / "outputs"
CKPT_DIR = OUTPUTS / "ckpts"
PRED_DIR = OUTPUTS / "preds"

IMG_W, IMG_H = 540, 420


def list_image_stems(directory: Path) -> list[str]:
    """返回目录下所有 .png 文件的 stem（不含扩展名），按数值排序。"""
    stems = sorted(
        (p.stem for p in directory.glob("*.png")),
        key=lambda s: int(s),
    )
    return stems


def load_gray(path: Path) -> np.ndarray:
    """读取灰度图，返回 float32 数组，值域 [0, 1]，形状 (H, W)。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img.astype(np.float32) / 255.0


def save_gray(path: Path, arr: np.ndarray) -> None:
    """保存 float32 [0,1] 灰度数组为 PNG。"""
    arr = np.clip(arr, 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (arr * 255.0).astype(np.uint8))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """逐像素 RMSE（与 Kaggle 评测一致）。"""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


# ---------------------------------------------------------------------------
# 验证集划分
# ---------------------------------------------------------------------------
def make_split(
    stems: list[str] | None = None,
    val_fraction: float = 0.13,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """按 seed 将 stem 列表划分为 train / val。

    采用固定随机打乱，保证可复现。返回 (train_stems, val_stems)。
    """
    if stems is None:
        stems = list_image_stems(TRAIN_DIR)
    stems = sorted(stems, key=lambda s: int(s))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(stems))
    n_val = max(1, int(round(len(stems) * val_fraction)))
    val_idx = set(idx[:n_val].tolist())
    train = [s for i, s in enumerate(stems) if i not in val_idx]
    val = [s for i, s in enumerate(stems) if i in val_idx]
    return train, val


# ---------------------------------------------------------------------------
# 提交文件生成（melted 格式）
# ---------------------------------------------------------------------------
def melt_ids(stem: str, h: int, w: int) -> list[str]:
    """生成单张图的 melted id 序列：image_row_col，行优先，索引从 1 开始。"""
    return [f"{stem}_{r}_{c}" for r in range(1, h + 1) for c in range(1, w + 1)]


def write_submission(
    preds: dict[str, np.ndarray],
    out_path: Path,
    h: int | None = None,
    w: int | None = None,
    one_based: bool = True,
) -> Path:
    """将 {stem: (H,W) float32 [0,1] 数组} 写出为 melted 提交 CSV。

    id 顺序：image_row_col，行优先。one_based=True 时 row/col 从 1 开始
    （与官方 sampleSubmission.csv 一致）；否则从 0 开始。
    值写入时 clip 到 [0,1]。流式写出以控制内存。

    若 h/w 未指定，则按每张图自身的实际形状输出（支持混合尺寸）。
    stem 排序采用字典序（与官方 sample 一致）。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(preds.keys())
    off = 1 if one_based else 0
    with open(out_path, "w", newline="") as f:
        f.write("id,value\n")
        for stem in ordered:
            arr = np.asarray(preds[stem], dtype=np.float32)
            ph, pw = (h, w) if h is not None else arr.shape
            if arr.shape != (ph, pw):
                raise ValueError(
                    f"{stem}: 形状 {arr.shape} 与预期 ({ph},{pw}) 不符"
                )
            arr = np.clip(arr, 0.0, 1.0).ravel()
            lines = [
                f"{stem}_{r}_{c},{v:.6f}"
                for (r, c), v in zip(
                    ((r, c) for r in range(off, ph + off) for c in range(off, pw + off)),
                    arr,
                )
            ]
            f.write("\n".join(lines) + "\n")
    return out_path


def parse_submission(path: Path, h: int | None = None, w: int | None = None):
    """读取 melted 提交文件，返回 {stem: (H,W) float32 数组}。

    反向校验用：保证 write_submission 的格式可无损还原。
    支持 1-based 索引与混合尺寸（h/w 未指定时按 id 的最大行列推断）。
    """
    rows_by_stem: dict[str, list[tuple[int, int, float]]] = {}
    order: list[str] = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            idpart, val = line.split(",")
            stem, r, c = idpart.rsplit("_", 2)
            r, c = int(r), int(c)
            if stem not in rows_by_stem:
                rows_by_stem[stem] = []
                order.append(stem)
            rows_by_stem[stem].append((r, c, float(val)))

    stems: dict[str, np.ndarray] = {}
    for stem in order:
        entries = rows_by_stem[stem]
        max_r = max(e[0] for e in entries)
        max_c = max(e[1] for e in entries)
        # 检测 1-based：最小行列是否为 1
        min_r = min(e[0] for e in entries)
        min_c = min(e[1] for e in entries)
        one_based = (min_r == 1 and min_c == 1)
        ph = h if h is not None else (max_r if one_based else max_r + 1)
        pw = w if w is not None else (max_c if one_based else max_c + 1)
        arr = np.zeros((ph, pw), dtype=np.float32)
        for r, c, v in entries:
            rr = r - 1 if one_based else r
            cc = c - 1 if one_based else c
            arr[rr, cc] = v
        stems[stem] = arr
    return stems
