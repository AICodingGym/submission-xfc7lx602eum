"""baseline.py 的单元测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import baseline as B
from src import common as C


def test_estimate_background_smooth():
    # 均匀背景 -> 背景估计应接近原值
    x = np.full((40, 40), 0.7, dtype=np.float32)
    bg = B.estimate_background(x, ksize=15)
    assert np.allclose(bg, 0.7, atol=1e-2)


def test_estimate_background_removes_text():
    # 白底上一条黑线：背景估计应把黑线抹平为白
    x = np.full((40, 40), 1.0, dtype=np.float32)
    x[20, 5:35] = 0.0
    bg = B.estimate_background(x, ksize=15)
    assert bg[20, 20] > 0.9  # 文字处背景被抹平


def test_denoise_baseline_output_range():
    rng = np.random.default_rng(0)
    x = rng.random((32, 32)).astype(np.float32)
    out = B.denoise_baseline(x)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_denoise_baseline_improves_rmse_on_real_pair():
    # 对真实脏图，基线输出应比原始图更接近 clean（否则方法无效）
    x = C.load_gray(C.TRAIN_DIR / "101.png")
    y = C.load_gray(C.CLEAN_DIR / "101.png")
    rmse_raw = C.rmse(x, y)
    out = B.denoise_baseline(x)
    rmse_out = C.rmse(out, y)
    assert rmse_out < rmse_raw


def test_denoise_baseline_batch():
    imgs = {
        "a": np.random.default_rng(1).random((20, 20)).astype(np.float32),
        "b": np.random.default_rng(2).random((20, 20)).astype(np.float32),
    }
    out = B.denoise_baseline_batch(imgs)
    assert set(out.keys()) == {"a", "b"}
    assert all(v.shape == (20, 20) for v in out.values())


def test_odd_ksize_guard():
    x = np.zeros((10, 10), dtype=np.float32)
    bg = B.estimate_background(x, ksize=10)  # 偶数会自动 +1
    assert bg.shape == x.shape
