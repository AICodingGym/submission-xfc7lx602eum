"""postprocess.py 的单元测试。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import common as C
from src.postprocess import soft_threshold, tune_k


def test_soft_threshold_identity():
    x = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    assert np.allclose(soft_threshold(x, k=1.0), x)


def test_soft_threshold_pushes_to_extremes():
    x = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    out = soft_threshold(x, k=2.0)
    # 0.4 -> 0.3, 0.5 -> 0.5, 0.6 -> 0.7
    assert np.allclose(out, [0.3, 0.5, 0.7])


def test_soft_threshold_range_and_monotonic():
    x = np.random.default_rng(1).random((100,)).astype(np.float32)
    out = soft_threshold(x, k=3.0)
    assert out.min() >= 0 and out.max() <= 1
    # 单调保序
    order = np.argsort(x)
    assert np.all(np.diff(out[order]) >= -1e-6)


def test_soft_threshold_rejects_k_below_1():
    with pytest.raises(ValueError):
        soft_threshold(np.zeros((4, 4), dtype=np.float32), k=0.5)


def test_tune_k_finds_better_k():
    # 构造：pred 偏离 0/1，target 是严格 0/1，期望 k>1 更优
    rng = np.random.default_rng(2)
    preds = {"a": np.abs(rng.normal(0.5, 0.2, (100,))).clip(0, 1).astype(np.float32)}
    targets = {"a": (preds["a"] > 0.5).astype(np.float32)}
    best_k, best_score = tune_k(preds, targets)
    assert best_k >= 1.0
    # 极值化后分数应不劣于 k=1
    base = C.rmse(soft_threshold(preds["a"], 1.0), targets["a"])
    assert best_score <= base + 1e-6
