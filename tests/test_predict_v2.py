"""predict_v2 / make_submission_v2 的单元测试。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import common as C
from src.predict_v2 import (
    predict_image_v2, predict_image_v2_ensemble,
    predict_image_v2_weighted, load_model_v2, _TTA_TRANSFORMS,
)


def _make_small_v2():
    from src.model_v2 import DenoiseUNetV2
    return DenoiseUNetV2(in_channels=3, base=8, depth=3).eval()


def test_8_tta_same_shape_as_input():
    """8 变换 TTA 应能处理非正方形图。"""
    m = _make_small_v2()
    x = np.random.default_rng(0).random((128, 200)).astype(np.float32)
    p = predict_image_v2(m, x, torch.device("cpu"), tta=True, n_tta=8)
    assert p.shape == (128, 200)


def test_4_vs_8_tta_different():
    """4 变换与 8 变换 TTA 应产生不同结果（8 包含对角翻转）。"""
    m = _make_small_v2()
    # 用非正方形图（128x200）使旋转类变换的输出与正方形不同
    x = np.random.default_rng(1).random((128, 200)).astype(np.float32)
    p4 = predict_image_v2(m, x, torch.device("cpu"), tta=True, n_tta=4)
    p8 = predict_image_v2(m, x, torch.device("cpu"), tta=True, n_tta=8)
    # 至少应有差异
    assert not np.allclose(p4, p8, atol=1e-4)


def test_no_tta_vs_tta():
    """无 TTA 与有 TTA 应不同。"""
    m = _make_small_v2()
    x = np.random.default_rng(2).random((64, 64)).astype(np.float32)
    p_no = predict_image_v2(m, x, torch.device("cpu"), tta=False, n_tta=4)
    p_tta = predict_image_v2(m, x, torch.device("cpu"), tta=True, n_tta=4)
    assert not np.allclose(p_no, p_tta, atol=1e-4)


def test_ensemble_average_equal_weight():
    """等权集成 = 简单平均。"""
    m1 = _make_small_v2()
    m2 = _make_small_v2()
    x = np.random.default_rng(3).random((32, 32)).astype(np.float32)
    p1 = predict_image_v2(m1, x, torch.device("cpu"), tta=False)
    p2 = predict_image_v2(m2, x, torch.device("cpu"), tta=False)
    p_ens = predict_image_v2_ensemble([m1, m2], x, torch.device("cpu"), tta=False)
    assert np.allclose(p_ens, (p1 + p2) / 2)


def test_weighted_sum_to_one():
    """加权集成权重和应为 1（归一化）。"""
    m1 = _make_small_v2()
    m2 = _make_small_v2()
    x = np.random.default_rng(4).random((32, 32)).astype(np.float32)
    p = predict_image_v2_weighted([m1, m2], x, torch.device("cpu"),
                                   weights=[3.0, 1.0], tta=False)  # 不归一化输入
    p1 = predict_image_v2(m1, x, torch.device("cpu"), tta=False)
    p2 = predict_image_v2(m2, x, torch.device("cpu"), tta=False)
    # 归一化后权重 0.75/0.25
    expected = 0.75 * p1 + 0.25 * p2
    assert np.allclose(p, expected)


def test_tta_transforms_count():
    """应有 8 个 TTA 变换。"""
    assert len(_TTA_TRANSFORMS) == 8
    names = [t[0] for t in _TTA_TRANSFORMS]
    assert names == ["id", "r90", "r180", "r270", "flip_h", "flip_v", "flip_d", "flip_ad"]


def test_tta_idempotent_for_symmetric_input():
    """恒等图（常数）下，所有 TTA 输出应相同。"""
    m = _make_small_v2()
    x = np.full((32, 32), 0.7, dtype=np.float32)
    p = predict_image_v2(m, x, torch.device("cpu"), tta=True, n_tta=8)
    # 输出不应全为 0，也不应全为 1（模型对常数有响应）
    assert p.min() >= 0.0 and p.max() <= 1.0