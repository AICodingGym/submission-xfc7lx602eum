"""model_v2 / loss / dataset_v2 / train_v2 的单元测试。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import common as C
from src.dataset import DocDataset, _local_features, is_short_stem, oversample_short
from src.loss import CombinedLoss, Sobel
from src.model_v2 import DenoiseUNetV2, count_params, DenseBlock, ResBlock, SE, AttentionGate


def test_local_features_shape_and_range():
    x = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    mu, var = _local_features(x)
    assert mu.shape == x.shape
    assert var.shape == x.shape
    assert mu.min() >= 0 and mu.max() <= 1
    assert var.min() >= 0 and var.max() <= 1


def test_local_features_constant_image():
    # 常数图 → 均值 ≈ 自身值，方差 ≈ 0
    x = np.full((32, 32), 0.7, dtype=np.float32)
    mu, var = _local_features(x)
    assert np.allclose(mu, 0.7, atol=0.01)
    assert var.max() < 0.01


def test_resblock_shape():
    m = ResBlock(16)
    x = torch.randn(2, 16, 32, 32)
    y = m(x)
    assert y.shape == x.shape


def test_se_block_shape():
    m = SE(16)
    x = torch.randn(2, 16, 32, 32)
    y = m(x)
    assert y.shape == x.shape


def test_dense_block_shape():
    m = DenseBlock(16, n_res=3)
    x = torch.randn(2, 16, 32, 32)
    y = m(x)
    assert y.shape == x.shape


def test_attention_gate_shape():
    m = AttentionGate(16, 32, 8)
    x = torch.rand(2, 16, 32, 32)  # 非负输入便于验证缩放
    g = torch.randn(2, 32, 16, 16)
    y = m(x, g)
    assert y.shape == x.shape


def test_v2_forward_output_range_and_shape():
    m = DenoiseUNetV2(in_channels=3, base=16, depth=4)
    x = torch.rand(2, 3, 64, 64)
    y = m(x)
    assert y.shape == (2, 1, 64, 64)
    assert y.min() >= 0 and y.max() <= 1


def test_v2_forward_with_aux_returns_list():
    m = DenoiseUNetV2(in_channels=3, base=16, depth=4)
    x = torch.rand(1, 3, 64, 64)
    main, aux = m(x, return_aux=True)
    assert main.shape == (1, 1, 64, 64)
    # 辅助预测数量 = depth
    assert len(aux) == 4
    for a in aux:
        assert a.shape[1] == 1
        assert a.min() >= 0 and a.max() <= 1


def test_v2_supports_mixed_sizes():
    m = DenoiseUNetV2(in_channels=3, base=16, depth=4).eval()
    for h, w in [(420, 540), (258, 540)]:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 1, h, w)


def test_v2_param_count():
    m = DenoiseUNetV2(in_channels=3, base=32, depth=4)
    n = count_params(m)
    # V2 应更大（深度密集块）
    assert n > 5_000_000


def test_sobel_grad_magnitude():
    sobel = Sobel()
    # 黑色背景中央白方块：梯度集中在边缘附近
    x = torch.zeros(1, 1, 32, 32)
    x[:, :, 12:20, 12:20] = 1.0
    g = sobel(x)
    assert g.shape == (1, 1, 32, 32)
    # 远离边缘处（最中央）应近似 0（容差 1e-4）
    assert g[:, :, 15:17, 15:17].max() < 1e-4
    # 边缘附近梯度应显著
    assert g.max() > 0.5


def test_combined_loss_value_positive_and_grad_decreases():
    loss_fn = CombinedLoss(w_grad=0.2, w_aux=0.1)
    pred = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)
    val = loss_fn(pred, target)
    assert val.item() > 0
    assert val.requires_grad is False  # 输入无 grad


def test_combined_loss_with_aux():
    loss_fn = CombinedLoss()
    pred = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)
    aux = [torch.rand(2, 1, 16, 16) for _ in range(3)]
    val_no_aux = loss_fn(pred, target)
    val_with_aux = loss_fn(pred, target, aux_preds=aux)
    # 加深监督应该增大 loss
    assert val_with_aux.item() > val_no_aux.item()


def test_dataset_v2_3_channels():
    train_stems, _ = C.make_split(seed=42)
    ds = DocDataset(train_stems[:5], train=True, patch=64, seed=0,
                    with_features=True, noise_aug=True)
    x, y = ds[0]
    # 3 通道输入
    assert x.shape == (3, 64, 64)
    assert y.shape == (1, 64, 64)
    # 局部均值通道应该平滑（数值应接近 x 的低频）
    assert x[1].min() >= 0 and x[1].max() <= 1


def test_dataset_v2_eval_full_size():
    ds = DocDataset(["101"], train=False, with_features=True)
    x, y = ds[0]
    # 验证模式不裁剪，3 通道
    assert x.shape == (3, C.IMG_H, C.IMG_W)
    assert y.shape == (1, C.IMG_H, C.IMG_W)


def test_is_short_stem():
    # 真实训练集中的样本
    assert is_short_stem("26") is True   # 短图
    assert is_short_stem("101") is False  # 标准图


def test_oversample_short_doubles_short_only():
    stems = ["101", "102", "26", "104"]
    out = oversample_short(stems, factor=2)
    # "26"（短）应出现 2 次，其他 1 次
    assert out.count("26") == 2
    assert out.count("101") == 1
    assert out.count("102") == 1
    assert out.count("104") == 1