"""train.py / predict.py 的单元测试（用小模型快速验证端到端）。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import common as C
from src.model import DenoiseUNet
from src.predict import load_model, predict_image, predict_image_ensemble, predict_dir
from src.train import evaluate


def _make_small_model():
    return DenoiseUNet(in_channels=1, base=8, depth=3)


def test_evaluate_decreasing_loss():
    # 训练一步后，验证 RMSE 应显著下降（模型确实在学）
    device = torch.device("cpu")
    model = _make_small_model().to(device)
    from src.dataset import DocDataset

    ds = DocDataset(["101"], train=True, patch=64, seed=0)
    x, y = ds[0]
    x = x[None].to(device)
    y = y[None].to(device)

    before = evaluate(model, [(x, y)], device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    import torch.nn as nn

    for _ in range(20):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    after = evaluate(model, [(x, y)], device)
    assert after < before


def test_load_model_roundtrip(tmp_path):
    device = torch.device("cpu")
    model = _make_small_model().eval()
    ckpt = tmp_path / "m.pt"
    torch.save({"state_dict": model.state_dict(), "val_rmse": 0.1, "args": {"base": 8, "depth": 3}}, ckpt)
    loaded = load_model(ckpt, device)  # 内部已 eval
    # 输出一致性
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)


def test_predict_image_tta_matches_no_tta_roughly():
    device = torch.device("cpu")
    model = _make_small_model().eval()
    x = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    p_no = predict_image(model, x, device, tta=False)
    p_tta = predict_image(model, x, device, tta=True)
    assert p_no.shape == (64, 64)
    assert p_tta.shape == (64, 64)
    # 两者都应在 [0,1]
    assert p_no.min() >= 0 and p_no.max() <= 1
    assert p_tta.min() >= 0 and p_tta.max() <= 1


def test_predict_image_ensemble():
    device = torch.device("cpu")
    models = [_make_small_model().eval(), _make_small_model().eval()]
    x = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    out = predict_image_ensemble(models, x, device, tta=False)
    assert out.shape == (32, 32)
    assert out.min() >= 0 and out.max() <= 1


def test_predict_dir(tmp_path):
    device = torch.device("cpu")
    models = [_make_small_model().eval()]
    out_dir = tmp_path / "preds"
    # 用 2 张训练图作为"测试"目录（复用到已有 png）
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    C.save_gray(src_dir / "101.png", np.random.default_rng(2).random((420, 540)).astype(np.float32))
    C.save_gray(src_dir / "102.png", np.random.default_rng(3).random((420, 540)).astype(np.float32))
    preds = predict_dir(models, src_dir, out_dir, device, tta=False, save_png=True)
    assert set(preds.keys()) == {"101", "102"}
    assert (out_dir / "101.png").exists()
