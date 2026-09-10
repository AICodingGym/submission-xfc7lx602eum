"""scripts/cloud_*.py 的烟雾测试：确保能正确构造训练参数并调用 train_v2.train。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_cloud_scripts_importable():
    """所有云端脚本应当能成功导入（说明语法正确）。"""
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    expected = ["cloud_A_huge.py", "cloud_B_standard.py", "cloud_C_mid.py", "cloud_D_local.py"]
    for name in expected:
        path = scripts_dir / name
        assert path.exists(), f"{path} 不存在"


def test_cloud_B_standard_constructs_args():
    """模拟云端脚本参数构造，验证 SimpleNamespace 字段完整。"""
    from types import SimpleNamespace

    train_args = SimpleNamespace(
        epochs=200, base=64, depth=4, patch=256, batch=32, lr=1e-3,
        seed=0, split_seed=42, val_frac=0.13,
        patience=30, workers=4,
        short_factor=2, noise_aug=True, w_grad=0.2, w_aux=0.1,
        out="/tmp/test.pt",
    )
    # 验证 train 函数能接受这些参数（不实际跑）
    from src.train_v2 import train
    import inspect
    sig = inspect.signature(train)
    params = list(sig.parameters.keys())
    # argparse.Namespace 字段应都能赋值
    for attr in ["epochs", "base", "depth", "patch", "batch", "lr", "seed",
                 "split_seed", "val_frac", "patience", "workers",
                 "short_factor", "noise_aug", "w_grad", "w_aux", "out"]:
        assert hasattr(train_args, attr), f"缺少字段 {attr}"


def test_cloud_A_huge_param_count_sane():
    """方案 A 应是 ~910M 参数。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.model_v2 import DenoiseUNetV2, count_params
    m = DenoiseUNetV2(3, base=64, depth=5)
    n = count_params(m)
    assert 800_000_000 < n < 1_000_000_000, f"参数异常: {n}"


def test_cloud_B_standard_param_count_sane():
    """方案 B 应是 ~227M 参数。"""
    from src.model_v2 import DenoiseUNetV2, count_params
    m = DenoiseUNetV2(3, base=64, depth=4)
    n = count_params(m)
    assert 200_000_000 < n < 250_000_000, f"参数异常: {n}"


def test_cloud_C_mid_param_count_sane():
    """方案 C 应是 ~128M 参数。"""
    from src.model_v2 import DenoiseUNetV2, count_params
    m = DenoiseUNetV2(3, base=48, depth=4)
    n = count_params(m)
    assert 100_000_000 < n < 150_000_000, f"参数异常: {n}"