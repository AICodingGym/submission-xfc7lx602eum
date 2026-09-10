"""common.py 的单元测试。

覆盖：图像 IO 往返、RMSE 边界、验证集划分性质、melted 提交往返一致性。
"""
import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import common as C


def test_list_image_stems():
    stems = C.list_image_stems(C.TRAIN_DIR)
    assert len(stems) == 115
    # 已按数值排序
    nums = [int(s) for s in stems]
    assert nums == sorted(nums)


def test_load_save_roundtrip(tmp_path):
    arr = np.random.default_rng(0).random((16, 20)).astype(np.float32)
    p = tmp_path / "x.png"
    C.save_gray(p, arr)
    back = C.load_gray(p)
    # 量化误差应 < 1/255
    assert np.abs(back - arr).max() < 1.0 / 255.0 + 1e-6
    assert back.dtype == np.float32
    assert back.shape == (16, 20)


def test_load_gray_range():
    img = C.load_gray(C.TRAIN_DIR / "101.png")
    assert img.shape == (C.IMG_H, C.IMG_W)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_rmse():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.full((4, 4), 0.5, dtype=np.float32)
    assert abs(C.rmse(a, b) - 0.5) < 1e-6
    assert C.rmse(a, a) == 0.0


def test_make_split():
    train, val = C.make_split(seed=42, val_fraction=0.13)
    assert len(train) + len(val) == 115
    assert len(set(train) & set(val)) == 0
    # 可复现
    train2, val2 = C.make_split(seed=42, val_fraction=0.13)
    assert train == train2 and val == val2
    # 换 seed 应产生不同划分
    train3, _ = C.make_split(seed=1, val_fraction=0.13)
    assert train != train3


def test_melt_ids():
    ids = C.melt_ids("5", h=2, w=3)
    # 1-based 索引
    assert ids == ["5_1_1", "5_1_2", "5_1_3", "5_2_1", "5_2_2", "5_2_3"]


def test_write_submission_format(tmp_path):
    # 用两张小图验证格式与数值（1-based）
    preds = {
        "2": np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0,
        "1": np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0,
    }
    out = C.write_submission(preds, tmp_path / "sub.csv", h=2, w=3)
    lines = out.read_text().splitlines()
    assert lines[0] == "id,value"
    # 字典序排序：stem "1" 在前
    assert lines[1].startswith("1_1_1,")
    assert lines[1] == "1_1_1,0.000000"
    # 行数 = 1 header + 2*6
    assert len(lines) == 13


def test_write_submission_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    preds = {s: rng.random((C.IMG_H, C.IMG_W)).astype(np.float32) for s in ["101", "102", "103"]}
    out = C.write_submission(preds, tmp_path / "sub.csv")
    back = C.parse_submission(out)
    assert set(back.keys()) == {"101", "102", "103"}
    for s in preds:
        # 6 位小数精度往返
        assert np.allclose(back[s], preds[s], atol=1e-6)


def test_write_submission_clips_and_shape_guard(tmp_path):
    preds = {"9": np.full((2, 2), 1.5, dtype=np.float32)}
    out = C.write_submission(preds, tmp_path / "s.csv", h=2, w=2)
    back = C.parse_submission(out, h=2, w=2)
    assert back["9"].max() <= 1.0
    # 形状不符应报错
    bad = {"9": np.zeros((3, 2), dtype=np.float32)}
    with pytest.raises(ValueError):
        C.write_submission(bad, tmp_path / "bad.csv", h=2, w=2)


def test_write_submission_mixed_sizes(tmp_path):
    # 数据集存在两种尺寸：420x540 与 258x540，提交应按各自真实尺寸输出
    preds = {
        "101": np.zeros((420, 540), dtype=np.float32),
        "102": np.zeros((258, 540), dtype=np.float32),
    }
    out = C.write_submission(preds, tmp_path / "sub.csv")  # 不指定 h/w
    lines = out.read_text().splitlines()
    # 1 header + 420*540 + 258*540
    assert len(lines) == 1 + 420 * 540 + 258 * 540
    # 短图最后一行 id 应精确对应其末像素（1-based）
    assert lines[-1].split(",")[0] == "102_258_540"


def test_write_submission_matches_sample_format(tmp_path):
    # 校验与官方 sampleSubmission.csv 的格式一致：1-based 索引
    # 官方首个 id 应为 image_1_1（row=1, col=1 起）
    preds = {"110": np.zeros((420, 540), dtype=np.float32)}
    out = C.write_submission(preds, tmp_path / "sub.csv")
    first_data = out.read_text().splitlines()[1].split(",")[0]
    assert first_data == "110_1_1"
    last_data = out.read_text().splitlines()[-1].split(",")[0]
    assert last_data == "110_420_540"
