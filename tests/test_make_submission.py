"""make_submission.py 的测试：验证最终提交文件的格式正确性。

为避免重复加载大模型，这里用小型模型 + 少量图像做端到端验证。
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import common as C


def test_submission_file_format():
    """验证已生成的最终提交文件（若存在）格式正确。"""
    sub = C.OUTPUTS / "submission.csv"
    if not sub.exists():
        import pytest

        pytest.skip("submission.csv 尚未生成")

    lines = sub.read_text().splitlines()
    assert lines[0] == "id,value"
    # 每个数据行格式：id=image_row_col, value in [0,1]
    id_re = re.compile(r"^\d+_\d+_\d+$")
    test_stems = set(C.list_image_stems(C.TEST_DIR))
    seen_pixels = 0
    cur_stem = None
    for ln in lines[1:]:
        idp, val = ln.split(",")
        stem, r, c = idp.rsplit("_", 2)
        assert stem in test_stems
        assert id_re.fullmatch(idp)
        v = float(val)
        assert 0.0 <= v <= 1.0
        seen_pixels += 1
    # 总像素数匹配
    expected = sum(C.load_gray(C.TEST_DIR / f"{s}.png").size for s in test_stems)
    assert seen_pixels == expected


def test_submission_has_all_test_images():
    sub = C.OUTPUTS / "submission.csv"
    if not sub.exists():
        import pytest

        pytest.skip("submission.csv 尚未生成")
    test_stems = set(C.list_image_stems(C.TEST_DIR))
    stems_in_sub = set()
    for ln in sub.read_text().splitlines()[1:]:
        stem = ln.split(",")[0].rsplit("_", 2)[0]
        stems_in_sub.add(stem)
    assert stems_in_sub == test_stems
