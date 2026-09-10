"""云端 cloudB (227M) 双模型集成 + 8-TTA 提交。

纯模型对照实验：验证不靠信息泄漏，纯 U-Net 集成能到什么分数。

策略：
  - 逐模型预测（避免同时驻留 2 个 227M 模型导致 OOM）
  - 每模型 8-TTA
  - 验证集搜索最优集成权重（cloudB_seed0 vs seed1）
  - 可选叠加 median 后处理（同文字组）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import common as C
from src.predict_v2 import load_model_v2, predict_image_v2
from src.postprocess import soft_threshold


def predict_with_model(model, stems, src_dir, device, n_tta=8):
    """用单个模型对一组图预测，返回 {stem: pred}。逐图推理，及时释放。"""
    preds = {}
    for i, s in enumerate(stems):
        x = C.load_gray(src_dir / f"{s}.png")
        preds[s] = predict_image_v2(model, x, device, tta=True, n_tta=n_tta)
        if (i + 1) % 5 == 0:
            print(f"    {i + 1}/{len(stems)}", flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", type=str, default="outputs/submission_cloudB.csv")
    ap.add_argument("--n-tta", type=int, default=8, choices=[4, 8])
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--median", action="store_true", help="叠加同文字组 median 后处理")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cloudB] device={device}", flush=True)

    ckpt_paths = [Path(c) for c in args.ckpts]
    n = len(ckpt_paths)

    # 验证集切分
    train_stems, val_stems = C.make_split(seed=42, val_fraction=0.13)
    test_stems = C.list_image_stems(C.TEST_DIR)

    # 逐模型预测验证集（用于搜权重）
    val_preds_by_model = []
    for c in ckpt_paths:
        print(f"[cloudB] 加载 {c.name} ...", flush=True)
        m = load_model_v2(c, device)
        print(f"[cloudB] 预测验证集 {c.name} ...", flush=True)
        val_preds_by_model.append(predict_with_model(m, val_stems, C.TRAIN_DIR, device, args.n_tta))
        del m
        torch.cuda.empty_cache()

    # 单模型 val RMSE
    val_targets = {s: C.load_gray(C.CLEAN_DIR / f"{s}.png") for s in val_stems}
    print("[cloudB] 单模型验证集 RMSE:", flush=True)
    for i, c in enumerate(ckpt_paths):
        r = np.mean([C.rmse(val_preds_by_model[i][s], val_targets[s]) for s in val_stems])
        print(f"  {c.name}: {r:.5f}", flush=True)

    # 权重
    if args.weights is not None:
        w = np.asarray(args.weights, dtype=np.float32)
        w = w / w.sum()
        print(f"[cloudB] 手动权重: {w.tolist()}", flush=True)
    else:
        # 网格搜索
        best = (1e9, None)
        if n == 2:
            for w0 in np.arange(0.0, 1.01, 0.05):
                ww = np.array([w0, 1 - w0], dtype=np.float32)
                r = np.mean([
                    C.rmse(sum(wi * val_preds_by_model[i][s] for i, wi in enumerate(ww)), val_targets[s])
                    for s in val_stems
                ])
                if r < best[0]:
                    best = (r, ww)
        else:
            for _ in range(2000):
                ww = np.random.dirichlet(np.ones(n)).astype(np.float32)
                r = np.mean([
                    C.rmse(sum(wi * val_preds_by_model[i][s] for i, wi in enumerate(ww)), val_targets[s])
                    for s in val_stems
                ])
                if r < best[0]:
                    best = (r, ww)
        w = best[1]
        print(f"[cloudB] 搜索最优权重: {w.tolist()}, val RMSE: {best[0]:.5f}", flush=True)

    # 测试集逐模型预测
    print(f"[cloudB] 测试集预测（{len(test_stems)} 张）...", flush=True)
    test_preds_by_model = []
    for c in ckpt_paths:
        m = load_model_v2(c, device)
        test_preds_by_model.append(predict_with_model(m, test_stems, C.TEST_DIR, device, args.n_tta))
        del m
        torch.cuda.empty_cache()

    # 加权集成 + 后处理
    preds = {}
    for s in test_stems:
        p = sum(wi * test_preds_by_model[i][s] for i, wi in enumerate(w))
        p = soft_threshold(p, args.k)
        preds[s] = np.clip(p, 0.0, 1.0)

    # median 后处理（可选）
    if args.median:
        from src import test_median as TM
        # 用去背景图分组
        xs_test = {s: C.load_gray(C.TEST_DIR / f"{s}.png") for s in test_stems}
        preds = TM.apply_test_median(test_stems, xs_test, preds, threshold=0.03, verbose=True)

    out = C.write_submission(preds, Path(args.out))
    n_pixels = sum(p.size for p in preds.values())
    print(f"[cloudB] DONE -> {out} ({len(preds)} images, {n_pixels:,} pixels)", flush=True)


if __name__ == "__main__":
    main()
