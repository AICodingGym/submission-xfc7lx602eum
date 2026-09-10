# 租卡训练方案集合

针对不同 GPU 配置和训练预算，提供几套现成脚本，去租卡时直接对应挑选运行。

## 环境要求

```bash
# 推荐 conda 环境
conda create -n denoise python=3.11 -y
conda activate denoise
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python-headless numpy pillow pytest
```

## 方案对比

| 方案 | base | depth | 参数量 | 适用显卡 | batch | patch | 200 epoch 大致耗时 |
|---|---|---|---|---|---|---|---|
| **A 超大模型** | 64 | 5 | 910M | A100 80GB / H100 | 16 | 256 | ~2-3 小时 |
| **B 标准大模型** | 64 | 4 | 227M | A100 40GB / RTX 4090 | 32 | 256 | ~30-40 分钟 |
| **C 大模型轻量** | 48 | 4 | 128M | RTX 3090 / A100 40GB | 32 | 256 | ~20 分钟 |
| **D 中等模型** | 32 | 4 | 57M | RTX 3080 / RTX 4060 | 16 | 192 | ~25 分钟 |
| **E 轻量（基线）** | 32 | 4 | 57M | RTX 3060 / RTX 4060 | 8 | 192 | ~20 分钟 |

## 快速启动

```bash
# A100 40GB 推荐方案 B
python scripts/cloud_B_standard.py --seed 0
python scripts/cloud_B_standard.py --seed 1
python scripts/cloud_B_standard.py --seed 2

# A100 80GB 预算充足选方案 A
python scripts/cloud_A_huge.py --seed 0
python scripts/cloud_A_huge.py --seed 1
```

## 训练完成后

1. 拷贝权重到本地 `outputs/ckpts/`：
   ```bash
   scp user@cloud:/path/to/cloud_v2_seed0.pt outputs/ckpts/
   ```
2. 改写 `make_submission.py` 加入 V2 模型，或用 `predict_v2.py` 推理：
   ```python
   from src.predict_v2 import load_model_v2, predict_image_v2
   from src.predict import predict_image_v2_ensemble
   ```
3. 混合 V1+V2 模型集成（V1 1-通道 + V2 3-通道），需要做通道对齐：
   ```python
   # V1 模型需要给 3 通道输入才能与 V2 集成
   x3 = np.stack([x, local_mean(x), local_var(x)])  # V1 也用 3 通道
   ```

## 集成策略（推荐）

训练完成后，用 **加权集成**（不是简单平均），验证集上搜索最优权重：

```python
# 在 make_submission_v2.py 中
weights = [w0, w1, w2]  # 验证集网格搜索得到
preds = [model.predict(x, tta=True) for model in models]
final = sum(w * p for w, p in zip(weights, preds))
```

## 预期性能

- 当前 V1 三模型集成：**0.01922**（银牌）
- V2 单模型（patch=192, 150 epoch，本地训练）：**预计 0.016-0.018**
- V2 三模型集成：**预计 0.015 左右**
- 大模型方案 B 训练：**预计 0.013-0.015**
- 多模型 + 方案 A + TTA 增强：**可能 0.012-0.014**（冲击金牌）

## 调试

训练时如遇 OOM：
1. 减小 `--batch`（如 32→16）
2. 减小 `--patch`（如 256→192）
3. 启用 `--no-noise-aug`

训练时如速度过慢：
1. 增加 `--workers`（多进程加载）
2. 启用 `--pin-memory`
3. 用更大 batch 喂满 GPU