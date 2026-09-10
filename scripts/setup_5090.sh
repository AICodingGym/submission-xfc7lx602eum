#!/usr/bin/env bash
# AutoDL RTX 5090 一键启动脚本
# 用法：上传 denoise_cloud.tar.gz 到 /root 后，执行
#   bash setup_5090.sh
set -e

echo "==================== 1. 校验环境 ===================="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('cap', torch.cuda.get_device_capability(0))"

echo "==================== 2. 解压代码包 ===================="
cd /root
if [ ! -f denoise_cloud.tar.gz ]; then
    echo "!! 未找到 denoise_cloud.tar.gz，请先上传到 /root 目录"
    exit 1
fi
tar -xzf denoise_cloud.tar.gz
cd /root/denoising-dirty-documents
echo "解压完成，项目目录结构："
ls src/ scripts/ data/

echo "==================== 3. 安装依赖 ===================="
pip install opencv-python-headless pillow numpy -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -2

echo "==================== 4. 冒烟测试（跑 1 epoch 验证链路） ===================="
python -c "from src.model_v2 import DenoiseUNetV2, count_params; m=DenoiseUNetV2(3,64,4); print('B模型参数量:', count_params(m))"
python -c "from src.model_v2 import DenoiseUNetV2, count_params; m=DenoiseUNetV2(3,64,5); print('A模型参数量:', count_params(m))"

echo "==================== 5. 启动训练 ===================="
echo "开始方案 B（标准大模型 227M）× 3 seed ..."
python scripts/cloud_B_standard.py --seed 0
python scripts/cloud_B_standard.py --seed 1
python scripts/cloud_B_standard.py --seed 2

echo "开始方案 A（超大模型 910M）× 2 seed ..."
python scripts/cloud_A_huge.py --seed 0
python scripts/cloud_A_huge.py --seed 1

echo "==================== 全部完成 ===================="
ls -lh outputs/ckpts/
echo "请用以下命令把权重拉回本地（本地 Windows 执行）："
echo "  scp -P <端口> root@<地址>:/root/denoising-dirty-documents/outputs/ckpts/cloud*.pt outputs/ckpts/"
