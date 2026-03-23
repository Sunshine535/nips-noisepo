#!/bin/bash
set -e

CONDA_BASE=/data/szs/250010072/szs/anaconda3
ENV_NAME=nips_noisepo
PROJECT_DIR=/data/szs/250010072/nwh/nips-noisepo
DATA_DIR=/data/szs/250010072/nwh/data
SHARE_DIR=/data/szs/share

source ${CONDA_BASE}/bin/activate
conda activate ${ENV_NAME}

echo "=== Installing PyTorch (CUDA 12.4) ==="
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing project dependencies ==="
pip install -r ${PROJECT_DIR}/requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/

echo "=== Installing extra deps ==="
pip install deepspeed bitsandbytes sentence-transformers \
    -i https://mirrors.aliyun.com/pypi/simple/

echo "=== Creating data directory ==="
mkdir -p ${DATA_DIR}/models
mkdir -p ${DATA_DIR}/datasets
mkdir -p ${DATA_DIR}/checkpoints
mkdir -p ${DATA_DIR}/logs

echo "=== Linking shared models ==="
ln -sf ${SHARE_DIR}/Qwen3.5-9B ${DATA_DIR}/models/Qwen3.5-9B
ln -sf ${SHARE_DIR}/Qwen3.5-27B ${DATA_DIR}/models/Qwen3.5-27B
ln -sf ${SHARE_DIR}/all-MiniLM-L6-v2 ${DATA_DIR}/models/all-MiniLM-L6-v2

echo "=== Verify ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
import transformers; print(f'Transformers: {transformers.__version__}')
import trl; print(f'TRL: {trl.__version__}')
import peft; print(f'PEFT: {peft.__version__}')
import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')
"
echo "=== Setup Complete ==="
