#!/bin/bash
set -e

CONDA_BASE=/data/szs/250010072/szs/anaconda3
ENV_NAME=llama_factory
PROJECT_DIR=/data/szs/250010072/nwh/nips-noisepo
DATA_DIR=/data/szs/share/noisepo
SHARE_DIR=/data/szs/share

source ${CONDA_BASE}/bin/activate
conda activate ${ENV_NAME}

echo "=== [1/4] Installing PyTorch via conda (CUDA 12.4) ==="
conda install -y pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.4 \
    -c pytorch -c nvidia 2>&1 || {
    echo "conda install failed, falling back to pip..."
    pip install torch==2.5.1 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
}

echo "=== [2/4] Installing project dependencies ==="
pip install transformers datasets accelerate numpy scipy matplotlib tqdm \
    pandas pyyaml huggingface_hub trl peft evaluate wandb \
    deepspeed bitsandbytes sentence-transformers rich hydra-core \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

echo "=== [3/4] Creating data directory ==="
mkdir -p ${DATA_DIR}/models
mkdir -p ${DATA_DIR}/datasets
mkdir -p ${DATA_DIR}/checkpoints
mkdir -p ${DATA_DIR}/logs
mkdir -p ${DATA_DIR}/hf_cache
mkdir -p ${DATA_DIR}/results

echo "=== Linking shared models ==="
ln -sf ${SHARE_DIR}/Qwen3.5-9B ${DATA_DIR}/models/Qwen3.5-9B
ln -sf ${SHARE_DIR}/Qwen3.5-27B ${DATA_DIR}/models/Qwen3.5-27B
ln -sf ${SHARE_DIR}/all-MiniLM-L6-v2 ${DATA_DIR}/models/all-MiniLM-L6-v2

echo "=== [4/4] Verify ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
import transformers; print(f'Transformers: {transformers.__version__}')
import trl; print(f'TRL: {trl.__version__}')
import peft; print(f'PEFT: {peft.__version__}')
try:
    import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')
except: print('DeepSpeed: not installed (install on GPU node)')
"
echo "=== Setup Complete ==="
echo "Data dir:   ${DATA_DIR}"
echo "Models:     $(ls -1 ${DATA_DIR}/models/)"
echo ""
echo "Next steps:"
echo "  1. Quick test:  bash submit_quick_test.sh"
echo "  2. Full sweep:  bash submit_train.sh"
