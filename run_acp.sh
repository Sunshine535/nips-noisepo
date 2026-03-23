#!/bin/bash
# ============================================================================
# NaCPO ACP Training — Startup Script for SCO ACP GPU Container
#
# This script runs inside the ACP container (8x N6IS-80G).
# AFS cloud disk is mounted at /mnt/afs.
#
# Usage (as ACP startup command):
#   bash /mnt/afs/250010072/nwh/nips-noisepo/run_acp.sh
# ============================================================================
set -euo pipefail

AFS_ROOT=/mnt/afs
USER_DIR=${AFS_ROOT}/250010072
PROJECT_DIR=${USER_DIR}/nwh/nips-noisepo
DATA_DIR=${USER_DIR}/nwh/noisepo_data
SHARE_DIR=${AFS_ROOT}/share
CONDA_BASE=${USER_DIR}/szs/anaconda3
ENV_NAME=noisepo_acp

LOG_DIR=${DATA_DIR}/logs
mkdir -p ${DATA_DIR}/{checkpoints,results,logs,hf_cache}
MASTER_LOG="${LOG_DIR}/acp_run_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "============================================"
echo " NaCPO ACP Training"
echo " Start:   $(date)"
echo " Host:    $(hostname)"
echo " Project: ${PROJECT_DIR}"
echo "============================================"

# ========== ENVIRONMENT ==========
if [ -d "${CONDA_BASE}/envs/${ENV_NAME}" ]; then
    echo "[env] Activating conda env: ${ENV_NAME}"
    source ${CONDA_BASE}/bin/activate ${ENV_NAME}
else
    echo "[env] Conda env '${ENV_NAME}' not found, creating..."
    source ${CONDA_BASE}/bin/activate
    conda create -n ${ENV_NAME} python=3.10 -y
    conda activate ${ENV_NAME}
    pip install torch==2.3.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
    pip install "trl>=0.29" "peft>=0.12" "transformers>=4.45" "datasets>=2.20" \
        "accelerate>=0.34" deepspeed scipy matplotlib \
        sentence-transformers rich pyyaml hydra-core \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com
    echo "[env] Conda env created and packages installed."
fi

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${DATA_DIR}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

cd ${PROJECT_DIR}

# ========== PRE-FLIGHT CHECK ==========
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
n = torch.cuda.device_count()
print(f'GPUs: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  {i}: {name} ({mem:.0f} GB)')
import trl; print(f'TRL: {trl.__version__}')
import peft; print(f'PEFT: {peft.__version__}')
import transformers; print(f'Transformers: {transformers.__version__}')
"

echo ""
echo "[check] Model path: ${SHARE_DIR}/Qwen3.5-9B"
ls -la ${SHARE_DIR}/Qwen3.5-9B/config.json 2>/dev/null && echo "[check] Model found." || echo "[WARN] Model not found!"
echo ""

# ========== RUN FULL SWEEP ==========
export NACPO_CONFIG="${PROJECT_DIR}/configs/nacpo_acp.yaml"

echo "============================================"
echo " Starting NaCPO sweep (baselines + 54 configs)"
echo " Config:  ${NACPO_CONFIG}"
echo " Output:  ${DATA_DIR}"
echo "============================================"

bash scripts/run_nacpo_sweep.sh 2>&1

echo "============================================"
echo " NaCPO ACP Training Complete: $(date)"
echo " Checkpoints: ${DATA_DIR}/checkpoints"
echo " Results:     ${DATA_DIR}/results"
echo " Log:         ${MASTER_LOG}"
echo "============================================"
