#!/bin/bash
# ============================================================================
# NaCPO ACP Training — Startup Script for SCO ACP GPU Container
#
# Uses the container's base Python (PyTorch 2.3.0 + CUDA 12.4 + DeepSpeed).
# Extra packages are pip installed to a persistent cache on cloud disk.
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
PIP_PKGS_DIR=${USER_DIR}/nwh/.pip_packages

LOG_DIR=${DATA_DIR}/logs
mkdir -p ${DATA_DIR}/{checkpoints,results,logs,hf_cache} ${PIP_PKGS_DIR}
MASTER_LOG="${LOG_DIR}/acp_run_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "============================================"
echo " NaCPO ACP Training"
echo " Start:   $(date)"
echo " Host:    $(hostname)"
echo " Project: ${PROJECT_DIR}"
echo "============================================"

# ========== INSTALL EXTRA PACKAGES (cached on cloud disk) ==========
export PYTHONPATH="${PIP_PKGS_DIR}:${PYTHONPATH:-}"

MARKER="${PIP_PKGS_DIR}/.install_done_v1"
if [ ! -f "${MARKER}" ]; then
    echo "[env] Installing extra packages to ${PIP_PKGS_DIR}..."
    pip install --target="${PIP_PKGS_DIR}" --no-deps \
        "trl>=0.29" "peft>=0.12" "transformers>=4.45" "datasets>=2.20" \
        "accelerate>=0.34" scipy matplotlib sentence-transformers \
        rich pyyaml hydra-core safetensors huggingface_hub tokenizers \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com 2>&1 | tail -20

    pip install --target="${PIP_PKGS_DIR}" \
        "trl>=0.29" "peft>=0.12" "transformers>=4.45" "datasets>=2.20" \
        "accelerate>=0.34" scipy matplotlib sentence-transformers \
        rich pyyaml hydra-core \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com 2>&1 | tail -20

    touch "${MARKER}"
    echo "[env] Package installation complete."
else
    echo "[env] Using cached packages from ${PIP_PKGS_DIR}"
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
try:
    import trl; print(f'TRL: {trl.__version__}')
except: print('TRL: import failed')
try:
    import peft; print(f'PEFT: {peft.__version__}')
except: print('PEFT: import failed')
try:
    import transformers; print(f'Transformers: {transformers.__version__}')
except: print('Transformers: import failed')
try:
    import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')
except: print('DeepSpeed: not available')
"

echo ""
echo "[check] Model: ${SHARE_DIR}/Qwen3.5-9B"
ls ${SHARE_DIR}/Qwen3.5-9B/config.json 2>/dev/null \
    && echo "[check] Model found." \
    || echo "[WARN] Model not found at ${SHARE_DIR}/Qwen3.5-9B!"
echo ""

# ========== RUN FULL SWEEP ==========
export NACPO_CONFIG="${PROJECT_DIR}/configs/nacpo_acp.yaml"

echo "============================================"
echo " Starting NaCPO sweep"
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
