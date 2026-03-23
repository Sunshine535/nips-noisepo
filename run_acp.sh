#!/bin/bash
# ============================================================================
# NaCPO ACP Training — Startup Script
#
# Container: ubuntu22.04, PyTorch 2.3.0, CUDA 12.4, DeepSpeed 0.14.2
# Cloud disk at /data/szs is directly accessible (no mount needed).
#
# Startup command for ACP web UI:
#   bash /data/szs/250010072/nwh/nips-noisepo/run_acp.sh
# ============================================================================
set -euo pipefail

PROJECT_DIR=/data/szs/250010072/nwh/nips-noisepo
DATA_DIR=/data/szs/250010072/nwh/noisepo_data
SHARE_DIR=/data/szs/share
PIP_PKGS=/data/szs/250010072/nwh/.pip_packages

LOG_DIR=${DATA_DIR}/logs
mkdir -p ${DATA_DIR}/{checkpoints,results,logs,hf_cache} ${PIP_PKGS}
LOG="${LOG_DIR}/acp_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================"
echo " NaCPO ACP Training"
echo " $(date) | $(hostname)"
echo " GPUs: ${SENSECORE_ACCELERATE_DEVICE_COUNT:-unknown}"
echo "============================================"

# ========== INSTALL EXTRA PACKAGES ==========
export PYTHONPATH="${PIP_PKGS}:${PYTHONPATH:-}"

MARKER="${PIP_PKGS}/.done_v2"
if [ ! -f "${MARKER}" ]; then
    echo "[env] Installing packages to ${PIP_PKGS}..."
    pip install --target="${PIP_PKGS}" \
        "trl>=0.29" "peft>=0.12" "transformers>=4.45" "datasets>=2.20" \
        "accelerate>=0.34" scipy matplotlib sentence-transformers \
        rich pyyaml safetensors huggingface_hub tokenizers \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com 2>&1 | tail -20
    touch "${MARKER}"
    echo "[env] Done."
else
    echo "[env] Using cached packages."
fi

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${DATA_DIR}/hf_cache"
export TOKENIZERS_PARALLELISM=false

cd ${PROJECT_DIR}

# ========== PRE-FLIGHT ==========
python -c "
import torch, sys
print(f'Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
n = torch.cuda.device_count()
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1e9:.0f}GB)')
import trl, peft, transformers
print(f'TRL {trl.__version__}, PEFT {peft.__version__}, Transformers {transformers.__version__}')
"
ls ${SHARE_DIR}/Qwen3.5-9B/config.json && echo "[ok] Model found." || echo "[WARN] Model missing!"

# ========== RUN SWEEP ==========
export NACPO_CONFIG="${PROJECT_DIR}/configs/nacpo_server.yaml"

echo ""
echo "Config: ${NACPO_CONFIG}"
echo "Output: ${DATA_DIR}"
echo ""

bash scripts/run_nacpo_sweep.sh 2>&1

echo ""
echo "Done: $(date)"
echo "Log: ${LOG}"
