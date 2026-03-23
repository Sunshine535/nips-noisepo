#!/bin/bash
# ============================================================================
# NaCPO ACP Training — Startup Script
#
# Container has: PyTorch 2.3.0, CUDA 12.4, DeepSpeed 0.14.2, TRL 0.9.4,
#                PEFT 0.11.1, Transformers 4.41.2, Datasets 2.19.2
# Only sentence-transformers is missing -> install system-wide (no --target).
#
# Startup command:
#   bash /data/szs/250010072/nwh/nips-noisepo/run_acp.sh
# ============================================================================
set -euo pipefail

PROJECT_DIR=/data/szs/250010072/nwh/nips-noisepo
DATA_DIR=/data/szs/250010072/nwh/noisepo_data
SHARE_DIR=/data/szs/share

LOG_DIR=${DATA_DIR}/logs
mkdir -p ${DATA_DIR}/{checkpoints,results,logs,hf_cache}
LOG="${LOG_DIR}/acp_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================"
echo " NaCPO ACP Training"
echo " $(date) | $(hostname)"
echo " GPUs: ${SENSECORE_ACCELERATE_DEVICE_COUNT:-unknown}"
echo "============================================"

# ========== UPGRADE PACKAGES FOR QWEN3.5 SUPPORT ==========
# Container has transformers 4.41.2 which doesn't know qwen3_5 arch.
# Upgrade HF stack together to stay compatible. Torch 2.3.0 is kept.
MARKER=/data/szs/250010072/nwh/.acp_pkg_done_v3
if [ ! -f "${MARKER}" ]; then
    echo "[env] Upgrading HF packages for Qwen3.5 support..."
    pip install --upgrade transformers trl peft accelerate datasets \
        sentence-transformers 2>&1 | tail -15
    touch "${MARKER}"
    echo "[env] Done."
else
    echo "[env] Packages already upgraded."
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
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} ({p.total_memory / 1e9:.0f}GB)')
import trl, peft, transformers, accelerate, deepspeed
print(f'TRL {trl.__version__}, PEFT {peft.__version__}, Transformers {transformers.__version__}')
print(f'Accelerate {accelerate.__version__}, DeepSpeed {deepspeed.__version__}')
"
ls ${SHARE_DIR}/Qwen3.5-9B/config.json 2>/dev/null \
    && echo "[ok] Model found." \
    || echo "[WARN] Model not at ${SHARE_DIR}/Qwen3.5-9B!"

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
