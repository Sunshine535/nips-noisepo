#!/bin/bash
# ============================================================================
# NaCPO ACP Training — Startup Script
#
# Custom image: nacpo-train:v3.0-torch2.10-cu128-full
#   PyTorch 2.10.0+cu128, Transformers 5.3.0, TRL 0.29.1, PEFT 0.18.1,
#   DeepSpeed 0.18.8 (fused_adam+cpu_adam pre-compiled), Accelerate 1.13.0,
#   SentenceTransformers 5.3.0, CUDA nvcc 12.8 (full toolkit)
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

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${DATA_DIR}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH=${CUDA_HOME}/bin:${PATH}

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

python -c "
import deepspeed
from deepspeed.ops.op_builder import FusedAdamBuilder, CPUAdamBuilder
for name, builder in [('fused_adam', FusedAdamBuilder()), ('cpu_adam', CPUAdamBuilder())]:
    status = 'PRE-BUILT' if builder.is_installed() else 'JIT (will compile at runtime)'
    print(f'  DS op {name}: {status}')
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
