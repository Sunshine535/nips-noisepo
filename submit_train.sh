#!/bin/bash
# ============================================================================
# NaCPO Training Job — Ready for Manual Invocation
# Following ACP pattern: set variables, activate env, run training
#
# Usage (on GPU node):
#   bash submit_train.sh              # Full sweep (8 GPU default)
#   GPU_NUMS=4 bash submit_train.sh   # With 4 GPUs
# ============================================================================
set -euo pipefail

# ========== VARIABLES (modify as needed) ==========
GPU_NUMS=${GPU_NUMS:-8}
CONDA_BASE=/data/szs/250010072/szs/anaconda3
ENV_NAME=llama_factory
PROJECT_DIR=/data/szs/250010072/nwh/nips-noisepo
DATA_DIR=/data/szs/share/noisepo
SHARE_DIR=/data/szs/share

LOG_FILE_PATH="${DATA_DIR}/logs"
mkdir -p "${LOG_FILE_PATH}"
LOG_FILE="${LOG_FILE_PATH}/train_$(date +%Y%m%d_%H%M%S).log"

CHECKPOINT_DIR="${DATA_DIR}/checkpoints"
RESULTS_DIR="${DATA_DIR}/results"
mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}"

MODEL_PATH="${SHARE_DIR}/Qwen3.5-9B"
JUDGE_MODEL_PATH="${SHARE_DIR}/Qwen3.5-27B"
EMBEDDING_MODEL_PATH="${SHARE_DIR}/all-MiniLM-L6-v2"

# ========== ENVIRONMENT ==========
source ${CONDA_BASE}/bin/activate
conda activate ${ENV_NAME}

# HF_ENDPOINT removed (use default huggingface.co)
export HF_HOME="${DATA_DIR}/hf_cache"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_NUMS-1)))
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

cd ${PROJECT_DIR}

# Use server config (local model paths)
export NACPO_CONFIG="${PROJECT_DIR}/configs/nacpo_server.yaml"
export NACPO_DATA_DIR="${DATA_DIR}"

# ========== PRE-FLIGHT CHECK ==========
echo "============================================" | tee -a ${LOG_FILE}
echo " NaCPO Training Job" | tee -a ${LOG_FILE}
echo " Start:  $(date)" | tee -a ${LOG_FILE}
echo " GPUs:   ${GPU_NUMS}" | tee -a ${LOG_FILE}
echo " Model:  ${MODEL_PATH}" | tee -a ${LOG_FILE}
echo " Config: ${NACPO_CONFIG}" | tee -a ${LOG_FILE}
echo " Ckpt:   ${CHECKPOINT_DIR}" | tee -a ${LOG_FILE}
echo " Log:    ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}

python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  {i}: {torch.cuda.get_device_name(i)}')
" 2>&1 | tee -a ${LOG_FILE}

# ========== RUN FULL SWEEP ==========
bash scripts/run_nacpo_sweep.sh 2>&1 | tee -a ${LOG_FILE}

echo "============================================" | tee -a ${LOG_FILE}
echo " NaCPO Training Complete: $(date)" | tee -a ${LOG_FILE}
echo " Results: ${RESULTS_DIR}" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}
