#!/bin/bash
# ============================================================================
# NaCPO Training Job Submission Script
# Following ACP pattern - ready for manual invocation
# ============================================================================

# ========== VARIABLES (modify as needed) ==========
GPU_NUMS=${GPU_NUMS:-8}
CONDA_BASE=/data/szs/250010072/szs/anaconda3
ENV_NAME=nips_noisepo
PROJECT_DIR=/data/szs/250010072/nwh/nips-noisepo
DATA_DIR=/data/szs/250010072/nwh/data
SHARE_DIR=/data/szs/share

LOG_FILE_PATH="${DATA_DIR}/logs"
mkdir -p "${LOG_FILE_PATH}"
LOG_FILE="${LOG_FILE_PATH}/train_$(date +%Y%m%d_%H%M%S).log"

CHECKPOINT_DIR="${DATA_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

# Model paths (use shared models)
MODEL_PATH="${SHARE_DIR}/Qwen3.5-9B"
JUDGE_MODEL_PATH="${SHARE_DIR}/Qwen3.5-27B"
EMBEDDING_MODEL_PATH="${SHARE_DIR}/all-MiniLM-L6-v2"

# ========== ENVIRONMENT ==========
source ${CONDA_BASE}/bin/activate
conda activate ${ENV_NAME}

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${DATA_DIR}/hf_cache"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_NUMS-1)))
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
mkdir -p "${HF_HOME}"

cd ${PROJECT_DIR}

# ========== TRAINING COMMAND ==========
echo "============================================" | tee -a ${LOG_FILE}
echo " NaCPO Training Start: $(date)" | tee -a ${LOG_FILE}
echo " GPUs: ${GPU_NUMS}" | tee -a ${LOG_FILE}
echo " Model: ${MODEL_PATH}" | tee -a ${LOG_FILE}
echo " Output: ${CHECKPOINT_DIR}" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}

bash scripts/run_nacpo_sweep.sh 2>&1 | tee -a ${LOG_FILE}

echo "============================================" | tee -a ${LOG_FILE}
echo " NaCPO Training Complete: $(date)" | tee -a ${LOG_FILE}
echo "============================================" | tee -a ${LOG_FILE}
