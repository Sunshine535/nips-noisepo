#!/bin/bash
# ============================================================================
# Quick Test: Single NaCPO run for validation
# Run this first to verify the environment works before full sweep
# ============================================================================

CONDA_BASE=/data/szs/250010072/szs/anaconda3
ENV_NAME=llama_factory
PROJECT_DIR=/data/szs/250010072/nwh/nips-noisepo
DATA_DIR=/data/szs/share/noisepo
SHARE_DIR=/data/szs/share

source ${CONDA_BASE}/bin/activate
conda activate ${ENV_NAME}

# HF_ENDPOINT removed (use default huggingface.co)
export HF_HOME="${DATA_DIR}/hf_cache"
mkdir -p "${HF_HOME}"

cd ${PROJECT_DIR}

LOG_FILE="${DATA_DIR}/logs/quick_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname ${LOG_FILE})"

echo "=== Quick Test: Single baseline DPO run ===" | tee ${LOG_FILE}
echo "Start: $(date)" | tee -a ${LOG_FILE}

GPU_NUMS=${GPU_NUMS:-1}
export CUDA_VISIBLE_DEVICES=0

python scripts/train_nacpo.py \
    --config configs/nacpo_configs.yaml \
    --model_name "${SHARE_DIR}/Qwen3.5-9B" \
    --noise_schedule none \
    --noise_type none \
    --noise_rate 0.0 \
    --seed 42 \
    --output_dir "${DATA_DIR}/checkpoints/quick_test_baseline" \
    --max_train_samples 100 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    2>&1 | tee -a ${LOG_FILE}

echo "End: $(date)" | tee -a ${LOG_FILE}
echo "=== Quick Test Complete ===" | tee -a ${LOG_FILE}
