#!/bin/bash
# NaCPO: Noise-as-Curriculum Preference Optimization
# Full sweep: 4 schedules x 3 noise types + baseline = 13 runs
# 8x A100-80GB with DeepSpeed ZeRO-3

set -euo pipefail

# Activate venv if available
_PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_PROJ_ROOT/.venv/bin/activate" ]; then source "$_PROJ_ROOT/.venv/bin/activate"; fi
export PATH="$HOME/.local/bin:$PATH"

# HF_ENDPOINT removed (use default huggingface.co)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/nacpo_configs.yaml"

CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
RESULTS_DIR="${PROJECT_DIR}/results"
LOG_DIR="${PROJECT_DIR}/logs"
DS_CONFIG="${PROJECT_DIR}/configs/ds_config.json"

mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR" "$LOG_DIR"

NUM_GPUS=8

# Create DeepSpeed config
cat > "$DS_CONFIG" << 'DSEOF'
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e7,
        "allgather_bucket_size": 5e7
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
DSEOF

SCHEDULES=(uniform ascending descending adversarial)
NOISE_TYPES=(random_flip confidence_weighted semantic_swap)
SEEDS=(42 43)

# =========================================
# Phase 1: Train baseline (no noise)
# =========================================
echo "========================================="
echo "[Phase 1] Training baseline (no noise)..."
echo "========================================="

for seed in "${SEEDS[@]}"; do
    TAG="baseline_seed${seed}"
    OUTPUT="${CHECKPOINT_DIR}/${TAG}"

    if [ -f "${OUTPUT}/train_metrics.json" ]; then
        echo "Skipping $TAG (already done)"
        continue
    fi

    echo "Training: $TAG"
    deepspeed --num_gpus $NUM_GPUS \
        "${SCRIPT_DIR}/train_nacpo.py" \
        --config "$CONFIG" \
        --schedule none \
        --noise_type none \
        --seed "$seed" \
        --output_dir "$OUTPUT" \
        --deepspeed "$DS_CONFIG" \
        2>&1 | tee "${LOG_DIR}/train_${TAG}.log"
done

# =========================================
# Phase 2: Train all noise combinations
# =========================================
echo "========================================="
echo "[Phase 2] Training noise combinations..."
echo "========================================="

TOTAL_RUNS=$(( ${#SCHEDULES[@]} * ${#NOISE_TYPES[@]} * ${#SEEDS[@]} ))
CURRENT=0

for schedule in "${SCHEDULES[@]}"; do
    for noise_type in "${NOISE_TYPES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))
            TAG="${schedule}_${noise_type}_seed${seed}"
            OUTPUT="${CHECKPOINT_DIR}/${TAG}"

            if [ -f "${OUTPUT}/train_metrics.json" ]; then
                echo "[$CURRENT/$TOTAL_RUNS] Skipping $TAG (already done)"
                continue
            fi

            echo "[$CURRENT/$TOTAL_RUNS] Training: $TAG"
            deepspeed --num_gpus $NUM_GPUS \
                "${SCRIPT_DIR}/train_nacpo.py" \
                --config "$CONFIG" \
                --schedule "$schedule" \
                --noise_type "$noise_type" \
                --seed "$seed" \
                --output_dir "$OUTPUT" \
                --deepspeed "$DS_CONFIG" \
                2>&1 | tee "${LOG_DIR}/train_${TAG}.log"
        done
    done
done

# =========================================
# Phase 3: Evaluate all checkpoints
# =========================================
echo "========================================="
echo "[Phase 3] Evaluating all models..."
echo "========================================="

eval_checkpoint() {
    local ckpt_dir=$1
    local schedule=$2
    local noise_type=$3
    local seed=$4
    local tag="${schedule}_${noise_type}_seed${seed}"

    if [ ! -d "$ckpt_dir" ]; then
        echo "Skip eval: $tag (no checkpoint)"
        return
    fi

    if [ -f "${RESULTS_DIR}/eval_${tag}.json" ]; then
        echo "Skip eval: $tag (already evaluated)"
        return
    fi

    echo "Evaluating: $tag"
    python "${SCRIPT_DIR}/eval_nacpo.py" \
        --config "$CONFIG" \
        --checkpoint_dir "$ckpt_dir" \
        --schedule "$schedule" \
        --noise_type "$noise_type" \
        --seed "$seed" \
        --output_dir "$RESULTS_DIR" \
        --eval_all \
        2>&1 | tee "${LOG_DIR}/eval_${tag}.log"
}

# Evaluate baseline
for seed in "${SEEDS[@]}"; do
    eval_checkpoint "${CHECKPOINT_DIR}/baseline_seed${seed}" "none" "none" "$seed"
done

# Evaluate all combinations
for schedule in "${SCHEDULES[@]}"; do
    for noise_type in "${NOISE_TYPES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            eval_checkpoint \
                "${CHECKPOINT_DIR}/${schedule}_${noise_type}_seed${seed}" \
                "$schedule" "$noise_type" "$seed"
        done
    done
done

echo "========================================="
echo "NaCPO sweep complete!"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Results:     $RESULTS_DIR"
echo "========================================="
