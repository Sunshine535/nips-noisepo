#!/bin/bash
# ============================================================================
# NaCPO Sweep: 3 noise types × 4 schedules × 3 noise rates = 36 configs
# Plus 4 baselines: standard DPO, DPO+label smoothing, SimPO, IPO
# 2 seeds each for top configs
# Total: ~48 training runs
# Hardware: 8x A100-80GB with DeepSpeed ZeRO-3
# ============================================================================
set -euo pipefail

# Activate venv if available
_PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_PROJ_ROOT/.venv/bin/activate" ]; then source "$_PROJ_ROOT/.venv/bin/activate"; fi
export PATH="$HOME/.local/bin:$PATH"

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/nacpo_configs.yaml"

CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
RESULTS_DIR="${PROJECT_DIR}/results"
LOG_DIR="${PROJECT_DIR}/logs"
DS_CONFIG="${PROJECT_DIR}/configs/ds_config_sweep.json"

mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR" "$LOG_DIR"

NUM_GPUS=8

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1"; }

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

NOISE_TYPES=(random_flip confidence_weighted semantic_swap)
SCHEDULES=(uniform ascending descending adversarial)
NOISE_RATES=(0.05 0.1 0.2)
SEEDS=(42 43)

run_train() {
    local tag=$1
    local schedule=$2
    local noise_type=$3
    local noise_rate=$4
    local seed=$5
    local extra_args="${6:-}"

    local output="${CHECKPOINT_DIR}/${tag}"

    if [ -f "${output}/train_metrics.json" ]; then
        log "Skipping $tag (already done)"
        return 0
    fi

    log "Training: $tag"
    deepspeed --num_gpus $NUM_GPUS \
        "${SCRIPT_DIR}/train_nacpo.py" \
        --config "$CONFIG" \
        --noise_schedule "$schedule" \
        --noise_type "$noise_type" \
        --noise_rate "$noise_rate" \
        --seed "$seed" \
        --output_dir "$output" \
        $extra_args \
        2>&1 | tee "${LOG_DIR}/train_${tag}.log"
    return $?
}

run_eval() {
    local tag=$1
    local ckpt_dir=$2

    if [ ! -d "$ckpt_dir" ]; then
        log "Skip eval: $tag (no checkpoint)"
        return 0
    fi

    if [ -f "${RESULTS_DIR}/eval_alignment_${tag}.json" ]; then
        log "Skip eval: $tag (already evaluated)"
        return 0
    fi

    log "Evaluating: $tag"
    python "${SCRIPT_DIR}/eval_alignment.py" \
        --config "$CONFIG" \
        --checkpoint_dir "$ckpt_dir" \
        --output_dir "$RESULTS_DIR" \
        --tag "$tag" \
        --eval_all \
        2>&1 | tee "${LOG_DIR}/eval_${tag}.log"
}

# ============================================================================
# Phase 1: Baselines (4 configs × 2 seeds = 8 runs)
# ============================================================================
log "========================================="
log "[Phase 1/4] Training baselines"
log "========================================="

# Standard DPO (no noise)
for seed in "${SEEDS[@]}"; do
    run_train "baseline_dpo_seed${seed}" "none" "none" "0.0" "$seed"
done

# DPO + label smoothing (beta=0.05, small uniform noise as proxy)
for seed in "${SEEDS[@]}"; do
    run_train "baseline_label_smooth_seed${seed}" "uniform" "random_flip" "0.02" "$seed" \
        "--beta 0.05"
done

# SimPO-style (higher beta, no noise)
for seed in "${SEEDS[@]}"; do
    run_train "baseline_simpo_seed${seed}" "none" "none" "0.0" "$seed" \
        "--beta 0.3"
done

# IPO-style (loss_type change handled at eval, train with beta=0.5)
for seed in "${SEEDS[@]}"; do
    run_train "baseline_ipo_seed${seed}" "none" "none" "0.0" "$seed" \
        "--beta 0.5"
done

# ============================================================================
# Phase 2: NaCPO sweep (36 configs, seed=42 first)
# ============================================================================
log "========================================="
log "[Phase 2/4] NaCPO sweep (36 configs)"
log "========================================="

TOTAL_RUNS=$(( ${#NOISE_TYPES[@]} * ${#SCHEDULES[@]} * ${#NOISE_RATES[@]} ))
CURRENT=0

for noise_type in "${NOISE_TYPES[@]}"; do
    for schedule in "${SCHEDULES[@]}"; do
        for noise_rate in "${NOISE_RATES[@]}"; do
            CURRENT=$((CURRENT + 1))
            TAG="${schedule}_${noise_type}_nr${noise_rate}_seed42"
            log "[$CURRENT/$TOTAL_RUNS] $TAG"
            run_train "$TAG" "$schedule" "$noise_type" "$noise_rate" 42
        done
    done
done

# ============================================================================
# Phase 3: Second seed for top configs (identified by eval)
# ============================================================================
log "========================================="
log "[Phase 3/4] Evaluating sweep + second seed runs"
log "========================================="

# Quick eval of all seed=42 runs to find top configs
for noise_type in "${NOISE_TYPES[@]}"; do
    for schedule in "${SCHEDULES[@]}"; do
        for noise_rate in "${NOISE_RATES[@]}"; do
            TAG="${schedule}_${noise_type}_nr${noise_rate}_seed42"
            run_eval "$TAG" "${CHECKPOINT_DIR}/${TAG}"
        done
    done
done

# Evaluate baselines
for seed in "${SEEDS[@]}"; do
    for base in baseline_dpo baseline_label_smooth baseline_simpo baseline_ipo; do
        TAG="${base}_seed${seed}"
        run_eval "$TAG" "${CHECKPOINT_DIR}/${TAG}"
    done
done

# Find top configs by TruthfulQA accuracy and run seed=43
log "Selecting top configs for second seed..."
TOP_CONFIGS=$(python -c "
import json, os, glob

results = {}
for f in glob.glob('${RESULTS_DIR}/eval_alignment_*_seed42.json'):
    tag = os.path.basename(f).replace('eval_alignment_', '').replace('.json', '')
    with open(f) as fh:
        data = json.load(fh)
    tqa = data.get('truthfulqa/accuracy', 0) or 0
    results[tag] = tqa

ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
for tag, score in ranked[:12]:
    base = tag.replace('_seed42', '')
    print(base)
" 2>/dev/null || echo "")

if [ -n "$TOP_CONFIGS" ]; then
    log "Running seed=43 for top configs"
    while IFS= read -r config; do
        parts=(${config//_/ })
        schedule="${parts[0]}"
        noise_type="${parts[1]}"
        noise_rate="${config##*nr}"

        TAG="${config}_seed43"
        run_train "$TAG" "$schedule" "$noise_type" "$noise_rate" 43
        run_eval "$TAG" "${CHECKPOINT_DIR}/${TAG}"
    done <<< "$TOP_CONFIGS"
fi

# ============================================================================
# Phase 4: Verify 27B validation placeholder
# ============================================================================
log "========================================="
log "[Phase 4/4] Summary"
log "========================================="

log "NOTE: 27B validation should be run separately with larger GPU allocation"
log "Use: python scripts/train_nacpo.py --model_name Qwen/Qwen3.5-27B --noise_schedule <best> --noise_type <best> --noise_rate <best>"

# Count completed runs
N_TRAINED=$(ls -d ${CHECKPOINT_DIR}/*/train_metrics.json 2>/dev/null | wc -l)
N_EVALED=$(ls ${RESULTS_DIR}/eval_alignment_*.json 2>/dev/null | wc -l)

log "========================================="
log "NaCPO sweep complete!"
log "  Trained:   $N_TRAINED checkpoints"
log "  Evaluated: $N_EVALED results"
log "  Checkpoints: $CHECKPOINT_DIR"
log "  Results:     $RESULTS_DIR"
log "========================================="
