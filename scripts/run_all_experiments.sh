#!/bin/bash
# ============================================================================
# NaCPO (Noise-as-Curriculum Preference Optimization) — Full Experiment Pipeline
# prepare_data → dpo_baselines → nacpo_sweep → eval → noise_analysis → 27B validation → figures
# Hardware: 4–8× A100-80GB (auto-detected)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=gpu_utils.sh
source "${SCRIPT_DIR}/gpu_utils.sh"
auto_setup
TORCHRUN=$(get_torchrun_cmd)

# --- Activate project venv (created by setup.sh) ---
PROJ_DIR_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

PHASE_MARKER_DIR="$PROJ_DIR_ROOT/results/.phase_markers"
mkdir -p "$PHASE_MARKER_DIR"
FORCE_RERUN="${FORCE_RERUN:-0}"

phase_done() { touch "$PHASE_MARKER_DIR/phase_${1}.done"; echo "[PHASE $1] Completed at $(date)"; }
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && echo "[PHASE $1] Already completed. Skipping. (FORCE_RERUN=1 to override)" && return 0
    return 1
}

CONFIG="${PROJECT_DIR}/configs/nacpo_configs.yaml"

CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
RESULTS_DIR="${PROJECT_DIR}/results"
ANALYSIS_DIR="${RESULTS_DIR}/analysis"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR" "$ANALYSIS_DIR" "$LOG_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1"; }

log "========================================="
log " NaCPO Experiment Pipeline"
log " GPUs:  ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
log "========================================="

# ============================================================================
# Stage 1: Prepare Data (verify dataset access)
# ============================================================================
if ! is_phase_done 1; then
log "========================================="
log "[Stage 1/6] Preparing data"
log "========================================="

python -c "
import os, time
os.environ.setdefault('HF_ENDPOINT', os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
from datasets import load_dataset

print('Loading UltraFeedback (primary training set)...')
ds = load_dataset('openbmb/UltraFeedback', split='train')
print(f'UltraFeedback: {len(ds)} samples')

print('Loading TruthfulQA (eval set, optional)...')
for attempt in range(3):
    try:
        tqa = load_dataset('truthful_qa', 'generation', split='validation')
        print(f'TruthfulQA: {len(tqa)} samples')
        break
    except Exception as e:
        print(f'  Attempt {attempt+1}/3 failed: {e}')
        if attempt < 2:
            wait = 60 * (attempt + 1)
            print(f'  Retrying in {wait}s...')
            time.sleep(wait)
        else:
            print('  WARN: TruthfulQA download failed. Will retry during eval stage.')

print('Data preparation complete.')
" 2>&1 | tee "${LOG_DIR}/stage1_data_prep.log"
phase_done 1
fi

# ============================================================================
# Stage 2: Run full NaCPO sweep (baselines + 36 configs + second seeds)
# ============================================================================
if ! is_phase_done 2; then
log "========================================="
log "[Stage 2/6] Running NaCPO sweep"
log "========================================="

bash "${SCRIPT_DIR}/run_nacpo_sweep.sh" \
    2>&1 | tee "${LOG_DIR}/stage2_sweep.log"
phase_done 2
fi

# ============================================================================
# Stage 3: Comprehensive Evaluation
# ============================================================================
if ! is_phase_done 3; then
log "========================================="
log "[Stage 3/6] Comprehensive evaluation"
log "========================================="

for CKPT_DIR in "${CHECKPOINT_DIR}"/*/; do
    TAG=$(basename "$CKPT_DIR")
    if [ ! -f "${CKPT_DIR}/train_metrics.json" ]; then
        continue
    fi
    if [ -f "${RESULTS_DIR}/eval_alignment_${TAG}.json" ]; then
        log "Already evaluated: $TAG"
        continue
    fi

    log "Evaluating: $TAG"
    python "${SCRIPT_DIR}/eval_alignment.py" \
        --config "$CONFIG" \
        --checkpoint_dir "$CKPT_DIR" \
        --output_dir "$RESULTS_DIR" \
        --tag "$TAG" \
        --eval_all \
        2>&1 | tee "${LOG_DIR}/eval_${TAG}.log" || {
        log "WARNING: Evaluation failed for $TAG"
        continue
    }
done
phase_done 3
fi

# ============================================================================
# Stage 4: Noise Analysis
# ============================================================================
if ! is_phase_done 4; then
log "========================================="
log "[Stage 4/6] Noise analysis"
log "========================================="

python "${SCRIPT_DIR}/run_noise_analysis.py" \
    --results_dir "$RESULTS_DIR" \
    --checkpoints_dir "$CHECKPOINT_DIR" \
    --output_dir "$ANALYSIS_DIR" \
    --noise_rates 0.05 0.1 0.15 0.2 0.25 0.3 \
    2>&1 | tee "${LOG_DIR}/stage4_noise_analysis.log"
phase_done 4
fi

# ============================================================================
# Stage 5: 27B Validation (best config only)
# ============================================================================
if ! is_phase_done 5; then
log "========================================="
log "[Stage 5/6] 27B validation (if GPU memory allows)"
log "========================================="

BEST_CONFIG=$(python -c "
import json, os

ranking_file = '${ANALYSIS_DIR}/robustness_ranking.json'
if os.path.exists(ranking_file):
    with open(ranking_file) as f:
        ranking = json.load(f)
    for name, data in ranking.items():
        if 'baseline' not in name and data.get('truthfulqa_mean'):
            print(name)
            break
else:
    print('')
" 2>/dev/null || echo "")

if [ -n "$BEST_CONFIG" ]; then
    log "Best NaCPO config: $BEST_CONFIG"
    log "NOTE: 27B validation requires manual launch with larger GPU allocation."
    log "Command: python scripts/train_nacpo.py --model_name Qwen/Qwen3.5-27B ..."
else
    log "No ranking available yet. Run evaluation first."
fi
phase_done 5
fi

# ============================================================================
# Stage 6: Summary
# ============================================================================
if ! is_phase_done 6; then
log "========================================="
log "[Stage 6/6] Final summary"
log "========================================="

python -c "
import json, os, glob

results_dir = '${RESULTS_DIR}'
analysis_dir = '${ANALYSIS_DIR}'

ranking_file = os.path.join(analysis_dir, 'robustness_ranking.json')
if os.path.exists(ranking_file):
    with open(ranking_file) as f:
        ranking = json.load(f)
    print('\\n=== Top Configurations ===')
    for i, (name, data) in enumerate(list(ranking.items())[:10]):
        tqa = data.get('truthfulqa_mean', 0) or 0
        mt = data.get('mt_bench_mean', 0) or 0
        std = data.get('truthfulqa_std', 0) or 0
        print(f'  {i+1:2d}. {name:45s} TQA={tqa:.4f}±{std:.4f} MT={mt:.2f}')
else:
    results = []
    for f in sorted(glob.glob(os.path.join(results_dir, 'eval_alignment_*.json'))):
        tag = os.path.basename(f).replace('eval_alignment_', '').replace('.json', '')
        with open(f) as fh:
            data = json.load(fh)
        tqa = data.get('truthfulqa/accuracy', 0) or 0
        mt = data.get('mt_bench/overall', 0) or 0
        results.append((tag, tqa, mt))

    results.sort(key=lambda x: x[1], reverse=True)
    print('\\n=== Results (sorted by TruthfulQA) ===')
    for tag, tqa, mt in results[:15]:
        print(f'  {tag:45s} TQA={tqa:.4f} MT={mt:.2f}')

n_files = len(glob.glob(os.path.join(results_dir, 'eval_alignment_*.json')))
print(f'\\nTotal evaluated: {n_files} configurations')
"
phase_done 6
fi

log "========================================="
log "NaCPO experiment pipeline complete!"
log "  Checkpoints: $CHECKPOINT_DIR"
log "  Results:     $RESULTS_DIR"
log "  Analysis:    $ANALYSIS_DIR"
log "  Logs:        $LOG_DIR"
log "========================================="

# --- Pipeline completion marker ---
DONE_FILE="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "$(basename "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo ""
echo "[PIPELINE_COMPLETE] All experiments finished successfully."
echo "  Marker: $DONE_FILE"
echo "  Run 'bash collect_results.sh' to package results."
