#!/bin/bash
# ============================================================================
# NaCPO ACP Training — Startup Script
#
# Container: ubuntu22.04, PyTorch 2.3.0, CUDA 12.4, DeepSpeed 0.14.2,
#            LLaMA-Factory 0.8.1 (most deps pre-installed)
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

# ========== CHECK / INSTALL PACKAGES ==========
export PYTHONPATH="${PIP_PKGS}:${PYTHONPATH:-}"

echo "[env] Checking pre-installed packages..."
python -c "
import sys
pkgs = {}
for mod, pkg in [('torch','torch'),('trl','trl'),('peft','peft'),
                 ('transformers','transformers'),('datasets','datasets'),
                 ('accelerate','accelerate'),('deepspeed','deepspeed'),
                 ('scipy','scipy'),('yaml','pyyaml'),('rich','rich'),
                 ('sentence_transformers','sentence-transformers')]:
    try:
        m = __import__(mod); pkgs[pkg] = getattr(m, '__version__', 'ok')
    except: pkgs[pkg] = None
for p, v in pkgs.items():
    print(f'  {p}: {v or \"MISSING\"}')
missing = [p for p,v in pkgs.items() if v is None]
if missing:
    print(f'NEED_INSTALL={\" \".join(missing)}')
else:
    print('ALL_INSTALLED')
" 2>&1 | tee /tmp/_pkg_check.txt

if grep -q "NEED_INSTALL" /tmp/_pkg_check.txt; then
    MISSING=$(grep "NEED_INSTALL=" /tmp/_pkg_check.txt | cut -d= -f2)
    echo "[env] Installing missing: ${MISSING}"

    pip install --target="${PIP_PKGS}" --timeout=60 ${MISSING} 2>&1 | tail -15 || \
    pip install --target="${PIP_PKGS}" --timeout=60 ${MISSING} \
        -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
        --trusted-host pypi.tuna.tsinghua.edu.cn 2>&1 | tail -15 || \
    echo "[WARN] pip install failed, continuing with what's available..."

    echo "[env] Install done."
else
    echo "[env] All packages available."
fi

# Upgrade trl if too old (need >=0.8 for DPOConfig)
python -c "
import trl
v = tuple(int(x) for x in trl.__version__.split('.')[:2])
print(f'TRL version: {trl.__version__} -> {\"OK\" if v >= (0,8) else \"TOO_OLD\"}')
if v < (0,8): exit(1)
" 2>/dev/null || {
    echo "[env] Upgrading trl..."
    pip install --target="${PIP_PKGS}" --timeout=60 "trl>=0.8" 2>&1 | tail -10 || \
    pip install --target="${PIP_PKGS}" --timeout=60 "trl>=0.8" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
        --trusted-host pypi.tuna.tsinghua.edu.cn 2>&1 | tail -10 || true
}

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
import trl, peft, transformers, accelerate
print(f'TRL {trl.__version__}, PEFT {peft.__version__}, Transformers {transformers.__version__}, Accelerate {accelerate.__version__}')
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
