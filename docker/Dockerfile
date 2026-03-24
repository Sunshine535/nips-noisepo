# ===========================================================================
# NaCPO Training Image — PyTorch 2.4.1 + HuggingFace + DeepSpeed
#
# Base: ACP official image (NCCL, AllReduce, TransformerEngine optimized)
# Upgrades: PyTorch 2.3 -> 2.4.1, HF stack for Qwen3.5 support
#
# Build:
#   docker build -t nacpo-train:latest -f docker/Dockerfile .
# ===========================================================================
FROM registry.cn-sh-01.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image

# Upgrade PyTorch to 2.4.1 (transformers >=5.x requires PyTorch >=2.4)
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install/upgrade HuggingFace ecosystem + DeepSpeed
RUN pip install --no-cache-dir \
    "transformers>=4.46.0" \
    "trl>=0.12.0" \
    "peft>=0.13.0" \
    "accelerate>=0.34.0" \
    "datasets>=2.21.0" \
    "sentence-transformers" \
    "evaluate>=0.4.0" \
    "deepspeed>=0.14.2" \
    "scipy>=1.14.0" \
    "pyyaml>=6.0.1"

# Verify key packages work
RUN python -c "\
import torch; \
print(f'PyTorch {torch.__version__}, CUDA support: {torch.version.cuda}'); \
import transformers; print(f'Transformers {transformers.__version__}'); \
import trl; print(f'TRL {trl.__version__}'); \
import peft; print(f'PEFT {peft.__version__}'); \
import deepspeed; print(f'DeepSpeed {deepspeed.__version__}'); \
import sentence_transformers; print(f'SentenceTransformers {sentence_transformers.__version__}'); \
print('All packages OK')"
