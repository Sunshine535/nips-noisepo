#!/bin/bash
# ============================================================================
# SCO ACP Job Submission — NaCPO Training
#
# Usage:
#   bash submit_acp.sh              # Submit 8-GPU job
#   GPU_NUMS=4 bash submit_acp.sh   # Submit 4-GPU job
#
# Manage:
#   sco acp jobs list --workspace-name=share_space_01g
#   sco acp jobs delete --workspace-name=share_space_01g pt-xxxxxxxx
# ============================================================================
set -euo pipefail

GPU_NUMS=${GPU_NUMS:-8}
WORKSPACE="share_space_01g"
CLUSTER="computing_cluster"
CONDA_ENV="noisepo_acp"

AFS_ROOT="/mnt/afs"
USER_DIR="${AFS_ROOT}/250010072"
PROJECT_DIR="${USER_DIR}/nwh/nips-noisepo"
DATA_DIR="${USER_DIR}/nwh/noisepo_data"
LOG_DIR="${DATA_DIR}/logs"

CONTAINER_IMAGE="registry.cn-sh-01.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image"
STORAGE_MOUNT="01995892-d478-76d8-aec7-13fd8284477e:/mnt/afs"

STARTUP_CMD="bash ${PROJECT_DIR}/run_acp.sh"

echo "============================================"
echo " Submitting NaCPO ACP Job"
echo "  Workspace: ${WORKSPACE}"
echo "  GPUs:      ${GPU_NUMS}x N6IS-80G"
echo "  Command:   ${STARTUP_CMD}"
echo "============================================"

sco acp jobs create \
  --workspace-name=${WORKSPACE} \
  --aec2-name=${CLUSTER} \
  --job-name=noisepo-sweep \
  --container-image-url="${CONTAINER_IMAGE}" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=n6ls.iu.i40.${GPU_NUMS} \
  --priority=normal \
  --storage-mount=${STORAGE_MOUNT} \
  --command="${STARTUP_CMD}"

echo ""
echo "Job submitted. Monitor with:"
echo "  sco acp jobs list --workspace-name=${WORKSPACE}"
echo ""
echo "Login to running node:"
echo '  Jobid=pt-xxxxxxxx; sco acp jobs exec --workspace-name='${WORKSPACE}' --worker-name=$(sco acp jobs get-workers --workspace-name='${WORKSPACE}' $Jobid | grep "worker-0" | awk -F"|" "{print \$2}" | xargs) $Jobid'
