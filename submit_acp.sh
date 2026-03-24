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
PROJECT_DIR="/data/szs/250010072/nwh/nips-noisepo"

CONTAINER_IMAGE="registry.cn-sh-01.sensecore.cn/ccr-zhicheng-01/nacpo-train:v1.0-torch2.4.1"

echo "Submitting: ${GPU_NUMS}x N6IS-80G"

sco acp jobs create \
  --workspace-name=${WORKSPACE} \
  --aec2-name=${CLUSTER} \
  --job-name=noisepo-sweep \
  --container-image-url="${CONTAINER_IMAGE}" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=n6ls.iu.i40.${GPU_NUMS} \
  --priority=normal \
  --command="bash ${PROJECT_DIR}/run_acp.sh"

echo ""
echo "Monitor:  sco acp jobs list --workspace-name=${WORKSPACE}"
echo 'Login:    Jobid=pt-xxx; sco acp jobs exec --workspace-name='${WORKSPACE}' --worker-name=$(sco acp jobs get-workers --workspace-name='${WORKSPACE}' $Jobid | grep worker-0 | awk -F"|" "{print \$2}" | xargs) $Jobid'
