#!/bin/bash
# ===========================================================================
# Build & Push NaCPO Training Image to SenseCore Registry
#
# Usage:
#   bash docker/build_push.sh                      # ACP base (need login first)
#   bash docker/build_push.sh --public              # Public PyTorch base
#   NAMESPACE=my-ns bash docker/build_push.sh       # Custom namespace
#
# Prerequisites:
#   podman login registry.cn-sh-01.sensecore.cn --username zhicheng-250010072
# ===========================================================================
set -euo pipefail

REGISTRY="registry.cn-sh-01.sensecore.cn"
NAMESPACE="${NAMESPACE:-zhicheng-250010072}"
IMAGE_NAME="nacpo-train"
TAG="v1.0-torch2.4.1-$(date +%Y%m%d)"
FULL_TAG="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${TAG}"
LATEST_TAG="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:latest"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Auto-detect container runtime
if command -v docker &>/dev/null; then
    CTR=docker
elif command -v podman &>/dev/null; then
    CTR=podman
else
    echo "ERROR: No container runtime found (docker or podman)"
    exit 1
fi

USE_PUBLIC=false
[[ "${1:-}" == "--public" ]] && USE_PUBLIC=true

echo "============================================"
echo " NaCPO Image Builder"
echo "============================================"
echo "Runtime:    ${CTR}"
echo "Registry:   ${REGISTRY}"
echo "Namespace:  ${NAMESPACE}"
echo "Image:      ${IMAGE_NAME}:${TAG}"
echo "Base:       $(${USE_PUBLIC} && echo 'Public PyTorch' || echo 'ACP base')"
echo "============================================"

DOCKERFILE="${SCRIPT_DIR}/Dockerfile"
${USE_PUBLIC} && DOCKERFILE="${SCRIPT_DIR}/Dockerfile.public"

echo ""
echo "[1/4] Building image..."
${CTR} build \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${DOCKERFILE}" \
    "${PROJECT_DIR}"
echo ""

echo "[2/4] Logging into registry..."
${CTR} login "${REGISTRY}" --username zhicheng-250010072
echo ""

echo "[3/4] Tagging..."
${CTR} tag "${IMAGE_NAME}:${TAG}" "${FULL_TAG}"
${CTR} tag "${IMAGE_NAME}:${TAG}" "${LATEST_TAG}"
echo "  ${FULL_TAG}"
echo "  ${LATEST_TAG}"
echo ""

echo "[4/4] Pushing to registry..."
${CTR} push "${FULL_TAG}"
${CTR} push "${LATEST_TAG}"
echo ""

echo "============================================"
echo " Done!"
echo " Image: ${FULL_TAG}"
echo ""
echo " Update submit_acp.sh CONTAINER_IMAGE to:"
echo "   ${FULL_TAG}"
echo "============================================"
