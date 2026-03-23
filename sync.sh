#!/bin/bash
# ============================================================================
# Three-way Sync: Local <-> Git <-> Server
# Usage:
#   bash sync.sh push    # Local -> Git -> Server
#   bash sync.sh pull    # Server -> Git -> Local
#   bash sync.sh status  # Show sync status across all three
# ============================================================================
set -e

SERVER="szs_cpu"
SERVER_DIR="/data/szs/250010072/nwh/nips-noisepo"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

action="${1:-status}"

sync_push() {
    echo "=== [1/3] Committing local changes ==="
    cd "${LOCAL_DIR}"
    git add -A
    if git diff --cached --quiet; then
        echo "  No local changes to commit."
    else
        git commit -m "sync: $(date +%Y%m%d_%H%M%S)"
    fi

    echo "=== [2/3] Pushing to Git remote ==="
    git push origin main

    echo "=== [3/3] Pulling on server ==="
    ssh ${SERVER} "cd ${SERVER_DIR} && git pull origin main"

    echo "=== Push sync complete ==="
}

sync_pull() {
    echo "=== [1/3] Committing server changes ==="
    ssh ${SERVER} "cd ${SERVER_DIR} && git add -A && git diff --cached --quiet || git commit -m 'sync: server $(date +%Y%m%d_%H%M%S)' && git push origin main"

    echo "=== [2/3] Pulling to local ==="
    cd "${LOCAL_DIR}"
    git pull origin main

    echo "=== Pull sync complete ==="
}

sync_status() {
    echo "=== Local Status ==="
    cd "${LOCAL_DIR}"
    git log --oneline -3
    git status -sb
    echo ""

    echo "=== Remote (origin) ==="
    git fetch origin
    LOCAL_HASH=$(git rev-parse HEAD)
    REMOTE_HASH=$(git rev-parse origin/main)
    echo "  Local:  ${LOCAL_HASH:0:7}"
    echo "  Remote: ${REMOTE_HASH:0:7}"
    if [ "$LOCAL_HASH" = "$REMOTE_HASH" ]; then
        echo "  Status: IN SYNC"
    else
        echo "  Status: OUT OF SYNC"
    fi
    echo ""

    echo "=== Server Status ==="
    ssh ${SERVER} "cd ${SERVER_DIR} && git log --oneline -3 && git status -sb"
    SERVER_HASH=$(ssh ${SERVER} "cd ${SERVER_DIR} && git rev-parse HEAD")
    echo "  Server: ${SERVER_HASH:0:7}"
    if [ "$LOCAL_HASH" = "$SERVER_HASH" ]; then
        echo "  Status: IN SYNC with local"
    else
        echo "  Status: OUT OF SYNC with local"
    fi
}

case "$action" in
    push)  sync_push ;;
    pull)  sync_pull ;;
    status) sync_status ;;
    *)
        echo "Usage: bash sync.sh {push|pull|status}"
        exit 1
        ;;
esac
