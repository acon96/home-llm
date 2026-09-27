#!/bin/bash
#
# Sync the llama_conversation custom component to a Home Assistant installation.
#
# Usage:
#   ./scripts/sync_to_vm.sh <host>
#
# Optional environment overrides:
#   HA_USER=pi ./scripts/sync_to_vm.sh 192.168.0.100
#   HA_REMOTE_PATH=/config/custom_components/llama_conversation ./scripts/sync_to_vm.sh 192.168.0.101

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
COMPONENT_DIR="$REPO_ROOT/custom_components/llama_conversation"
REMOTE_PATH="${HA_REMOTE_PATH:-/config/custom_components/llama_conversation}"
REMOTE_USER="${HA_USER:-root}" # ssh extension requires user to be root to use scp

usage() {
    echo "Usage: $0 <host>"
    exit 1
}

HOST="${1:-}"
if [[ -z "$HOST" ]]; then
    usage
fi

if [[ ! -d "$COMPONENT_DIR" ]]; then
    echo "Error: component directory not found at $COMPONENT_DIR"
    exit 1
fi

echo "Syncing $COMPONENT_DIR -> $REMOTE_USER@$HOST:$REMOTE_PATH"

# Stream a tar over ssh so we can filter out junk (compiled bytecode is the
# wrong arch on the target, .DS_Store etc.). cd into the parent and archive
# only the basename so the remote tree isn't recreated in full.
ssh "$REMOTE_USER@$HOST" "mkdir -p '$REMOTE_PATH'"
(cd "$REPO_ROOT/custom_components" \
  && tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' -cf - "$(basename "$COMPONENT_DIR")") \
  | ssh "$REMOTE_USER@$HOST" "tar -xf - -C '$REMOTE_PATH/..'"

ssh "$REMOTE_USER@$HOST" "find '$REMOTE_PATH' -type f"

echo "Done. Don't forget to restart Home Assistant (or reload the integration)."

