#!/usr/bin/env bash
# Remote VetQwen training on a CUDA workstation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Remote VetQwen training on a CUDA workstation.

Usage:
  ./scripts/train_remote.sh
  ./scripts/train_remote.sh --run-name vetqwen_r16_trial
  VETQWEN_REMOTE_HOST=blackbox ./scripts/train_remote.sh --lora-r 32

Defaults:
  - Uses configs/train_default.yaml when --config is omitted

Environment overrides:
  VETQWEN_REMOTE_HOST
  VETQWEN_REMOTE_DIR
  VETQWEN_REMOTE_PYTHON
  VETQWEN_REMOTE_TORCH_VERSION
  VETQWEN_REMOTE_TORCH_INDEX_URL
EOF
    exit 0
fi

train_args=()
if ! has_arg "--config" "$@"; then
    train_args+=(--config "configs/train_default.yaml")
fi
train_args+=("$@")

remote_print_context "VetQwen remote training"
remote_ensure_project_dir
remote_sync_project \
    --exclude 'adapter' \
    --exclude 'checkpoints' \
    --exclude 'results' \
    --exclude 'data/raw'
remote_bootstrap_env remote --skip-ollama

echo ""
echo "=== Starting training on $REMOTE_HOST ==="
remote_run_python_script "scripts/train.py" "${train_args[@]}"

echo ""
echo "=== Pulling artifacts back ==="
mkdir -p "$PROJECT_ROOT/adapter" "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/results"
remote_pull_dir "adapter" "adapter"
remote_pull_dir "checkpoints" "checkpoints"
remote_pull_dir "results" "results"

echo ""
echo "=== Done ==="
echo "Local adapter:"
ls -la "$PROJECT_ROOT/adapter" || true
echo ""
echo "Local checkpoints:"
ls -la "$PROJECT_ROOT/checkpoints" || true
echo ""
echo "Local results:"
ls -la "$PROJECT_ROOT/results" || true
