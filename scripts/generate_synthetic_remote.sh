#!/usr/bin/env bash
# Remote VetQwen synthetic data generation through Ollama on the workstation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Remote VetQwen synthetic livestock generation on the workstation.

Usage:
  ./scripts/generate_synthetic_remote.sh
  ./scripts/generate_synthetic_remote.sh --n 400 --output data/raw/synthetic.jsonl
  ./scripts/generate_synthetic_remote.sh --n 100 --species Cattle --output data/raw/synthetic_cattle.jsonl

Defaults:
  - Uses --output data/raw/synthetic.jsonl when omitted
  - Uses the remote Ollama URL from VETQWEN_REMOTE_OLLAMA_URL unless overridden

Environment overrides:
  VETQWEN_REMOTE_HOST
  VETQWEN_REMOTE_DIR
  VETQWEN_REMOTE_PYTHON
  VETQWEN_REMOTE_TORCH_VERSION
  VETQWEN_REMOTE_TORCH_INDEX_URL
  VETQWEN_REMOTE_OLLAMA_URL
EOF
    exit 0
fi

synthetic_args=()
if ! has_arg "--output" "$@"; then
    synthetic_args+=(--output "data/raw/synthetic.jsonl")
fi
if ! has_arg "--ollama-url" "$@"; then
    synthetic_args+=(--ollama-url "$REMOTE_OLLAMA_URL")
fi
synthetic_args+=("$@")

output_path="$(arg_value "--output" "${synthetic_args[@]}")"
if [[ -z "$output_path" ]]; then
    echo "Unable to resolve output path." >&2
    exit 1
fi

remote_print_context "VetQwen remote synthetic generation"
echo "Output path:  $output_path"
echo "Ollama URL:   $REMOTE_OLLAMA_URL"
remote_ensure_project_dir
remote_sync_project \
    --exclude 'adapter' \
    --exclude 'checkpoints' \
    --exclude 'results' \
    --exclude 'data/processed'
remote_bootstrap_env remote --ollama-url "$REMOTE_OLLAMA_URL"

echo ""
echo "=== Starting synthetic generation on $REMOTE_HOST ==="
remote_run_python_script "scripts/generate_synthetic.py" "${synthetic_args[@]}"

echo ""
echo "=== Pulling generated data back ==="
remote_pull_file "$output_path" "$output_path"

echo ""
echo "=== Done ==="
ls -la "$PROJECT_ROOT/$(dirname "$output_path")" || true
