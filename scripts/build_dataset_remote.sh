#!/usr/bin/env bash
# Remote VetQwen dataset build on the workstation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Remote VetQwen dataset build on the workstation.

Usage:
  ./scripts/build_dataset_remote.sh
  ./scripts/build_dataset_remote.sh --synthetic data/raw/synthetic.jsonl
  ./scripts/build_dataset_remote.sh --output-dir data/processed

Defaults:
  - Uses --output-dir data/processed when omitted
  - Pulls the generated split JSONL files and duplicate_audit.json back automatically
  - Requires uv to be installed on the remote host

Environment overrides:
  VETQWEN_REMOTE_HOST
  VETQWEN_REMOTE_DIR
  VETQWEN_REMOTE_PYTHON
EOF
    exit 0
fi

build_args=()
if ! has_arg "--output-dir" "$@"; then
    build_args+=(--output-dir "data/processed")
fi
build_args+=("$@")

output_dir="$(arg_value "--output-dir" "${build_args[@]}")"
if [[ -z "$output_dir" ]]; then
    echo "Unable to resolve output directory." >&2
    exit 1
fi

remote_print_context "VetQwen remote dataset build"
echo "Output dir:   $output_dir"
remote_ensure_project_dir
remote_sync_project \
    --exclude 'adapter' \
    --exclude 'checkpoints' \
    --exclude 'results'
remote_bootstrap_env remote --skip-ollama

echo ""
echo "=== Building dataset on $REMOTE_HOST ==="
remote_run_python_script "scripts/build_dataset.py" "${build_args[@]}"

echo ""
echo "=== Pulling processed dataset back ==="
remote_pull_dir "$output_dir" "$output_dir"

echo ""
echo "=== Done ==="
ls -la "$PROJECT_ROOT/$output_dir" || true
