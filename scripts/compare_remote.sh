#!/usr/bin/env bash
# Remote VetQwen result comparison on the workstation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Remote VetQwen comparison pass on the workstation.

Usage:
  ./scripts/compare_remote.sh \
    --baseline results/baseline.json \
    --candidate results/candidate.json

  ./scripts/compare_remote.sh \
    --baseline results/baseline.json \
    --candidate results/candidate.json \
    --baseline-judge results/baseline_judge.json \
    --candidate-judge results/candidate_judge.json \
    --output results/comparisons/candidate_vs_baseline.md

Defaults:
  - Derives --output when omitted
  - Pulls the markdown summary and JSON sidecar back automatically
  - Requires uv to be installed on the remote host

Environment overrides:
  VETQWEN_REMOTE_HOST
  VETQWEN_REMOTE_DIR
  VETQWEN_REMOTE_PYTHON
EOF
    exit 0
fi

if ! has_arg "--baseline" "$@"; then
    echo "compare_remote.sh requires --baseline" >&2
    exit 1
fi
if ! has_arg "--candidate" "$@"; then
    echo "compare_remote.sh requires --candidate" >&2
    exit 1
fi

compare_args=("$@")
output_path="$(arg_value "--output" "${compare_args[@]}" || true)"
if [[ -z "$output_path" ]]; then
    baseline_path="$(arg_value "--baseline" "${compare_args[@]}")"
    candidate_path="$(arg_value "--candidate" "${compare_args[@]}")"
    baseline_name="$(basename -- "${baseline_path%.*}")"
    candidate_name="$(basename -- "${candidate_path%.*}")"
    output_path="results/comparisons/${candidate_name}_vs_${baseline_name}.md"
    compare_args+=(--output "$output_path")
fi

json_output_path="${output_path%.md}.json"

remote_print_context "VetQwen remote comparison"
echo "Output path:  $output_path"
remote_ensure_project_dir
remote_sync_project \
    --exclude 'adapter' \
    --exclude 'checkpoints' \
    --exclude 'data/raw' \
    --exclude 'data/processed'
remote_bootstrap_env remote --skip-ollama

echo ""
echo "=== Comparing results on $REMOTE_HOST ==="
remote_run_python_script "scripts/compare_results.py" "${compare_args[@]}"

echo ""
echo "=== Pulling comparison summary back ==="
remote_pull_file "$output_path" "$output_path"
remote_pull_file "$json_output_path" "$json_output_path"

echo ""
echo "=== Done ==="
ls -la "$PROJECT_ROOT/$(dirname "$output_path")" || true
