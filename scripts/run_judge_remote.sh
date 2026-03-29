#!/usr/bin/env bash
# Remote VetQwen judge scoring on a workstation running Ollama.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_common.sh"

latest_predictions_file() {
    if compgen -G "$PROJECT_ROOT/results/*_predictions.jsonl" > /dev/null; then
        ls -1t "$PROJECT_ROOT"/results/*_predictions.jsonl | head -n 1
        return 0
    fi
    return 1
}

derive_run_name() {
    local predictions_path="$1"
    local base_name
    base_name="$(basename -- "$predictions_path")"
    echo "${base_name%_predictions.jsonl}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Remote VetQwen judge scoring on a workstation running Ollama.

Usage:
  ./scripts/run_judge_remote.sh
  ./scripts/run_judge_remote.sh --predictions results/vetqwen_r16_full_predictions.jsonl
  ./scripts/run_judge_remote.sh --predictions results/vetqwen_r16_full_predictions.jsonl --sample 50 --seed 42

Defaults:
  - Uses the newest local results/*_predictions.jsonl when --predictions is omitted
  - Derives --run-name from the predictions filename when omitted
  - Uses the remote Ollama URL unless overridden
  - Uses scripts/run_judge.py default judge model unless VETQWEN_REMOTE_JUDGE_MODEL is set
  - Requires uv to be installed on the remote host

Environment overrides:
  VETQWEN_REMOTE_HOST
  VETQWEN_REMOTE_DIR
  VETQWEN_REMOTE_PYTHON
  VETQWEN_REMOTE_OLLAMA_URL
  VETQWEN_REMOTE_JUDGE_MODEL
EOF
    exit 0
fi

judge_args=()

if ! has_arg "--predictions" "$@"; then
    if latest_predictions="$(latest_predictions_file)"; then
        judge_args+=(--predictions "results/$(basename -- "$latest_predictions")")
    else
        echo "No local results/*_predictions.jsonl file found." >&2
        echo "Run evaluation first or pass --predictions explicitly." >&2
        exit 1
    fi
fi

if ! has_arg "--run-name" "$@"; then
    predictions_value="$(arg_value "--predictions" "$@" || true)"
    if [[ -n "$predictions_value" ]]; then
        judge_args+=(--run-name "$(derive_run_name "$predictions_value")")
    elif [[ -n "${latest_predictions:-}" ]]; then
        judge_args+=(--run-name "$(derive_run_name "$latest_predictions")")
    fi
fi

if ! has_arg "--ollama-url" "$@"; then
    judge_args+=(--ollama-url "$REMOTE_OLLAMA_URL")
fi

if [[ -n "$REMOTE_JUDGE_MODEL" ]] && ! has_arg "--model" "$@"; then
    judge_args+=(--model "$REMOTE_JUDGE_MODEL")
fi

judge_args+=("$@")

predictions_arg="$(arg_value "--predictions" "${judge_args[@]}" || true)"
if [[ -z "$predictions_arg" ]]; then
    echo "Unable to resolve predictions path." >&2
    exit 1
fi

local_predictions_path="$PROJECT_ROOT/${predictions_arg#./}"
if [[ ! -f "$local_predictions_path" ]]; then
    echo "Predictions file not found locally: $local_predictions_path" >&2
    exit 1
fi

remote_print_context "VetQwen remote judge"
echo "Predictions: $predictions_arg"
echo "Ollama URL:  $REMOTE_OLLAMA_URL"
if [[ -n "$REMOTE_JUDGE_MODEL" ]]; then
    echo "Judge model: $REMOTE_JUDGE_MODEL"
fi
remote_ensure_project_dir
remote_sync_project \
    --exclude 'adapter' \
    --exclude 'checkpoints' \
    --exclude 'data/raw'
remote_bootstrap_env remote --ollama-url "$REMOTE_OLLAMA_URL"

echo ""
echo "=== Starting judge scoring on $REMOTE_HOST ==="
remote_run_python_script "scripts/run_judge.py" "${judge_args[@]}"

echo ""
echo "=== Pulling judge results back ==="
mkdir -p "$PROJECT_ROOT/results"
remote_pull_dir "results" "results"

echo ""
echo "=== Done ==="
echo "Local results:"
ls -la "$PROJECT_ROOT/results" || true
