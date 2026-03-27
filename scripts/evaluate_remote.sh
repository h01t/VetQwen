#!/usr/bin/env bash
# Remote VetQwen evaluation on a CUDA workstation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Remote VetQwen evaluation on a CUDA workstation.

Usage:
  ./scripts/evaluate_remote.sh
  ./scripts/evaluate_remote.sh --model Qwen/Qwen2.5-3B-Instruct --run-name baseline_3b
  ./scripts/evaluate_remote.sh --model ./adapter --base-model Qwen/Qwen2.5-3B-Instruct --run-name vetqwen_r16_full

Defaults:
  - Uses ./adapter when a local adapter exists and --model is omitted
  - Otherwise uses Qwen/Qwen2.5-3B-Instruct
  - Defaults to --split test
  - Defaults to --no-judge unless judge-related flags are passed

Environment overrides:
  VETQWEN_REMOTE_HOST
  VETQWEN_REMOTE_DIR
  VETQWEN_REMOTE_PYTHON
  VETQWEN_REMOTE_TORCH_VERSION
  VETQWEN_REMOTE_TORCH_INDEX_URL
  VETQWEN_REMOTE_OLLAMA_URL
  VETQWEN_REMOTE_JUDGE_MODEL
EOF
    exit 0
fi

local_adapter_ready=0
if [[ -d "$PROJECT_ROOT/adapter" ]]; then
    if find "$PROJECT_ROOT/adapter" -mindepth 1 -type f ! -name '.gitkeep' | grep -q .; then
        local_adapter_ready=1
    fi
fi

eval_args=()
if ! has_arg "--model" "$@"; then
    if [[ "$local_adapter_ready" -eq 1 ]]; then
        eval_args+=(--model "./adapter")
        if ! has_arg "--base-model" "$@"; then
            eval_args+=(--base-model "$DEFAULT_BASE_MODEL")
        fi
    else
        eval_args+=(--model "$DEFAULT_BASE_MODEL")
    fi
fi

if ! has_arg "--split" "$@"; then
    eval_args+=(--split "test")
fi

if ! has_arg "--run-name" "$@"; then
    if [[ "$local_adapter_ready" -eq 1 ]]; then
        eval_args+=(--run-name "remote_adapter_eval")
    else
        eval_args+=(--run-name "remote_baseline_eval")
    fi
fi

judge_enabled=1
if ! has_arg "--no-judge" "$@" \
    && ! has_arg "--judge-sample" "$@" \
    && ! has_arg "--judge-model" "$@" \
    && ! has_arg "--ollama-url" "$@"; then
    eval_args+=(--no-judge)
    judge_enabled=0
fi

if [[ "$judge_enabled" -eq 1 ]] && ! has_arg "--ollama-url" "$@"; then
    eval_args+=(--ollama-url "$REMOTE_OLLAMA_URL")
fi

if [[ -n "$REMOTE_JUDGE_MODEL" ]] && ! has_arg "--judge-model" "$@" && [[ "$judge_enabled" -eq 1 ]]; then
    eval_args+=(--judge-model "$REMOTE_JUDGE_MODEL")
fi

eval_args+=("$@")

remote_print_context "VetQwen remote evaluation"
remote_ensure_project_dir
remote_sync_project \
    --exclude 'checkpoints' \
    --exclude 'results' \
    --exclude 'data/raw'
if [[ "$judge_enabled" -eq 1 ]]; then
    remote_bootstrap_env remote --ollama-url "$REMOTE_OLLAMA_URL"
else
    remote_bootstrap_env remote --skip-ollama
fi

echo ""
echo "=== Starting evaluation on $REMOTE_HOST ==="
remote_run_python_script "scripts/evaluate.py" "${eval_args[@]}"

echo ""
echo "=== Pulling evaluation results back ==="
mkdir -p "$PROJECT_ROOT/results"
remote_pull_dir "results" "results"

echo ""
echo "=== Done ==="
echo "Local results:"
ls -la "$PROJECT_ROOT/results" || true
