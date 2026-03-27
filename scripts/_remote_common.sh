#!/usr/bin/env bash

REMOTE_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$REMOTE_COMMON_DIR/.." && pwd)"

REMOTE_HOST="${VETQWEN_REMOTE_HOST:-blackbox}"
REMOTE_DIR="${VETQWEN_REMOTE_DIR:-~/vetqwen}"
REMOTE_PYTHON="${VETQWEN_REMOTE_PYTHON:-python3.11}"
REMOTE_TORCH_VERSION="${VETQWEN_REMOTE_TORCH_VERSION:-2.4.1}"
REMOTE_TORCH_INDEX_URL="${VETQWEN_REMOTE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
REMOTE_OLLAMA_URL="${VETQWEN_REMOTE_OLLAMA_URL:-http://127.0.0.1:11434}"
REMOTE_JUDGE_MODEL="${VETQWEN_REMOTE_JUDGE_MODEL:-}"
DEFAULT_BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"

has_arg() {
    local flag="$1"
    shift || true
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$flag" || "$arg" == "$flag="* ]]; then
            return 0
        fi
    done
    return 1
}

arg_value() {
    local flag="$1"
    shift || true
    local previous=""
    local arg
    for arg in "$@"; do
        if [[ "$previous" == "$flag" ]]; then
            printf '%s' "$arg"
            return 0
        fi
        if [[ "$arg" == "$flag="* ]]; then
            printf '%s' "${arg#"$flag="}"
            return 0
        fi
        previous="$arg"
    done
    return 1
}

quote_args() {
    local quoted=""
    local item
    for item in "$@"; do
        quoted+=" $(printf '%q' "$item")"
    done
    printf '%s' "$quoted"
}

remote_print_context() {
    local title="$1"
    echo "=== $title ==="
    echo "Remote host: $REMOTE_HOST"
    echo "Remote dir:  $REMOTE_DIR"
    echo "Python:      $REMOTE_PYTHON"
}

remote_ensure_project_dir() {
    echo ""
    echo "=== Ensuring remote project directory exists ==="
    ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"
}

remote_sync_project() {
    echo ""
    echo "=== Syncing project files to $REMOTE_HOST ==="
    rsync -avz \
        --exclude '.git' \
        --exclude '.venv' \
        --exclude '.ruff_cache' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude 'sessions' \
        "$@" \
        "$PROJECT_ROOT/" "$REMOTE_HOST:$REMOTE_DIR/"
}

remote_bootstrap_env() {
    local profile="${1:-remote}"
    shift || true
    local preflight_args
    preflight_args="$(quote_args "$@")"

    echo ""
    echo "=== Bootstrapping remote environment ==="
    ssh "$REMOTE_HOST" "bash -lc '
set -euo pipefail
cd $REMOTE_DIR

if [ ! -d .venv ]; then
    echo Creating remote virtualenv with $REMOTE_PYTHON
    $REMOTE_PYTHON -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip

if ! python -c \"import torch\" >/dev/null 2>&1; then
    python -m pip install torch==$REMOTE_TORCH_VERSION --index-url $REMOTE_TORCH_INDEX_URL
fi

python -m pip install -r requirements.txt
python scripts/preflight.py --profile $profile$preflight_args
'"
}

remote_run_python_script() {
    local script_path="$1"
    shift
    local remote_args
    remote_args="$(quote_args "$@")"
    ssh "$REMOTE_HOST" "bash -lc '
set -euo pipefail
cd $REMOTE_DIR
source .venv/bin/activate
python $script_path$remote_args
'"
}

remote_run_shell() {
    local command="$1"
    ssh "$REMOTE_HOST" "bash -lc '
set -euo pipefail
cd $REMOTE_DIR
source .venv/bin/activate
$command
'"
}

remote_pull_dir() {
    local remote_path="$1"
    local local_path="${2:-$1}"
    mkdir -p "$PROJECT_ROOT/$local_path"
    if ssh "$REMOTE_HOST" "test -d $REMOTE_DIR/$remote_path"; then
        rsync -avz "$REMOTE_HOST:$REMOTE_DIR/$remote_path/" "$PROJECT_ROOT/$local_path/"
    fi
}

remote_pull_file() {
    local remote_path="$1"
    local local_path="${2:-$1}"
    mkdir -p "$PROJECT_ROOT/$(dirname "$local_path")"
    if ssh "$REMOTE_HOST" "test -f $REMOTE_DIR/$remote_path"; then
        rsync -avz "$REMOTE_HOST:$REMOTE_DIR/$remote_path" "$PROJECT_ROOT/$local_path"
    fi
}
