#!/usr/bin/env bash

REMOTE_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$REMOTE_COMMON_DIR/.." && pwd)"

REMOTE_HOST="${VETQWEN_REMOTE_HOST:-blackbox}"
REMOTE_DIR="${VETQWEN_REMOTE_DIR:-~/Dev/vetqwen}"
REMOTE_PYTHON="${VETQWEN_REMOTE_PYTHON:-3.11}"
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
    echo "Python:      $REMOTE_PYTHON (uv-managed allowed)"
}

remote_shell_quote() {
    printf '%q' "$1"
}

remote_ensure_project_dir() {
    echo ""
    echo "=== Ensuring remote project directory exists ==="
    local remote_dir_quoted
    remote_dir_quoted="$(remote_shell_quote "$REMOTE_DIR")"
    ssh "$REMOTE_HOST" "mkdir -p $remote_dir_quoted"
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
    local dependency_group="research"
    if [[ "$profile" == "demo" ]]; then
        dependency_group="demo"
    fi
    local remote_dir_quoted host_quoted py_quoted profile_quoted dep_group_quoted
    remote_dir_quoted="$(remote_shell_quote "$REMOTE_DIR")"
    host_quoted="$REMOTE_HOST"
    py_quoted="$REMOTE_PYTHON"
    profile_quoted="$profile"
    dep_group_quoted="$dependency_group"

    echo ""
    echo "=== Bootstrapping remote environment ==="
    ssh "$REMOTE_HOST" "bash -lc '
set -euo pipefail
cd $remote_dir_quoted

if ! command -v uv >/dev/null 2>&1; then
    echo \"uv is required on $host_quoted but was not found on PATH.\" >&2
    echo \"Install uv on the workstation first, then rerun this wrapper.\" >&2
    exit 1
fi

requested_python=\"$py_quoted\"

echo "Syncing uv environment with Python: \$requested_python"
uv sync --group $dep_group_quoted --locked --python "\$requested_python"
uv run --no-sync --group $dep_group_quoted --python "\$requested_python" python scripts/preflight.py --profile $profile_quoted$preflight_args
'"
}

remote_run_python_script() {
    local script_path="$1"
    shift
    local remote_args
    remote_args="$(quote_args "$@")"
    local remote_dir_quoted py_quoted script_quoted
    remote_dir_quoted="$(remote_shell_quote "$REMOTE_DIR")"
    py_quoted="$(remote_shell_quote "$REMOTE_PYTHON")"
    script_quoted="$(remote_shell_quote "$script_path")"

    ssh "$REMOTE_HOST" "bash -lc '
set -euo pipefail
cd $remote_dir_quoted
requested_python=$py_quoted
uv run --no-sync --group research --python \"\$requested_python\" python $script_quoted$remote_args
'"
}

remote_run_shell() {
    local command="$1"
    local remote_dir_quoted py_quoted
    remote_dir_quoted="$(remote_shell_quote "$REMOTE_DIR")"
    py_quoted="$(remote_shell_quote "$REMOTE_PYTHON")"
    ssh "$REMOTE_HOST" "bash -lc '
set -euo pipefail
cd $remote_dir_quoted
export VETQWEN_REMOTE_RESOLVED_PYTHON=$py_quoted
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
