#!/usr/bin/env bash

REMOTE_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$REMOTE_COMMON_DIR/.." && pwd)"

REMOTE_HOST="${VETQWEN_REMOTE_HOST:-blackbox}"
REMOTE_DIR="${VETQWEN_REMOTE_DIR:-~/Dev/vetqwen}"
REMOTE_PYTHON="${VETQWEN_REMOTE_PYTHON:-auto}"
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
    if [[ "$REMOTE_PYTHON" == "auto" ]]; then
        echo "Python:      auto (prefers python3.11, then python3.12, then python3)"
    else
        echo "Python:      $REMOTE_PYTHON"
    fi
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
    local dependency_group="research"
    if [[ "$profile" == "demo" ]]; then
        dependency_group="demo"
    fi

    echo ""
    echo "=== Bootstrapping remote environment ==="
    ssh "$REMOTE_HOST" "bash -lc '
set -euo pipefail
cd $REMOTE_DIR

if ! command -v uv >/dev/null 2>&1; then
    echo \"uv is required on $REMOTE_HOST but was not found on PATH.\" >&2
    echo \"Install uv on the workstation first, then rerun this wrapper.\" >&2
    exit 1
fi

requested_python=$REMOTE_PYTHON
resolved_python=\"\"
resolved_python_path=\"\"

resolve_system_python() {
    local candidate=\"\$1\"
    local candidate_path=\"\"
    local candidate_realpath=\"\"

    candidate_path=\"\$(command -v \"\$candidate\" 2>/dev/null || true)\"
    if [[ -z \"\$candidate_path\" ]]; then
        return 1
    fi

    if ! \"\$candidate_path\" --version >/dev/null 2>&1; then
        return 1
    fi

    candidate_realpath=\"\$(readlink -f \"\$candidate_path\" 2>/dev/null || printf '%s' \"\$candidate_path\")\"
    if [[ \"\$candidate_realpath\" == \"\$HOME/.local/share/uv/python/\"* ]]; then
        return 1
    fi

    resolved_python=\"\$candidate\"
    resolved_python_path=\"\$candidate_realpath\"
    return 0
}

if [[ \"\$requested_python\" == \"auto\" ]]; then
    for candidate in python3.11 python3.12 python3; do
        if resolve_system_python \"\$candidate\"; then
            break
        fi
    done
else
    if ! resolve_system_python \"\$requested_python\"; then
        requested_path=\"\$(command -v \"\$requested_python\" 2>/dev/null || true)\"
        requested_realpath=\"\$(readlink -f \"\$requested_path\" 2>/dev/null || printf '%s' \"\$requested_path\")\"
        if [[ -n \"\$requested_realpath\" && \"\$requested_realpath\" == \"\$HOME/.local/share/uv/python/\"* ]]; then
            echo \"Requested Python '\$requested_python' resolves to a uv-managed interpreter, which is excluded by --no-managed-python.\" >&2
            echo \"Set VETQWEN_REMOTE_PYTHON to a system interpreter such as python3.12.\" >&2
        fi
        echo \"Requested Python '\$requested_python' was not found as a usable system interpreter on $REMOTE_HOST.\" >&2
        exit 1
    fi
fi

if [[ -z \"\$resolved_python\" ]] || [[ -z \"\$resolved_python_path\" ]]; then
    echo \"No supported system Python was found on $REMOTE_HOST.\" >&2
    echo \"Set VETQWEN_REMOTE_PYTHON explicitly, or install python3.11/python3.12 on the workstation.\" >&2
    exit 1
fi

echo \"Using remote Python interpreter: \$resolved_python (\$resolved_python_path)\"
uv sync --group $dependency_group --locked --python \"\$resolved_python\" --no-managed-python
uv run --no-sync --group $dependency_group --python \"\$resolved_python\" python scripts/preflight.py --profile $profile$preflight_args
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

requested_python=$REMOTE_PYTHON
resolved_python=\"\"

resolve_system_python() {
    local candidate=\"\$1\"
    local candidate_path=\"\"
    local candidate_realpath=\"\"

    candidate_path=\"\$(command -v \"\$candidate\" 2>/dev/null || true)\"
    if [[ -z \"\$candidate_path\" ]]; then
        return 1
    fi

    if ! \"\$candidate_path\" --version >/dev/null 2>&1; then
        return 1
    fi

    candidate_realpath=\"\$(readlink -f \"\$candidate_path\" 2>/dev/null || printf '%s' \"\$candidate_path\")\"
    if [[ \"\$candidate_realpath\" == \"\$HOME/.local/share/uv/python/\"* ]]; then
        return 1
    fi

    resolved_python=\"\$candidate\"
    return 0
}

if [[ \"\$requested_python\" == \"auto\" ]]; then
    for candidate in python3.11 python3.12 python3; do
        if resolve_system_python \"\$candidate\"; then
            break
        fi
    done
else
    resolve_system_python \"\$requested_python\" || true
fi

if [[ -n \"\$resolved_python\" ]]; then
    uv run --no-sync --group research --python \"\$resolved_python\" python $script_path$remote_args
else
    uv run --no-sync --group research python $script_path$remote_args
fi
'"
}

remote_run_shell() {
    local command="$1"
    ssh "$REMOTE_HOST" "bash -lc '
set -euo pipefail
cd $REMOTE_DIR

requested_python=$REMOTE_PYTHON
resolved_python=\"\"

resolve_system_python() {
    local candidate=\"\$1\"
    local candidate_path=\"\"
    local candidate_realpath=\"\"

    candidate_path=\"\$(command -v \"\$candidate\" 2>/dev/null || true)\"
    if [[ -z \"\$candidate_path\" ]]; then
        return 1
    fi

    if ! \"\$candidate_path\" --version >/dev/null 2>&1; then
        return 1
    fi

    candidate_realpath=\"\$(readlink -f \"\$candidate_path\" 2>/dev/null || printf '%s' \"\$candidate_path\")\"
    if [[ \"\$candidate_realpath\" == \"\$HOME/.local/share/uv/python/\"* ]]; then
        return 1
    fi

    resolved_python=\"\$candidate\"
    return 0
}

if [[ \"\$requested_python\" == \"auto\" ]]; then
    for candidate in python3.11 python3.12 python3; do
        if resolve_system_python \"\$candidate\"; then
            break
        fi
    done
else
    resolve_system_python \"\$requested_python\" || true
fi

if [[ -z \"\$resolved_python\" ]]; then
    echo \"No supported system Python was found on $REMOTE_HOST.\" >&2
    echo \"Set VETQWEN_REMOTE_PYTHON explicitly, or install python3.11/python3.12 on the workstation.\" >&2
    exit 1
fi

export VETQWEN_REMOTE_RESOLVED_PYTHON=\"\$resolved_python\"
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
