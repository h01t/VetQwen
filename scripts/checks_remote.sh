#!/usr/bin/env bash
# Remote VetQwen verification checks on the workstation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_remote_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Remote VetQwen checks on the workstation.

Runs:
  - python -m py_compile for scripts/tests/app
  - python -m unittest discover -s tests -v
  - --help smoke checks for low-level Python entrypoints
  - bash -n for shell scripts

Usage:
  ./scripts/checks_remote.sh
  ./scripts/checks_remote.sh --with-ollama
  Requires uv to be installed on the remote host.

Environment overrides:
  VETQWEN_REMOTE_HOST
  VETQWEN_REMOTE_DIR
  VETQWEN_REMOTE_PYTHON
EOF
    exit 0
fi

remote_print_context "VetQwen remote checks"
remote_ensure_project_dir
remote_sync_project \
    --exclude 'adapter' \
    --exclude 'checkpoints' \
    --exclude 'results' \
    --exclude 'data/raw' \
    --exclude 'data/processed'

preflight_args=(remote --skip-ollama)
if has_arg "--with-ollama" "$@"; then
    preflight_args=(remote)
fi
remote_bootstrap_env "${preflight_args[@]}"

echo ""
echo "=== Running checks on $REMOTE_HOST ==="
remote_run_shell "
uv run --no-sync --group research --python \"\$VETQWEN_REMOTE_RESOLVED_PYTHON\" python -m py_compile scripts/*.py tests/*.py app/*.py
uv run --no-sync --group research --python \"\$VETQWEN_REMOTE_RESOLVED_PYTHON\" python -m unittest discover -s tests -v
uv run --no-sync --group research --python \"\$VETQWEN_REMOTE_RESOLVED_PYTHON\" python scripts/build_dataset.py --help >/dev/null
uv run --no-sync --group research --python \"\$VETQWEN_REMOTE_RESOLVED_PYTHON\" python scripts/generate_synthetic.py --help >/dev/null
uv run --no-sync --group research --python \"\$VETQWEN_REMOTE_RESOLVED_PYTHON\" python scripts/train.py --help >/dev/null
uv run --no-sync --group research --python \"\$VETQWEN_REMOTE_RESOLVED_PYTHON\" python scripts/evaluate.py --help >/dev/null
uv run --no-sync --group research --python \"\$VETQWEN_REMOTE_RESOLVED_PYTHON\" python scripts/run_judge.py --help >/dev/null
uv run --no-sync --group research --python \"\$VETQWEN_REMOTE_RESOLVED_PYTHON\" python scripts/preflight.py --help >/dev/null
uv run --no-sync --group research --python \"\$VETQWEN_REMOTE_RESOLVED_PYTHON\" python scripts/build_review_subset.py --help >/dev/null
uv run --no-sync --group research --python \"\$VETQWEN_REMOTE_RESOLVED_PYTHON\" python scripts/compare_results.py --help >/dev/null
uv run --no-sync --group demo --python \"\$VETQWEN_REMOTE_RESOLVED_PYTHON\" python app/gradio_demo.py --help >/dev/null
bash -n scripts/*.sh
"

echo ""
echo "=== Done ==="
echo "Remote checks completed successfully."
