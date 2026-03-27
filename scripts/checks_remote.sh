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

Environment overrides:
  VETQWEN_REMOTE_HOST
  VETQWEN_REMOTE_DIR
  VETQWEN_REMOTE_PYTHON
  VETQWEN_REMOTE_TORCH_VERSION
  VETQWEN_REMOTE_TORCH_INDEX_URL
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
remote_bootstrap_env remote --skip-ollama

echo ""
echo "=== Running checks on $REMOTE_HOST ==="
remote_run_shell "
python -m py_compile scripts/*.py tests/*.py app/*.py
python -m unittest discover -s tests -v
python scripts/build_dataset.py --help >/dev/null
python scripts/generate_synthetic.py --help >/dev/null
python scripts/train.py --help >/dev/null
python scripts/evaluate.py --help >/dev/null
python scripts/run_judge.py --help >/dev/null
python scripts/preflight.py --help >/dev/null
python scripts/build_review_subset.py --help >/dev/null
python scripts/compare_results.py --help >/dev/null
python app/gradio_demo.py --help >/dev/null
bash -n scripts/*.sh
"

echo ""
echo "=== Done ==="
echo "Remote checks completed successfully."
