# Implementation Plan

[Overview]
Migrate VetQwen to a complete uv-managed workflow (including Python runtime management) and remove operational dependency on pip/system-Python assumptions in remote wrappers.

The project uses `pyproject.toml` and `uv.lock` as canonical dependency sources. Migration work has focused on removing system-Python assumptions in remote wrappers, standardizing runtime policy to uv-managed Python pinning, and validating behavior through local and remote execution paths.

[Types]
No domain data-model changes are required; this is operational/runtime-policy work plus demo runtime compatibility hardening (MPS support on MacBook Apple Silicon).

Runtime contract (current):
- `VETQWEN_REMOTE_PYTHON` targets uv-managed runtime selection (default pinned runtime behavior).
- Remote bootstrap/run paths use uv runtime selection directly.
- Invalid override diagnostics preserve exact requested token.

Validation policy:
- Research runtime default pinned to Python 3.11 unless overridden.
- Demo/runtime commands remain lockfile-driven.
- pip is not canonical.

[Files]
Modified:
- `scripts/_remote_common.sh`
- `README.md`
- `requirements.txt`
- `requirements-demo.txt`
- `app/gradio_demo.py`
- `scripts/vetqwen_core/inference.py`
- `implementation_plan.md`

Inspected:
- `pyproject.toml`
- `scripts/checks_remote.sh`
- `scripts/train_remote.sh`
- `scripts/build_dataset.py`

[Functions]
`scripts/_remote_common.sh`:
- `remote_print_context`: updated to uv-managed policy wording.
- `remote_bootstrap_env`: migrated to uv-managed `uv sync/run --python ...` flow.
- `remote_run_python_script`: aligned with uv runtime policy.
- `remote_run_shell`: aligned with uv runtime policy.
- Invalid override diagnostic rendering fixed and validated.

`scripts/vetqwen_core/inference.py`:
- `resolve_device` now supports `mps` and auto fallback order:
  - CUDA → MPS → CPU.
- Explicit `--device mps` validates MPS availability with clear error.
- Non-CUDA dtype now:
  - MPS: `torch.float16`
  - CPU: `torch.float32`

`app/gradio_demo.py`:
- CLI `--device` choices expanded to include `mps`.
- Help text updated for MacBook Apple Silicon usage guidance.

[Classes]
No class changes.

[Dependencies]
Canonical source of truth:
- `pyproject.toml`
- `uv.lock`

Compatibility exports retained (non-authoritative):
- `requirements.txt`
- `requirements-demo.txt`

Reconciled wording:
- removed stale `--no-managed-python` usage hints from compatibility comments.

[Testing]
Executed and verified:

1) Local regression:
- `uv run --no-sync --group research python -m unittest discover -s tests -v`
- Result: PASS (20/20).

2) Remote checks (uv-managed path):
- `VETQWEN_REMOTE_HOST=blackbox bash scripts/checks_remote.sh`
- Result: PASS.

3) Invalid override diagnostics:
- `VETQWEN_REMOTE_HOST=blackbox VETQWEN_REMOTE_PYTHON=python3.13 bash scripts/checks_remote.sh`
- Result: PASS (error preserves token: `Requested Python python3.13 ...`).

4) Remote synthetic smoke:
- `VETQWEN_REMOTE_HOST=blackbox bash scripts/generate_synthetic_remote.sh --n 10 --output data/raw/smoke_uv_migration.jsonl`
- Result: PASS, generated + pulled back artifact.

5) Remote dataset-build smoke:
- `VETQWEN_REMOTE_HOST=blackbox bash scripts/build_dataset_remote.sh --synthetic data/raw/smoke_uv_migration.jsonl`
- Result: PASS, processed artifacts pulled back.

6) Wrapper help/smoke:
- `scripts/train_remote.sh --help`
- `scripts/evaluate_remote.sh --help`
- `scripts/run_judge_remote.sh --help`
- `scripts/compare_remote.sh --help`
- Result: PASS.

7) Demo-side checks:
- `uv run --no-sync --group demo python scripts/preflight.py --profile demo --skip-ollama`
- `uv run --no-sync --group demo python app/gradio_demo.py --help`
- Result: PASS, includes `--device {auto,cpu,cuda,mps}`.

8) Post-MPS-change regression:
- Re-ran local unit tests + demo help.
- Result: PASS.

Observed issue from testing:
- Dataset build logs still emit repeated `trust_remote_code is not supported anymore` warnings for HF datasets, while pipeline still completes successfully. This is a follow-up cleanup candidate in `scripts/build_dataset.py`.

[Implementation Order]
1. Baseline checks captured. ✅
2. Runtime core refactor in `_remote_common.sh`. ✅
3. Diagnostic + green-path verification. ✅
4. Wrapper help/runtime consistency checks. ✅
5. Remote smoke workflows (synthetic + dataset build). ✅
6. README/doc migration messaging updates. ✅ (primary lines updated)
7. Compatibility export reference reconciliation. ✅ (requirements comments updated)
8. Final regression sweep + readiness summary. ⏳ (partially complete; heavy-path remote train/eval/judge/compare execution still pending)

task_progress Items:
- [x] Step 1: Baseline lock-in and command snapshot verification for local+remote green state
- [x] Step 2: Implement uv-managed runtime core in `scripts/_remote_common.sh` (remove system-only/`--no-managed-python` policy)
- [x] Step 3: Verify wrapper diagnostics and green path (invalid override token + normal remote checks)
- [x] Step 4: Update all remote wrapper scripts/help text to match uv-managed runtime policy
- [x] Step 5: Run remote smoke workflows (synthetic + dataset build) and verify artifact pullback under migrated policy
- [x] Step 6: Update README and related docs to complete “pip → uv” migration messaging
- [x] Step 7: Reconcile compatibility exports policy (`requirements*.txt`) and references
- [ ] Step 8: Final local+remote regression sweep and migration readiness summary (remaining: heavy-path remote train/eval/judge/compare runs)

[Current Status Notes]
- Major migration blocker (invalid override token loss) is fixed.
- uv-managed runtime path is validated on local and remote checks/smoke workflows.
- Mac demo now supports MPS explicitly and reports proper CLI options.
- Remaining work is full heavy-path remote execution (train/eval/judge/compare) and final closeout summary.
