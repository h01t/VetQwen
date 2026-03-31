# Implementation Plan

[Overview]
Migrate VetQwen to a complete uv-managed workflow (including Python runtime management) and remove operational dependency on pip/system-Python assumptions in remote wrappers.

The project already uses `pyproject.toml` and `uv.lock` as canonical dependency sources, but remote orchestration still enforced `--no-managed-python` and assumed a usable system interpreter (`python3.11/3.12/3`). This created friction on hosts where system Python is outside the pinned research target and conflicted with the target end-state of a fully uv-managed stack.

This implementation shifts remote wrappers from system-Python detection to uv-managed interpreter pinning, standardizes docs around the new policy, retains compatibility exports only as non-authoritative artifacts, and preserves reproducibility by explicitly pinning the runtime used by research jobs.

[Types]
No domain data-model changes are required; this is operational/runtime-policy work.

Shell/runtime contract changes:
- `VETQWEN_REMOTE_PYTHON` semantics shifted toward uv runtime selection with pinned default.
- Remote bootstrap contract shifted from system-resolution + `--no-managed-python` to direct uv runtime use.
- Invalid override diagnostics must preserve the exact requested value.

Validation rules:
- Research runtime default pinned to Python 3.11 unless overridden.
- Demo/runtime commands remain lockfile-driven.
- pip is not the canonical installation path.

[Files]
- Modified:
  - `scripts/_remote_common.sh` (active migration target)
- Inspected:
  - `README.md`
  - `pyproject.toml`
  - `scripts/checks_remote.sh`
  - `scripts/train_remote.sh`
  - `scripts/vetqwen_core/inference.py`
  - `scripts/build_dataset.py`

[Functions]
In `scripts/_remote_common.sh`:
- `remote_print_context`
  - Updated to print uv-managed policy context.
- `remote_bootstrap_env`
  - Migrated to uv-managed runtime sync/run behavior with `--python "$requested_python"` and no `--no-managed-python`.
- `remote_run_python_script`
  - Migrated to run directly with requested runtime via `uv run --python`.
- `remote_run_shell`
  - Migrated to export runtime directly without system resolver dependency.
- Invalid override diagnostic token rendering was fixed and verified in test runs.

[Classes]
No class changes.

[Dependencies]
Canonical remains:
- `pyproject.toml`
- `uv.lock`

Compatibility exports still exist:
- `requirements.txt`
- `requirements-demo.txt`

These remain non-authoritative and still need final wording reconciliation in docs/scripts.

[Testing]
Executed and verified:

1. Local regression:
- `uv run --no-sync --group research python -m unittest discover -s tests -v`
- Result: 20/20 tests pass.

2. Remote checks regression (green path):
- `VETQWEN_REMOTE_HOST=blackbox bash scripts/checks_remote.sh | cat`
- Result: pass (preflight + remote tests + wrapper checks).

3. Invalid override diagnostic validation:
- `VETQWEN_REMOTE_HOST=blackbox VETQWEN_REMOTE_PYTHON=python3.13 bash scripts/checks_remote.sh | cat`
- Result: error now preserves token:
  - `Requested Python python3.13 was not found ...`

4. Post-migration remote checks (uv-managed flow):
- `VETQWEN_REMOTE_HOST=blackbox bash scripts/checks_remote.sh | cat`
- Result: pass after runtime-flow edits in `_remote_common.sh`.

[Implementation Order]
1. Baseline local/remote checks captured. ✅
2. Refactor `_remote_common.sh` for uv-managed runtime policy. ✅ (in progress refinement)
3. Verify diagnostic + green-path checks. ✅
4. Propagate wrapper help/runtime policy consistency. ⏳
5. Run remote smoke workflows (synthetic + dataset build) under migrated policy. ⏳
6. Update README/docs for complete uv migration messaging. ⏳
7. Reconcile compatibility export policy references. ⏳
8. Final full regression sweep + readiness summary. ⏳

task_progress Items:
- [x] Step 1: Baseline lock-in and command snapshot verification for local+remote green state
- [x] Step 2: Implement uv-managed runtime core in `scripts/_remote_common.sh` (remove system-only/`--no-managed-python` policy)
- [x] Step 3: Verify wrapper diagnostics and green path (invalid override token + normal remote checks)
- [ ] Step 4: Update all remote wrapper scripts/help text to match uv-managed runtime policy
- [ ] Step 5: Run remote smoke workflows (synthetic + dataset build) and verify artifact pullback under migrated policy
- [ ] Step 6: Update README and related docs to complete “pip → uv” migration messaging
- [ ] Step 7: Reconcile compatibility exports policy (`requirements*.txt`) and references
- [ ] Step 8: Final local+remote regression sweep and migration readiness summary

[Current Status Notes]
- The major blocker from earlier (missing token in invalid-override diagnostics) is fixed.
- Remote checks now pass with uv-managed bootstrap path.
- Remaining work is primarily consistency cleanup across docs/help text and full end-to-end smoke coverage for all wrappers after migration.
