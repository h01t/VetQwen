# Implementation Plan

[Overview]  
Stabilize the uv-based VetQwen workflow by hardening remote wrappers, removing deprecated dataset-loading behavior, improving repo/docs hygiene, and expanding validation so local+remote flows remain reliable.

This implementation targets the migration debt discovered during review and runtime tests: noisy Hugging Face loader warnings, shell wrapper fragility from interpolated command strings, duplicated Python resolver logic, malformed error diagnostics for invalid Python overrides, and documentation/repo hygiene drift. The goal is not to change model behavior, but to improve operational reliability and maintainability for the intended workflow: remote CUDA research/training on `blackbox` and local Mac demo usage.

Scope includes shell wrappers in `scripts/`, dataset loading in `scripts/build_dataset.py`, documentation cleanup in `README.md`, and quality gates/test additions in `tests/` and remote checks. The high-level approach is phased and verification-driven: refactor with minimal behavior drift, then validate each phase through targeted tests and smoke commands before moving on.

[Types]  
No new domain model types are required; changes are primarily shell/runtime behavior and CLI safety, with optional lightweight typed helper structures in Python tests.

If needed for testability, add typed fixtures in Python tests using existing patterns:
- `dict[str, Any]` dataset record fixtures
- deterministic expected outputs for wrapper argument/path handling contracts
Validation rules remain unchanged:
- split outputs: JSONL + duplicate audit JSON
- remote wrappers: fail fast on missing host/python/uv
- interpreter policy: system Python only when `--no-managed-python` is used

[Files]  
Implementation will modify shell wrappers, dataset loader usage, docs, and tests, plus add one planning artifact.

- **New files**
  - `implementation_plan.md` — this implementation plan (already created).

- **Existing files to modify**
  - `scripts/_remote_common.sh`
    - centralize Python resolution logic once
    - remove fragile interpolated command blobs where possible
    - improve quoting for remote directory and command execution
    - fix invalid-Python override diagnostics to always include the requested interpreter value (e.g., `python3.13`)
  - `scripts/build_dataset.py`
    - remove deprecated/unsupported `trust_remote_code` usage path for HF dataset loading
    - preserve current successful behavior for supported datasets
    - improve error clarity when dataset loading truly fails
  - `scripts/checks_remote.sh`
    - optionally include stronger shell/doc checks (non-breaking)
  - `README.md`
    - remove prompt-artifact contamination (`checksAction: ...`)
    - align Python support messaging with `pyproject.toml`
  - `.gitignore` (only if required)
    - verify transient dirs (`.venv/`, `sessions/`) are ignored consistently

- **Files to inspect for consistency (may modify if mismatches found)**
  - `pyproject.toml`
  - `requirements.txt`
  - `requirements-demo.txt`

[Functions]  
Primary function changes are in script helpers and dataset loading.

- **New functions**
  - In `scripts/_remote_common.sh`:
    - a single reusable remote Python resolver helper (e.g., `remote_resolve_python`) used by bootstrap/run/shell helpers.
  - Optional in `scripts/build_dataset.py`:
    - small loader wrapper function for HF dataset calls that enforces current supported options cleanly.

- **Modified functions**
  - `load_hf_dataset(repo_id: str) -> list[dict]` in `scripts/build_dataset.py`
    - remove deprecated args triggering trust_remote_code warnings
    - maintain robust conversion to list-of-records
  - `remote_bootstrap_env`, `remote_run_python_script`, `remote_run_shell` in `scripts/_remote_common.sh`
    - consume shared resolver helper
    - safer quoting and execution patterns
    - preserve and correctly render requested interpreter tokens in failure paths
  - helper command orchestration in `scripts/checks_remote.sh`
    - optional extra checks/lints if non-disruptive

- **Removed functions**
  - none expected; focus is consolidation rather than deletion.

[Classes]  
No class additions/modifications are required; current codebase is function/script oriented.

[Dependencies]  
No required runtime dependency additions are expected for core fixes.

Optional quality additions (only if approved and low-risk):
- shell lint tooling (e.g., `shellcheck`) as optional check path
- markdown lint/grep guard in checks pipeline

Dependency policy remains:
- canonical: `pyproject.toml` + `uv.lock`
- compatibility exports: `requirements*.txt` if retained must be documented and synchronized

[Testing]  
Testing will be phase-gated, with explicit verification after each step.

- **Existing tests to run repeatedly**
  - `uv run --no-sync --group research python -m unittest discover -s tests -v`
  - `VETQWEN_REMOTE_HOST=blackbox bash scripts/checks_remote.sh`

- **New/extended tests**
  - Add/extend unit tests for `scripts/build_dataset.py` around dataset loading path and warning-free behavior assumptions (where mockable).
  - Add targeted tests for parser/argument edge cases impacted by refactors.
  - Add explicit wrapper failure-path validation for invalid interpreter override:
    - run `VETQWEN_REMOTE_PYTHON=python3.13` and assert stderr includes the exact requested interpreter token.
  - Add smoke verification for remote wrappers:
    - synthetic generation smoke (small `--n`)
    - dataset build smoke with synthetic input
    - confirm artifact pull-back presence/expected files

- **Validation strategy**
  - Preserve existing green baseline (20/20 tests)
  - ensure remote wrappers still pass preflight + checks
  - ensure no regression in generated dataset artifacts and split outputs
  - verify README consistency and command snippets remain valid

[Implementation Order]  
Implement from highest-risk runtime reliability to hygiene, testing each phase before proceeding.

1. Baseline safety checkpoint: run local+remote checks and snapshot current status.
2. Refactor `_remote_common.sh` quoting/execution and deduplicate Python resolver.
3. Test Phase 2: run remote checks wrapper and one remote smoke command; confirm no behavior regressions.
4. Add explicit pre-coding validation for wrapper failure diagnostics:
   - run invalid override (`VETQWEN_REMOTE_PYTHON=python3.13`) and capture stderr.
   - confirm error message includes the requested interpreter value.
5. Update `scripts/build_dataset.py` HF loading path to remove deprecated trust_remote_code behavior.
6. Test Phase 5: run local unit tests + remote dataset build smoke and verify artifacts.
7. Clean docs/config hygiene (`README.md` artifact cleanup, version consistency, requirements messaging).
8. Test Phase 7: rerun checks and verify README command validity.
9. Optional quality hardening (`checks_remote.sh` enhancements for lint/consistency guards).
10. Final full verification sweep and produce completion summary with changed files, tests, and residual risks.

task_progress Items:
- [x] Step 1: Establish baseline and confirm current green local+remote test state before refactors
- [ ] Step 2: Harden `scripts/_remote_common.sh` (quoting + shared Python resolver refactor)
- [ ] Step 3: Verify wrapper behavior after shell refactor with remote checks and smoke run
- [ ] Step 4: Add explicit pre-coding validation for invalid Python override diagnostics (must include requested token)
- [ ] Step 5: Fix `scripts/build_dataset.py` HF loading path to remove deprecated trust_remote_code behavior
- [ ] Step 6: Validate dataset pipeline after loader fix (local tests + remote build smoke + artifact checks)
- [ ] Step 7: Clean docs/hygiene issues (`README.md` contamination, version/dependency messaging consistency)
- [ ] Step 8: Add/adjust quality checks and run final verification sweep
