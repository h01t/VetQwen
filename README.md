# VetQwen

VetQwen is a QLoRA fine-tune of `Qwen/Qwen2.5-3B-Instruct` for structured veterinary differential diagnosis. Given patient signalment and presenting symptoms, it produces:

- a short assessment
- ranked differentials
- a triage recommendation
- suggested next steps

> **Disclaimer:** VetQwen is a research prototype for educational use. It is not a substitute for professional veterinary examination, diagnosis, or treatment.
>
> **Project status:** Scaffold-stage research prototype. The repo is organized for reproducible remote runs, but it is still an experimental pipeline rather than a production clinical system.

## Model Overview

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Fine-tuning method | QLoRA (4-bit, NF4) |
| Hardware target | Single NVIDIA GPU with ~8GB VRAM |
| Species scope | Dog, Cat, Cattle, Pig, Sheep |
| Synthetic data | Ollama + `qwen2.5:7b` |
| Judge model | Ollama, typically `qwen2.5:3b-instruct` or `qwen2.5:7b` |

## Output Format

```text
**Species & Signalment:** [species, age, sex, breed]
**Presenting Symptoms:** [summary of reported symptoms]

**Assessment:**
[2-3 sentence clinical reasoning]

**Differential Diagnoses (ranked by likelihood):**
1. [Most likely diagnosis] — [brief rationale]
2. [Second differential] — [brief rationale]
3. [Third differential] — [brief rationale]

**Triage Recommendation:** [Urgent / Schedule within 48h / Monitor at home]
**Suggested Next Steps:** [diagnostics or first-line care considerations]
```

## Project Structure

```text
vetqwen/
├── app/
│   └── gradio_demo.py
├── configs/
│   └── train_default.yaml
├── data/
│   ├── raw/                   # Synthetic/raw inputs (git-ignored)
│   ├── processed/             # Built train/val/test JSONL splits (git-ignored)
│   ├── review/                # Generated manual-review exports (git-ignored)
│   └── dataset_card.md
├── results/                   # Metrics, predictions, comparison summaries (git-ignored)
├── scripts/
│   ├── vetqwen_core/          # Shared internal helpers
│   ├── _remote_common.sh
│   ├── generate_synthetic.py
│   ├── build_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_judge.py
│   ├── compare_results.py
│   ├── build_review_subset.py
│   ├── preflight.py
│   ├── generate_synthetic_remote.sh
│   ├── build_dataset_remote.sh
│   ├── checks_remote.sh
│   ├── train_remote.sh
│   ├── evaluate_remote.sh
│   ├── run_judge_remote.sh
│   └── compare_remote.sh
├── requirements.txt           # Full remote research stack
├── requirements-demo.txt      # Local Mac demo stack
└── README.md
```

## Two Supported Workflows

### 1. Local Demo on Mac

Use the Mac for:

- running the Gradio demo
- reviewing pulled artifacts in `results/`
- launching the remote wrappers

Do not use the Mac as the canonical environment for dataset build, training, evaluation, or judge scoring.

### 2. Remote Research Pipeline on Blackbox

Use `blackbox` over SSH for:

- synthetic data generation
- dataset building
- checks/tests
- training
- evaluation
- judge scoring
- result comparison

All remote wrappers start on the Mac, run the heavy work on `blackbox`, and pull artifacts back automatically.

## Prerequisites

### Local Mac

- Python 3.11 recommended
- SSH access to `blackbox`
- `rsync`

### Remote Workstation (`blackbox`)

- Linux + NVIDIA GPU + CUDA drivers
- Python 3.11 available as `python3.11` or overridden via `VETQWEN_REMOTE_PYTHON`
- Ollama running for synthetic generation and judge scoring

Optional SSH config:

```sshconfig
Host blackbox
  HostName your-host-or-ip
  User your-username
```

## Local Demo on Mac

### 1. Install demo-only dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-demo.txt
python scripts/preflight.py --profile demo --skip-ollama
```

### 2. Launch the demo

```bash
python app/gradio_demo.py --device auto --adapter ./adapter
```

Notes:

- `--device auto` is the safe default on Mac.
- If `./adapter` is missing, the demo falls back to the base model.
- The demo path is intentionally separate from the remote research stack.

## Remote Research Pipeline on Blackbox

The wrappers below are the canonical workflow.

Shared environment overrides:

- `VETQWEN_REMOTE_HOST` default: `blackbox`
- `VETQWEN_REMOTE_DIR` default: `~/vetqwen`
- `VETQWEN_REMOTE_PYTHON` default: `python3.11`
- `VETQWEN_REMOTE_TORCH_VERSION` default: `2.4.1`
- `VETQWEN_REMOTE_TORCH_INDEX_URL` default: `https://download.pytorch.org/whl/cu121`
- `VETQWEN_REMOTE_OLLAMA_URL` default: `http://127.0.0.1:11434`
- `VETQWEN_REMOTE_JUDGE_MODEL` optional

Important behavior:

- The remote project directory is created automatically if it does not exist.
- The remote venv is bootstrapped automatically.
- `requirements.txt` is installed remotely by the wrappers.
- Artifacts are pulled back automatically at the end of each wrapper run.

### Synthetic generation

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/generate_synthetic_remote.sh \
  --n 400 \
  --output data/raw/synthetic.jsonl
```

### Dataset build

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/build_dataset_remote.sh \
  --synthetic data/raw/synthetic.jsonl
```

This pulls back:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/duplicate_audit.json`

### Remote checks

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/checks_remote.sh
```

This runs:

- `python -m py_compile`
- `python -m unittest discover -s tests -v`
- `--help` smoke checks for the Python entrypoints
- `bash -n scripts/*.sh`

### Remote training

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/train_remote.sh \
  --config configs/train_default.yaml \
  --run-name vetqwen_r16_clean_v2
```

This pulls back:

- `adapter/`
- `checkpoints/`
- `results/`

### Remote evaluation

Baseline:

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/evaluate_remote.sh \
  --model Qwen/Qwen2.5-3B-Instruct \
  --split test \
  --run-name baseline_3b_clean_v2 \
  --seed 42 \
  --no-judge
```

Fine-tuned adapter:

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/evaluate_remote.sh \
  --model ./adapter \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --split test \
  --run-name vetqwen_r16_clean_v2 \
  --seed 42 \
  --no-judge
```

### Remote judge scoring

Baseline:

```bash
VETQWEN_REMOTE_HOST=blackbox \
VETQWEN_REMOTE_JUDGE_MODEL=qwen2.5:3b-instruct \
./scripts/run_judge_remote.sh \
  --predictions results/baseline_3b_clean_v2_predictions.jsonl \
  --run-name baseline_3b_clean_v2 \
  --sample 50 \
  --seed 42
```

Fine-tuned adapter:

```bash
VETQWEN_REMOTE_HOST=blackbox \
VETQWEN_REMOTE_JUDGE_MODEL=qwen2.5:3b-instruct \
./scripts/run_judge_remote.sh \
  --predictions results/vetqwen_r16_clean_v2_predictions.jsonl \
  --run-name vetqwen_r16_clean_v2 \
  --sample 50 \
  --seed 42
```

### Remote comparison pass

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/compare_remote.sh \
  --baseline results/baseline_3b_clean_v2.json \
  --candidate results/vetqwen_r16_clean_v2.json \
  --baseline-judge results/baseline_3b_clean_v2_judge.json \
  --candidate-judge results/vetqwen_r16_clean_v2_judge.json \
  --output results/comparisons/vetqwen_r16_clean_v2_vs_baseline.md
```

This pulls back:

- `results/comparisons/*.md`
- `results/comparisons/*.json`

## Next-Run Checklist

This is the concrete recommended sequence for the next full cycle.

### 1. Optional: regenerate synthetic livestock data remotely

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/generate_synthetic_remote.sh \
  --n 400 \
  --output data/raw/synthetic.jsonl
```

### 2. Rebuild the cleaned dataset remotely

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/build_dataset_remote.sh \
  --synthetic data/raw/synthetic.jsonl
```

### 3. Run remote checks before training

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/checks_remote.sh
```

### 4. Retrain on the rebuilt dataset

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/train_remote.sh \
  --config configs/train_default.yaml \
  --run-name vetqwen_r16_clean_v2
```

### 5. Run deterministic clean baseline evaluation

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/evaluate_remote.sh \
  --model Qwen/Qwen2.5-3B-Instruct \
  --split test \
  --run-name baseline_3b_clean_v2 \
  --seed 42 \
  --no-judge
```

### 6. Run deterministic clean post-retrain adapter evaluation

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/evaluate_remote.sh \
  --model ./adapter \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --split test \
  --run-name vetqwen_r16_clean_v2 \
  --seed 42 \
  --no-judge
```

### 7. Run clean baseline judge scoring remotely

```bash
VETQWEN_REMOTE_HOST=blackbox \
VETQWEN_REMOTE_JUDGE_MODEL=qwen2.5:3b-instruct \
./scripts/run_judge_remote.sh \
  --predictions results/baseline_3b_clean_v2_predictions.jsonl \
  --run-name baseline_3b_clean_v2 \
  --sample 50 \
  --seed 42
```

### 8. Run clean post-retrain adapter judge scoring remotely

```bash
VETQWEN_REMOTE_HOST=blackbox \
VETQWEN_REMOTE_JUDGE_MODEL=qwen2.5:3b-instruct \
./scripts/run_judge_remote.sh \
  --predictions results/vetqwen_r16_clean_v2_predictions.jsonl \
  --run-name vetqwen_r16_clean_v2 \
  --sample 50 \
  --seed 42
```

### 9. Run the post-retrain comparison pass remotely

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/compare_remote.sh \
  --baseline results/baseline_3b_clean_v2.json \
  --candidate results/vetqwen_r16_clean_v2.json \
  --baseline-judge results/baseline_3b_clean_v2_judge.json \
  --candidate-judge results/vetqwen_r16_clean_v2_judge.json \
  --output results/comparisons/vetqwen_r16_clean_v2_vs_baseline.md
```

### 10. Review against these acceptance gates

- `parse_success_rate >= 0.95`
- `format_compliance >= 0.95`
- `diagnosis_hit_rate` must beat the clean baseline on the same split
- `urgent_recall > 0.0`
- judge `clinical_accuracy` must not regress below baseline
- VetPetCare and VetHealthAssessment source metrics must not regress materially

If a gate fails, generate a manual review subset from the predictions and inspect all urgent cases plus the labeled misses before changing model size or architecture.

## Manual Review Subset

Use this after evaluation if you want a stable urgent-plus-labeled review slice:

```bash
python scripts/build_review_subset.py \
  --input results/vetqwen_r16_clean_v2_predictions.jsonl \
  --output data/review/vetqwen_r16_clean_v2_review.jsonl \
  --target-size 40 \
  --seed 42
```

`data/review/*.jsonl` is treated as generated output, not source.

## Preflight Profiles

Remote research stack:

```bash
python scripts/preflight.py --profile remote
```

Local demo stack:

```bash
python scripts/preflight.py --profile demo --skip-ollama
```

Behavior:

- `remote` is strict about CUDA, Python compatibility, and the full research stack.
- `demo` is relaxed about CUDA and remote-only packages.

## Advanced: Directly on Blackbox

The Python entrypoints still work directly on the workstation if you are already logged into `blackbox` and have activated the venv:

```bash
source .venv/bin/activate
python scripts/generate_synthetic.py --n 100 --output data/raw/synthetic.jsonl
python scripts/build_dataset.py --synthetic data/raw/synthetic.jsonl
python scripts/train.py --config configs/train_default.yaml
python scripts/evaluate.py --model ./adapter --base-model Qwen/Qwen2.5-3B-Instruct --split test --run-name local_eval --no-judge
python scripts/run_judge.py --predictions results/local_eval_predictions.jsonl --run-name local_eval
python scripts/compare_results.py --baseline results/baseline.json --candidate results/candidate.json
```

The wrappers remain the recommended path for normal use.

## Pinned Versions

The intended remote research stack is pinned around:

| Package | Version |
|---|---|
| `torch` | `2.4.1+cu121` |
| `transformers` | `4.44.2` |
| `bitsandbytes` | `0.43.3` |
| `accelerate` | `0.33.0` |
| `peft` | `0.12.0` |
| `trl` | `0.9.6` |

Interpreter guidance:

- Recommended: Python 3.11
- Remote research stack support target: Python 3.10-3.12
- Python 3.13 is not part of the pinned remote stack

## Data Sources

- `karenwky/pet-health-symptoms-dataset`
- `infinite-dataset-hub/VetHealthAssessment`
- `infinite-dataset-hub/VetPetCare`
- synthetic livestock cases generated with Ollama
