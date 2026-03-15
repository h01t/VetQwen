# VetQwen — Fine-Tuned Veterinary Diagnostic LLM

A QLoRA fine-tuned version of **Qwen2.5-3B-Instruct** for structured veterinary differential diagnosis.
Given a patient signalment and presenting symptoms, VetQwen produces a ranked differential diagnosis,
clinical reasoning, triage recommendation, and suggested next steps.

> **Disclaimer:** This model is for educational and research purposes only.
> It is **not** a substitute for professional veterinary examination or diagnosis.

---

## Model Overview

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Fine-tuning method | QLoRA (4-bit, NF4) |
| VRAM requirement | ≤8GB (RTX 3070 or equivalent) |
| Species scope | Dogs, Cats, Cattle, Pigs, Sheep |
| Task | Symptom → Structured Differential Diagnosis |
| Framework | PyTorch + HuggingFace `transformers` + `peft` + `trl` |
| Synthetic data | Ollama + `qwen2.5:7b` (livestock cases) |
| LLM-as-judge | Ollama + `qwen2.5:7b` (run separately after evaluation) |

---

## Output Format

```
**Species & Signalment:** [species, age, sex, breed]
**Presenting Symptoms:** [summary of reported symptoms]

**Assessment:**
[2-3 sentence clinical reasoning based on symptom pattern]

**Differential Diagnoses (ranked by likelihood):**
1. [Most likely diagnosis] — [brief rationale]
2. [Second differential] — [brief rationale]
3. [Third differential] — [brief rationale]

**Triage Recommendation:** [Urgent / Schedule within 48h / Monitor at home]
**Suggested Next Steps:** [key diagnostics or first-line treatment considerations]
```

---

## Project Structure

```
vetqwen/
├── data/
│   ├── raw/                   # Downloaded HF datasets (git-ignored)
│   ├── processed/             # Cleaned JSONL splits (git-ignored)
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── dataset_card.md
├── scripts/
│   ├── build_dataset.py       # Download, process, split datasets
│   ├── generate_synthetic.py  # Ollama-based synthetic data generation
│   ├── train.py               # QLoRA fine-tuning with SFTTrainer
│   ├── evaluate.py            # ROUGE-L, BERTScore, format compliance
│   └── run_judge.py           # Standalone LLM-as-judge via Ollama
├── configs/
│   └── train_default.yaml     # Training hyperparameters
├── app/
│   └── gradio_demo.py         # Gradio inference UI
├── adapter/                   # LoRA adapter weights after training (git-ignored)
├── results/                   # Evaluation results JSON (git-ignored)
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **NVIDIA GPU** with CUDA drivers (RTX 3070 or equivalent, ≤8GB VRAM)
- **CUDA toolkit 12.x** installed on the system
- **Python 3.10+**
- **Ollama** with `qwen2.5:7b` pulled (for synthetic data generation and LLM-as-judge)

---

## Quickstart

### 1. Environment setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support FIRST (must match your CUDA version)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

**Verify GPU is working:**

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')"
```

Both should print without errors or warnings.

### 2. Generate synthetic livestock data (optional)

```bash
# Requires Ollama running locally: ollama serve
# Requires model pulled: ollama pull qwen2.5:7b
python scripts/generate_synthetic.py --n 400
# Output: data/raw/synthetic.jsonl
```

### 3. Build the dataset

```bash
# Without synthetic data:
python scripts/build_dataset.py

# With synthetic data (recommended — run after step 2):
python scripts/build_dataset.py --synthetic data/raw/synthetic.jsonl
```

### 4. Run baseline evaluation

```bash
python scripts/evaluate.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --split test \
    --run-name baseline_3b \
    --no-judge

# Quick dry-run (2 samples):
python scripts/evaluate.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --split test \
    --run-name baseline_3b_quick \
    --limit 2 --no-judge
```

### 5. Fine-tune

```bash
python scripts/train.py --config configs/train_default.yaml
```

### 6. View training curves

```bash
tensorboard --logdir ./checkpoints --bind_all
# Open http://localhost:6006
```

### 7. Evaluate fine-tuned model

```bash
python scripts/evaluate.py \
    --model ./adapter \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --split test \
    --run-name vetqwen_3b_r16 \
    --no-judge
```

### 8. Run LLM-as-judge (separately, after evaluation)

The judge runs via Ollama and needs the GPU free (no evaluation model loaded).
Run this **after** the evaluation scripts above have finished:

```bash
# Judge baseline predictions
python scripts/run_judge.py \
    --predictions results/baseline_3b_predictions.jsonl \
    --run-name baseline_3b

# Judge fine-tuned predictions
python scripts/run_judge.py \
    --predictions results/vetqwen_3b_r16_predictions.jsonl \
    --run-name vetqwen_3b_r16
```

### 9. Compare results

```bash
# Automatic metrics:
cat results/baseline_3b.json
cat results/vetqwen_3b_r16.json

# Judge scores:
cat results/baseline_3b_judge.json
cat results/vetqwen_3b_r16_judge.json
```

### 10. Launch demo

```bash
python app/gradio_demo.py --adapter ./adapter --base-model Qwen/Qwen2.5-3B-Instruct
# Open http://localhost:7860
```

---

## Remote Access (SSH from Mac)

```bash
# SSH into workstation
ssh user@workstation-ip

# Activate environment and run scripts from project root
cd ~/dev/vetqwen && source .venv/bin/activate

# SSH tunnels for web UIs (run from Mac terminal):
ssh -L 6006:localhost:6006 user@workstation-ip   # TensorBoard
ssh -L 7860:localhost:7860 user@workstation-ip   # Gradio demo
```

---

## Pinned Package Versions

These exact versions are tested and known to work for QLoRA on 8GB GPUs:

| Package | Version | Notes |
|---------|---------|-------|
| `torch` | 2.4.1+cu121 | Must install separately with `--index-url` |
| `transformers` | 4.44.2 | Later versions have OOM issues during 4-bit loading |
| `bitsandbytes` | 0.43.3 | Compatible with transformers 4.44.2 |
| `accelerate` | 0.33.0 | Compatible with bitsandbytes 0.43.3 |
| `peft` | 0.12.0 | QLoRA / LoRA adapter support |
| `trl` | 0.9.6 | SFTTrainer for supervised fine-tuning |

---

## Hardware Constraints

| Constraint | Decision |
|---|---|
| ≤8GB VRAM | 4-bit QLoRA, 3B model |
| Max sequence length | 512 tokens |
| Batch size | 1, gradient_accumulation_steps=16 |
| Optimizer | paged_adamw_8bit |
| Precision | bf16 |

---

## Ablation Runs

| Run | LoRA rank `r` | Dataset size | Notes |
|-----|--------------|--------------|-------|
| A | 8 | 100% | Low rank |
| B | 16 | 100% | **Primary run** |
| C | 32 | 100% | High rank — monitor VRAM |
| D | 16 | 50% | Data size sensitivity |
| E | 16 | 25% | Minimum viable data |

---

## Data Sources

- `karenwky/pet-health-symptoms-dataset` — 2,000 labeled samples
- `infinite-dataset-hub/VetHealthAssessment` — structured Q&A
- `infinite-dataset-hub/VetPetCare` — clinic records
- Synthetic: Ollama + Qwen2.5:7b (livestock cases)

---

*Model: Qwen2.5-3B-Instruct | Fine-tuning: QLoRA 4-bit | Hardware target: RTX 3070 ≤8GB VRAM*
