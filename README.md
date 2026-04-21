# VetQwen

VetQwen is a QLoRA fine-tune of `Qwen/Qwen2.5-3B-Instruct` for structured veterinary differential diagnosis. Given patient signalment and presenting symptoms, it produces a short assessment, ranked differentials, a triage recommendation, and suggested next steps in a consistent clinical template.

[Project overview report](docs/project-overview.md)

> **Disclaimer:** VetQwen is a research prototype for educational use. It is not a substitute for professional veterinary examination, diagnosis, or treatment.
>
> **Academic note:** This repository is being prepared as a polished academic project submission for a master's application. It is demo-ready, but it remains a non-clinical research system rather than a production medical product.

## Project Snapshot

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Fine-tuning method | QLoRA (4-bit, NF4) |
| Primary task | Structured veterinary differential diagnosis |
| Species scope | Dog, Cat, Cattle, Pig, Sheep |
| Demo interface | Gradio |
| Research workflow | Remote-first `uv` pipeline for data build, training, evaluation, judging, and comparison |
| Canonical package metadata | [pyproject.toml](pyproject.toml) |
| Full submission document | [docs/project-overview.md](docs/project-overview.md) |

## Why This Repo Exists

- Demonstrates an end-to-end LLM fine-tuning workflow for a domain-specific healthcare-adjacent task.
- Packages the project as a reproducible research codebase rather than a one-off notebook.
- Includes both a local demo path and a remote GPU workflow for heavier experimentation.
- Publishes evaluation and comparison artifacts so results can be inspected directly.

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

## Repository Layout

```text
vetqwen/
├── app/                      # Gradio demo
├── configs/                  # Training configuration
├── data/
│   ├── dataset_card.md       # Dataset summary and limitations
│   ├── raw/                  # Generated / downloaded raw data (git-ignored)
│   ├── processed/            # Built dataset splits (git-ignored)
│   └── review/               # Manual review exports (git-ignored)
├── docs/
│   ├── assets/               # Submission figures and demo preview
│   └── project-overview.md   # Canonical report for PDF export
├── results/                  # Versioned reference metrics plus ignored regenerated outputs
├── scripts/
│   ├── vetqwen_core/         # Shared helpers
│   ├── build_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_judge.py
│   └── compare_results.py
├── tests/                    # Unit tests
├── pyproject.toml            # Canonical package metadata
├── uv.lock                   # Locked environment
└── README.md
```

Large or regeneratable artifacts such as checkpoints, local sessions, raw data dumps, and exported PDFs stay out of version control. The repo keeps lightweight reference artifacts such as selected evaluation JSON files and adapter metadata so the project remains inspectable after cloning.

## Local Demo Quickstart

```bash
uv sync --group demo --locked --python 3.11
uv run --no-sync --group demo python scripts/preflight.py --profile demo --skip-ollama
uv run --no-sync --group demo python app/gradio_demo.py --device auto --adapter ./adapter
```

Notes:

- `--device auto` is the safe default on macOS.
- If `./adapter` is missing, the demo falls back to the base model for interface smoke testing.
- The demo is intentionally separate from the heavier remote research stack.

## Research Workflow Summary

The full operational detail lives in [docs/project-overview.md](docs/project-overview.md), but the research loop is:

1. Generate livestock-focused synthetic cases with Ollama.
2. Build and normalize the instruction-tuning dataset.
3. Run checks and unit tests.
4. Fine-tune the model with QLoRA on a remote GPU machine.
5. Evaluate baseline and fine-tuned variants.
6. Score predictions with an LLM judge.
7. Compare runs and inspect guardrails.

Representative commands:

```bash
VETQWEN_REMOTE_HOST=blackbox ./scripts/build_dataset_remote.sh --synthetic data/raw/synthetic.jsonl
VETQWEN_REMOTE_HOST=blackbox ./scripts/train_remote.sh --config configs/train_default.yaml --run-name vetqwen_r16_clean_v2
VETQWEN_REMOTE_HOST=blackbox ./scripts/evaluate_remote.sh --model ./adapter --base-model Qwen/Qwen2.5-3B-Instruct --split test --run-name vetqwen_r16_clean_v2 --seed 42 --no-judge
```

## Headline Results

The main public comparison currently emphasized in the repo is [`results/comparisons/vetqwen_r16_uv_migration_vs_baseline.md`](results/comparisons/vetqwen_r16_uv_migration_vs_baseline.md).

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Diagnosis hit rate | 0.0000 | 0.7500 |
| Parse success rate | 0.0000 | 1.0000 |
| Format compliance | 0.0000 | 1.0000 |
| Triage accuracy | 0.0227 | 0.9545 |
| Urgent recall | 0.6667 | 1.0000 |
| ROUGE-L | 0.1455 | 0.9855 |
| Judge clinical accuracy | 3.1600 | 1.7000 |

Takeaway:

- Fine-tuning dramatically improves structure adherence and task-specific output quality.
- The current judge comparison also shows a regression in `clinical_accuracy`, so the system should be presented honestly as promising but still clinically limited.

## Limitations

- Public veterinary datasets are sparse and skew toward companion animals.
- The livestock portion relies partly on synthetic generation and has not been validated by a licensed veterinarian.
- Evaluation artifacts are useful research evidence, not proof of clinical safety.
- The intended research workflow assumes access to a remote CUDA-capable machine.

## Documentation

- [docs/project-overview.md](docs/project-overview.md): formal project report with diagrams, results discussion, and appendix.
- [data/dataset_card.md](data/dataset_card.md): dataset composition, processing steps, and known limitations.

## License

This project is released under the [MIT License](LICENSE).
