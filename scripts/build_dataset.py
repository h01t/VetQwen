"""
build_dataset.py — VetQwen dataset construction pipeline

Steps:
1. Download Tier 1 datasets from Hugging Face Hub
2. Normalize each source into a unified ChatML instruction dict
3. Combine with any synthetic data from generate_synthetic.py
4. Deduplicate using sentence-transformer cosine similarity (threshold 0.95)
5. Stratified 80/10/10 split by species and condition
6. Write data/processed/train.jsonl, val.jsonl, test.jsonl

Usage:
    python scripts/build_dataset.py [--synthetic data/raw/synthetic.jsonl] [--output-dir data/processed]
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are VetQwen, an expert veterinary diagnostic assistant. "
    "Given a patient signalment and clinical symptoms, provide a structured "
    "differential diagnosis with clinical reasoning, ranked differentials, and "
    "a triage recommendation. Always remind the user that your output is not a "
    "substitute for professional veterinary examination."
)

HF_SOURCES = [
    "karenwky/pet-health-symptoms-dataset",
    "infinite-dataset-hub/VetHealthAssessment",
    "infinite-dataset-hub/VetPetCare",
]

DEDUP_THRESHOLD = 0.95  # cosine similarity above this → drop duplicate
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train / val / test
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_hf_dataset(repo_id: str) -> list[dict]:
    """Download a dataset from HuggingFace Hub and return raw records as a list of dicts."""
    from datasets import load_dataset  # type: ignore

    log.info(f"Downloading {repo_id} ...")
    ds = load_dataset(repo_id, trust_remote_code=True)

    # Flatten all splits into a single list
    records = []
    for split_name, split_ds in ds.items():
        for row in split_ds:
            row["_source"] = repo_id
            row["_split_origin"] = split_name
            records.append(dict(row))

    log.info(f"  {repo_id}: {len(records)} raw records")
    return records


# ---------------------------------------------------------------------------
# Normalization — one function per source
# ---------------------------------------------------------------------------


def normalize_pet_health_symptoms(record: dict) -> dict | None:
    """
    karenwky/pet-health-symptoms-dataset
    Expected fields: symptom, diagnosis, species (or similar — inspect and adapt)
    Returns a ChatML dict or None if record is unusable.
    """
    # TODO: inspect actual column names after download and adjust field access
    try:
        species = record.get("species", "Unknown species")
        symptom = record.get("symptom") or record.get("symptoms") or ""
        diagnosis = record.get("diagnosis") or record.get("label") or ""

        if not symptom or not diagnosis:
            return None

        user_content = f"Species: {species}\nPresenting complaint: {symptom}"
        assistant_content = (
            f"**Differential Diagnoses:**\n1. {diagnosis}\n\n"
            f"**Triage Recommendation:** Schedule within 48h\n"
            f"**Suggested Next Steps:** Please consult a veterinarian for a full examination.\n\n"
            f"*Note: This output is not a substitute for professional veterinary examination.*"
        )

        return _make_chatml(
            user_content, assistant_content, meta={"source": record["_source"]}
        )
    except Exception as e:
        log.warning(f"Skipping record from {record.get('_source')}: {e}")
        return None


def normalize_vet_health_assessment(record: dict) -> dict | None:
    """
    infinite-dataset-hub/VetHealthAssessment
    Expected fields: question, answer (inspect and adapt)
    """
    try:
        question = record.get("question") or record.get("input") or ""
        answer = record.get("answer") or record.get("output") or ""

        if not question or not answer:
            return None

        return _make_chatml(question, answer, meta={"source": record["_source"]})
    except Exception as e:
        log.warning(f"Skipping record: {e}")
        return None


def normalize_vet_pet_care(record: dict) -> dict | None:
    """
    infinite-dataset-hub/VetPetCare
    Expected fields: species, symptoms, diagnosis, treatment (inspect and adapt)
    """
    try:
        species = record.get("species", "Unknown species")
        symptoms = record.get("symptoms") or record.get("complaint") or ""
        diagnosis = record.get("diagnosis") or ""
        treatment = record.get("treatment") or record.get("treatment_plan") or ""

        if not symptoms or not diagnosis:
            return None

        user_content = f"Species: {species}\nPresenting complaint: {symptoms}"
        assistant_content = (
            f"**Differential Diagnoses:**\n1. {diagnosis}\n\n"
            f"**Suggested Next Steps:** {treatment}\n\n"
            f"*Note: This output is not a substitute for professional veterinary examination.*"
        )

        return _make_chatml(
            user_content, assistant_content, meta={"source": record["_source"]}
        )
    except Exception as e:
        log.warning(f"Skipping record: {e}")
        return None


# Map source repo_id → normalizer function
NORMALIZERS = {
    "karenwky/pet-health-symptoms-dataset": normalize_pet_health_symptoms,
    "infinite-dataset-hub/VetHealthAssessment": normalize_vet_health_assessment,
    "infinite-dataset-hub/VetPetCare": normalize_vet_pet_care,
}


def _make_chatml(
    user_content: str, assistant_content: str, meta: dict | None = None
) -> dict:
    """Wrap content in the ChatML message format VetQwen expects."""
    record: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
    if meta:
        record["_meta"] = meta
    return record


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate(samples: list[dict], threshold: float = DEDUP_THRESHOLD) -> list[dict]:
    """
    Remove near-duplicate samples using sentence-transformer cosine similarity.
    Pairs with cosine similarity > threshold are deduplicated (keep first occurrence).
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np
    except ImportError:
        log.warning("sentence-transformers not installed — skipping deduplication.")
        return samples

    log.info(f"Deduplicating {len(samples)} samples (threshold={threshold}) ...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build a text representation for each sample (user turn content)
    texts = [s["messages"][1]["content"] for s in samples]
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )

    # Greedy deduplication: O(n^2) — acceptable for <10k samples; use FAISS for larger sets
    keep = [True] * len(samples)
    for i in range(len(samples)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(samples)):
            if not keep[j]:
                continue
            sim = float(
                np.dot(embeddings[i], embeddings[j])
                / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8)
            )
            if sim > threshold:
                keep[j] = False

    deduped = [s for s, k in zip(samples, keep) if k]
    log.info(
        f"After deduplication: {len(deduped)} samples ({len(samples) - len(deduped)} removed)"
    )
    return deduped


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split(
    samples: list[dict],
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified 80/10/10 split.
    Stratification key: species extracted from the user turn (best-effort).
    """
    random.seed(seed)

    def extract_species(sample: dict) -> str:
        user_text = sample["messages"][1]["content"].lower()
        for s in ["dog", "cat", "cattle", "pig", "sheep"]:
            if s in user_text:
                return s
        return "unknown"

    # Group by species
    buckets: dict[str, list[dict]] = {}
    for sample in samples:
        key = extract_species(sample)
        buckets.setdefault(key, []).append(sample)

    train, val, test = [], [], []
    for key, group in buckets.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
        log.info(
            f"  {key}: {n_train} train / {int(n * ratios[1])} val / {n - n_train - n_val} test"
        )

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records)} records to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)

    # 1. Download and normalize Tier 1 datasets
    all_samples: list[dict] = []
    for repo_id in HF_SOURCES:
        raw_records = load_hf_dataset(repo_id)
        normalizer = NORMALIZERS[repo_id]
        normalized = [normalizer(r) for r in raw_records]
        normalized = [n for n in normalized if n is not None]
        log.info(f"  Normalized: {len(normalized)} usable samples from {repo_id}")
        all_samples.extend(normalized)

    # 2. Load synthetic data (Tier 2) if provided
    if args.synthetic:
        synth_path = Path(args.synthetic)
        if synth_path.exists():
            synth = load_jsonl(synth_path)
            log.info(f"Loaded {len(synth)} synthetic samples from {synth_path}")
            all_samples.extend(synth)
        else:
            log.warning(f"Synthetic file not found: {synth_path}")

    log.info(f"Total samples before deduplication: {len(all_samples)}")

    # 3. Deduplicate
    all_samples = deduplicate(all_samples, threshold=DEDUP_THRESHOLD)

    # 4. Stratified split
    log.info("Splitting dataset ...")
    train, val, test = stratified_split(all_samples)
    log.info(f"Final split — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    # 5. Write JSONL
    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")
    write_jsonl(test, output_dir / "test.jsonl")

    log.info("Dataset construction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VetQwen training dataset")
    parser.add_argument(
        "--synthetic",
        type=str,
        default=None,
        help="Path to synthetic JSONL file from generate_synthetic.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to write train/val/test JSONL files",
    )
    args = parser.parse_args()
    main(args)
