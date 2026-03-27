"""
build_dataset.py — VetQwen dataset construction pipeline

Steps:
1. Download Tier 1 datasets from Hugging Face Hub
2. Normalize each source into a unified ChatML instruction dict
3. Combine with any synthetic data from generate_synthetic.py
4. Filter to the README target species by default
5. Deduplicate using sentence-transformer cosine similarity (threshold 0.95)
6. Stratified 80/10/10 split by normalized species and condition
7. Write data/processed/train.jsonl, val.jsonl, test.jsonl

Usage:
    python scripts/build_dataset.py [--synthetic data/raw/synthetic.jsonl] [--output-dir data/processed]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from vetqwen_core.jsonl import load_jsonl, write_jsonl
from vetqwen_core.records import (
    build_patient_prompt as build_user_content,
    build_structured_response,
    get_message_content,
    make_chatml_record as make_chatml,
)
from vetqwen_core.text import canonicalize_triage, clean_text, extract_triage_from_response

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

HF_SOURCES = [
    "karenwky/pet-health-symptoms-dataset",
    "infinite-dataset-hub/VetHealthAssessment",
    "infinite-dataset-hub/VetPetCare",
]

DEDUP_THRESHOLD = 0.95  # cosine similarity above this → drop duplicate
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train / val / test
RANDOM_SEED = 42
TARGET_SPECIES = {"dog", "cat", "cattle", "pig", "sheep"}

SPECIES_ALIASES = {
    "dog": [
        "dog",
        "dogs",
        "canine",
        "canines",
        "puppy",
        "puppies",
    ],
    "cat": [
        "cat",
        "cats",
        "feline",
        "felines",
        "kitten",
        "kittens",
    ],
    "cattle": [
        "cattle",
        "bovine",
        "cow",
        "cows",
        "calf",
        "calves",
        "heifer",
        "heifers",
        "bull",
        "bulls",
        "steer",
        "steers",
    ],
    "pig": [
        "pig",
        "pigs",
        "swine",
        "porcine",
        "piglet",
        "piglets",
        "sow",
        "sows",
        "boar",
        "boars",
    ],
    "sheep": [
        "sheep",
        "ovine",
        "ewe",
        "ewes",
        "ram",
        "rams",
        "lamb",
        "lambs",
    ],
    "horse": [
        "horse",
        "horses",
        "equine",
        "mare",
        "mares",
        "stallion",
        "stallions",
        "gelding",
        "geldings",
        "foal",
        "foals",
    ],
}

URGENT_TRIAGE_PATTERNS = [
    "bloody diarrhea",
    "difficulty breathing",
    "respiratory distress",
    "unable to rise",
    "unable to stand",
    "unable to walk",
    "head pressing",
    "seizure",
    "seizures",
    "fainting",
    "collapse",
    "collapsed",
    "severe abdominal distension",
    "intestinal blockage",
    "obstruction",
    "colic",
    "parvovirus",
    "parvoviral",
    "heartworm",
    "down in the field",
    "paralyzed",
]

NON_CONDITION_KEYWORDS = [
    "therapy",
    "support",
    "supportive care",
    "surgery",
    "surgical",
    "procedure",
    "management",
    "pain management",
    "control",
    "medication",
    "medications",
    "hydration",
    "diet change",
    "dietary changes",
    "monitoring",
    "tooth extraction",
    "tail docking",
    "orchiectomy",
    "knee surgery",
]

GENERIC_SYMPTOM_LABELS = {
    "abnormal behavior",
    "anorexia",
    "behavior change",
    "bloody diarrhea",
    "cough",
    "dehydration",
    "diarrhea",
    "difficulty breathing",
    "fever",
    "head pressing",
    "joint pain",
    "lethargy",
    "night blindness",
    "pain",
    "seizures",
    "vomiting",
    "weight loss",
}

CONDITION_HINTS = [
    "disease",
    "syndrome",
    "virus",
    "viral",
    "infection",
    "influenza",
    "bronchitis",
    "dermatitis",
    "distemper",
    "gastroenteritis",
    "hyperthyroidism",
    "hypothyroidism",
    "diabetes",
    "epilepsy",
    "parvovirus",
    "herpesvirus",
    "heartworm",
    "stones",
    "asthma",
    "atopy",
    "osteoarthritis",
    "cryptorchidism",
    "ivdd",
    "intervertebral disc disease",
    "gdv",
    "uti",
    "kidney disease",
    "heart disease",
    "anemia",
    "cognitive dysfunction",
    "heat stroke",
    "heat exhaustion",
    "chronic",
    "contracted",
]

CONDITION_SUFFIXES = (
    "emia",
    "itis",
    "osis",
    "iasis",
    "pathy",
    "plasia",
    "dysfunction",
    "syndrome",
    "disease",
)

URGENT_TARGET_TRAIN_RATIO = 0.15
NEAR_DUPLICATE_RATIO = 0.92


def canonicalize_condition(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    text = text.casefold()
    text = re.sub(r"[^a-z0-9\s/&-]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text or None
def is_urgent_triage(value: Any) -> bool:
    return canonicalize_triage(value) == "Urgent"


def normalize_source_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).split(",")
    return [clean_text(item) for item in items if clean_text(item)]


def cleanup_condition_text(value: Any) -> str | None:
    text = canonicalize_condition(value)
    if not text:
        return None

    cleanup_patterns = [
        r"^(?:dog|cat|horse|cow|pig|sheep|animal)\s+suffering from\s+",
        r"^(?:dog|cat|horse|cow|pig|sheep)\s+with\s+",
        r"^contracted\s+",
        r"^diagnosed with\s+",
    ]
    for pattern in cleanup_patterns:
        text = re.sub(pattern, "", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text or None


def looks_like_non_condition_label(value: Any) -> bool:
    text = cleanup_condition_text(value)
    if not text:
        return True
    return any(keyword in text for keyword in NON_CONDITION_KEYWORDS)


def looks_like_condition_label(value: Any) -> bool:
    text = cleanup_condition_text(value)
    if not text:
        return False
    if looks_like_non_condition_label(text):
        return False
    if text in GENERIC_SYMPTOM_LABELS:
        return False
    if any(hint in text for hint in CONDITION_HINTS):
        return True
    return any(text.endswith(suffix) for suffix in CONDITION_SUFFIXES)


def select_condition_label(
    diagnosis_text: str,
    complaint_text: str,
) -> tuple[str | None, str | None]:
    diagnosis = cleanup_condition_text(diagnosis_text)
    complaint = cleanup_condition_text(complaint_text)

    if diagnosis and not looks_like_non_condition_label(diagnosis):
        if looks_like_condition_label(diagnosis) or (
            diagnosis not in GENERIC_SYMPTOM_LABELS and len(diagnosis.split()) >= 2
        ):
            return diagnosis, "diagnosis"
    if complaint and looks_like_condition_label(complaint):
        return complaint, "complaint_fallback"
    return None, None


def detect_species_from_text(text: str) -> str:
    lowered = clean_text(text).casefold()
    if not lowered:
        return "unknown"

    for species, aliases in SPECIES_ALIASES.items():
        for alias in aliases:
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, lowered):
                return species
    return "unknown"


def normalize_species(
    raw_species: Any = None, text: str = "", labels: list[str] | None = None
) -> str:
    labels = labels or []
    for candidate in [clean_text(raw_species), *labels]:
        if not candidate:
            continue
        lowered = candidate.casefold()
        for species, aliases in SPECIES_ALIASES.items():
            if any(alias in lowered for alias in aliases):
                return species

    detected = detect_species_from_text(text)
    return detected


def infer_triage(symptoms: str, diagnosis: str = "") -> str:
    combined = f"{clean_text(symptoms)} {clean_text(diagnosis)}".casefold()
    if any(keyword in combined for keyword in URGENT_TRIAGE_PATTERNS):
        return "Urgent"
    if combined:
        return "Schedule within 48h"
    return "Monitor at home"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_hf_dataset(repo_id: str) -> list[dict]:
    """Download a dataset from Hugging Face Hub and return raw records as a list of dicts."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: 'datasets'.\n"
            "Install the local data-prep dependencies first, for example:\n"
            "  python -m pip install \"datasets>=2.20.0\"\n"
            "Optional but recommended for deduplication:\n"
            "  python -m pip install \"sentence-transformers>=3.0.0\"\n"
            "You can also run:\n"
            "  python scripts/preflight.py --skip-ollama"
        ) from exc

    log.info("Downloading %s ...", repo_id)
    dataset = load_dataset(repo_id, trust_remote_code=True)

    records: list[dict] = []
    for split_name, split_ds in dataset.items():
        for row in split_ds:
            record = dict(row)
            record["_source"] = repo_id
            record["_split_origin"] = split_name
            records.append(record)

    log.info("  %s: %s raw records", repo_id, len(records))
    return records


# ---------------------------------------------------------------------------
# Normalization — one function per source
# ---------------------------------------------------------------------------


def normalize_pet_health_symptoms(record: dict) -> dict[str, Any] | None:
    """
    karenwky/pet-health-symptoms-dataset
    Real fields: text, condition, record_type
    """
    text = clean_text(record.get("text"))
    condition_label = clean_text(record.get("condition"))
    record_type = clean_text(record.get("record_type"))
    if not text or not condition_label:
        return None

    species = normalize_species(text=text)
    display_condition = condition_label
    user_content = build_user_content(
        species=species,
        complaint=text,
        note_type=record_type or None,
    )
    assistant_content = build_structured_response(
        species=species,
        presenting_symptoms=text,
        signalment="Signalment not provided",
        assessment=(
            f"The source note maps to the '{display_condition}' problem category. "
            "Use the clinical history and physical examination to narrow the diagnosis."
        ),
        differential_lines=[
            f"1. {display_condition} — Source category label associated with this note."
        ],
        triage=infer_triage(text, display_condition),
        next_steps=(
            "Perform a veterinary examination and targeted diagnostics to refine this broad problem category."
        ),
    )
    triage = infer_triage(text, display_condition)
    source_labels = [label for label in [condition_label, record_type] if label]
    return make_chatml(
        user_content=user_content,
        assistant_content=assistant_content,
        source=record["_source"],
        species=species,
        condition=None,
        triage=triage,
        source_labels=source_labels,
    )


def is_symptom_assessment_row(question: str, labels: list[str]) -> bool:
    lowered_question = question.casefold()
    lowered_labels = [label.casefold() for label in labels]
    if any("symptomassessment" in label for label in lowered_labels):
        return True

    symptom_markers = [
        "symptoms of",
        "symptoms are characteristic of",
        "symptoms might indicate",
        "list the symptoms of",
        "typical symptoms of",
    ]
    return any(marker in lowered_question for marker in symptom_markers)


def extract_condition_from_question(question: str) -> str | None:
    patterns = [
        r"common symptoms of (?P<condition>.+?)\??$",
        r"typical symptoms of (?P<condition>.+?)\??$",
        r"list the symptoms of (?P<condition>.+?)\??$",
        r"symptoms are characteristic of (?P<condition>.+?)\??$",
        r"symptoms might indicate (?:a |an )?(?:dog|cat|horse|cow|pig|sheep|animal) has (?P<condition>.+?)\??$",
        r"symptoms are characteristic of a[n]? .+? suffering from (?P<condition>.+?)\??$",
    ]
    question_text = clean_text(question)
    for pattern in patterns:
        match = re.search(pattern, question_text, flags=re.IGNORECASE)
        if match:
            condition = clean_text(match.group("condition"))
            condition = re.sub(r"^(?:a|an|the)\s+", "", condition, flags=re.IGNORECASE)
            return cleanup_condition_text(condition)
    return None


def normalize_symptom_answer(answer: str) -> str:
    text = clean_text(answer)
    prefixes = [
        r"^symptoms include\s+",
        r"^symptoms can include\s+",
        r"^symptoms may include\s+",
        r"^symptoms may involve\s+",
        r"^common symptoms of .+? include\s+",
    ]
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)
    return text


def normalize_vet_health_assessment(record: dict) -> dict[str, Any] | None:
    """
    infinite-dataset-hub/VetHealthAssessment
    Real fields: Question, Answer, Labels
    """
    question = clean_text(record.get("Question") or record.get("question"))
    answer = clean_text(record.get("Answer") or record.get("answer"))
    labels = normalize_source_labels(record.get("Labels") or record.get("labels"))

    if not question or not answer or not is_symptom_assessment_row(question, labels):
        return None

    condition = extract_condition_from_question(question)
    if not condition:
        return None

    species = normalize_species(text=question, labels=labels)
    presenting_symptoms = normalize_symptom_answer(answer)
    triage = infer_triage(presenting_symptoms, condition)
    assessment = (
        f"The symptom pattern in the source answer is compatible with {condition}. "
        "Confirm the diagnosis with species-appropriate examination and testing."
    )

    user_content = build_user_content(species=species, complaint=presenting_symptoms)
    assistant_content = build_structured_response(
        species=species,
        presenting_symptoms=presenting_symptoms,
        signalment="Signalment not provided",
        assessment=assessment,
        differential_lines=[
            f"1. {condition} — Symptoms from the source QA pair align with this condition."
        ],
        triage=triage,
        next_steps=(
            "Correlate the symptom history with physical examination findings and confirmatory diagnostics."
        ),
    )

    return make_chatml(
        user_content=user_content,
        assistant_content=assistant_content,
        source=record["_source"],
        species=species,
        condition=cleanup_condition_text(condition),
        triage=triage,
        source_labels=labels,
        condition_source="question_extraction",
    )


def normalize_vet_pet_care(record: dict) -> dict[str, Any] | None:
    """
    infinite-dataset-hub/VetPetCare
    Real fields: species, breed, age, symptoms, diagnosis, treatment_plan, follow_up_result
    """
    raw_species = clean_text(record.get("species"))
    breed = clean_text(record.get("breed"))
    age = clean_text(record.get("age"))
    symptoms = clean_text(record.get("symptoms"))
    diagnosis = clean_text(record.get("diagnosis"))
    treatment = clean_text(record.get("treatment_plan"))
    follow_up = clean_text(record.get("follow_up_result"))

    if not symptoms or not diagnosis:
        return None

    species = normalize_species(raw_species=raw_species, text=symptoms)
    condition_label, condition_source = select_condition_label(diagnosis, symptoms)
    if not condition_label:
        return None
    display_condition = condition_label or diagnosis
    triage = infer_triage(symptoms, display_condition)
    user_content = build_user_content(
        species=species,
        complaint=symptoms,
        age=f"{age} years" if age else None,
        breed=breed or None,
    )

    next_steps = treatment or "Perform targeted diagnostics and start condition-appropriate therapy."
    if follow_up:
        next_steps = f"{next_steps} Monitor response: {follow_up}."

    assistant_content = build_structured_response(
        species=species,
        presenting_symptoms=symptoms,
        signalment=", ".join(part for part in [f"{age} years" if age else "", breed] if part) or "Signalment not provided",
        assessment=(
            f"The reported symptoms are compatible with {display_condition}. "
            "Use examination findings and diagnostics to confirm severity and rule out close differentials."
        ),
        differential_lines=[
            f"1. {display_condition} — Primary source diagnosis label after normalization."
        ],
        triage=triage,
        next_steps=next_steps,
    )

    source_labels = [label for label in [raw_species, diagnosis, treatment, follow_up] if label]
    return make_chatml(
        user_content=user_content,
        assistant_content=assistant_content,
        source=record["_source"],
        species=species,
        condition=condition_label,
        triage=triage,
        source_labels=source_labels,
        condition_source=condition_source,
    )


NORMALIZERS = {
    "karenwky/pet-health-symptoms-dataset": normalize_pet_health_symptoms,
    "infinite-dataset-hub/VetHealthAssessment": normalize_vet_health_assessment,
    "infinite-dataset-hub/VetPetCare": normalize_vet_pet_care,
}


def normalize_source_records(repo_id: str, raw_records: list[dict]) -> list[dict[str, Any]]:
    normalizer = NORMALIZERS[repo_id]
    normalized = [normalizer(record) for record in raw_records]
    return [record for record in normalized if record is not None]


def extract_first_differential(text: str) -> str | None:
    match = re.search(r"^\s*1\.\s+(.+?)(?:\s+—|\s+-|$)", text, flags=re.MULTILINE)
    if not match:
        return None
    return clean_text(match.group(1))


def normalize_synthetic_sample(record: dict) -> dict[str, Any]:
    messages = record.get("messages", [])
    user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
    assistant_content = next(
        (m["content"] for m in messages if m["role"] == "assistant"), ""
    )
    meta = dict(record.get("_meta") or {})
    source = clean_text(meta.get("source")) or "synthetic_ollama"
    source_labels = normalize_source_labels(meta.get("source_labels"))

    species = normalize_species(
        raw_species=meta.get("species"),
        text=user_content,
        labels=source_labels,
    )
    condition = cleanup_condition_text(meta.get("condition"))
    if condition is None:
        condition = cleanup_condition_text(extract_first_differential(assistant_content))
    triage = canonicalize_triage(meta.get("triage")) or extract_triage_from_response(
        assistant_content
    )

    if "_meta" not in record:
        record["_meta"] = {}
    record["_meta"].update(
        {
            "source": source,
            "species": species,
            "condition": condition,
            "triage": triage,
            "urgent": bool(triage == "Urgent"),
            "source_labels": source_labels,
        }
    )
    return record


# ---------------------------------------------------------------------------
# Filtering, deduplication, splitting
# ---------------------------------------------------------------------------


def filter_target_species(
    samples: list[dict[str, Any]],
    allowed_species: set[str] = TARGET_SPECIES,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    dropped = Counter()

    for sample in samples:
        species = sample.get("_meta", {}).get("species", "unknown")
        if species in allowed_species:
            kept.append(sample)
        else:
            dropped[species] += 1

    return kept, dropped


def log_condition_quality(samples: list[dict[str, Any]], label: str) -> None:
    sources = Counter()
    unlabeled = Counter()
    urgent = 0

    for sample in samples:
        meta = sample.get("_meta", {})
        source = clean_text(meta.get("source")) or "unknown"
        sources[source] += 1
        if not meta.get("condition"):
            unlabeled[source] += 1
        if meta.get("urgent"):
            urgent += 1

    summary = ", ".join(f"{source}={count}" for source, count in sorted(sources.items()))
    unlabeled_summary = ", ".join(
        f"{source}={count}" for source, count in sorted(unlabeled.items())
    )
    log.info("%s source counts: %s", label, summary or "none")
    log.info("%s unlabeled conditions: %s", label, unlabeled_summary or "none")
    log.info("%s urgent samples: %s/%s", label, urgent, len(samples))


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

    log.info("Deduplicating %s samples (threshold=%s) ...", len(samples), threshold)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [sample["messages"][1]["content"] for sample in samples]
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )

    keep = [True] * len(samples)
    for i in range(len(samples)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(samples)):
            if not keep[j]:
                continue
            similarity = float(
                np.dot(embeddings[i], embeddings[j])
                / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8)
            )
            if similarity > threshold:
                keep[j] = False

    deduped = [sample for sample, is_kept in zip(samples, keep) if is_kept]
    log.info(
        "After deduplication: %s samples (%s removed)",
        len(deduped),
        len(samples) - len(deduped),
    )
    return deduped


def split_counts_for_bucket(
    total: int,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 0, 1
    if total < 10:
        return total - 1, 0, 1

    n_train = max(1, int(total * ratios[0]))
    n_val = max(1, int(total * ratios[1]))
    n_test = total - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    while n_train + n_val + n_test > total:
        if n_train > max(1, n_val):
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    while n_train + n_val + n_test < total:
        n_train += 1

    return n_train, n_val, n_test


def oversample_urgent_train(
    train: list[dict[str, Any]],
    target_ratio: float = URGENT_TARGET_TRAIN_RATIO,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    if not train:
        return train

    urgent_indices = [
        index for index, sample in enumerate(train) if sample.get("_meta", {}).get("urgent")
    ]
    if not urgent_indices:
        log.warning("No urgent samples available for train-set rebalancing.")
        return train

    current_ratio = len(urgent_indices) / len(train)
    if current_ratio >= target_ratio:
        return train

    rng = random.Random(seed)
    augmented = list(train)
    while urgent_indices and (sum(sample.get("_meta", {}).get("urgent", False) for sample in augmented) / len(augmented)) < target_ratio:
        source_index = rng.choice(urgent_indices)
        cloned = copy.deepcopy(train[source_index])
        cloned.setdefault("_meta", {})
        cloned["_meta"]["oversampled"] = True
        cloned["_meta"]["oversample_source_index"] = source_index
        augmented.append(cloned)

    log.info(
        "Oversampled urgent train cases from %.2f%% to %.2f%% (%s -> %s records)",
        current_ratio * 100,
        (
            sum(sample.get("_meta", {}).get("urgent", False) for sample in augmented)
            / len(augmented)
        )
        * 100,
        len(train),
        len(augmented),
    )
    return augmented


def _count_matching(
    samples: list[dict[str, Any]],
    predicate,
) -> int:
    return sum(1 for sample in samples if predicate(sample))


def _move_matching_sample(
    source_split: list[dict[str, Any]],
    target_split: list[dict[str, Any]],
    predicate,
    seed: int,
) -> bool:
    candidate_indices = [
        index for index, sample in enumerate(source_split) if predicate(sample)
    ]
    if not candidate_indices:
        return False

    rng = random.Random(seed)
    chosen_index = rng.choice(candidate_indices)
    target_split.append(source_split.pop(chosen_index))
    return True


def ensure_eval_split_coverage(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    requirements = {
        "condition_labeled": lambda sample: bool(sample.get("_meta", {}).get("condition")),
        "urgent": lambda sample: bool(sample.get("_meta", {}).get("urgent")),
    }
    split_map = {"train": train, "val": val, "test": test}

    for split_name in ["val", "test"]:
        target_split = split_map[split_name]
        for requirement_name, predicate in requirements.items():
            if _count_matching(target_split, predicate) >= 1:
                continue

            moved = _move_matching_sample(train, target_split, predicate, seed)
            if not moved:
                sibling_name = "test" if split_name == "val" else "val"
                sibling_split = split_map[sibling_name]
                if _count_matching(sibling_split, predicate) > 1:
                    moved = _move_matching_sample(
                        sibling_split,
                        target_split,
                        predicate,
                        seed,
                    )

            if moved:
                log.info(
                    "Moved one %s sample into %s to guarantee evaluation coverage.",
                    requirement_name,
                    split_name,
                )
            else:
                log.warning(
                    "Unable to guarantee %s coverage for %s; no donor sample was available.",
                    requirement_name,
                    split_name,
                )

    return train, val, test


def _split_record_for_audit(sample: dict[str, Any], split_name: str, index: int) -> dict[str, Any]:
    meta = sample.get("_meta", {})
    return {
        "split": split_name,
        "index": index,
        "species": meta.get("species", "unknown"),
        "condition": meta.get("condition"),
        "source": meta.get("source"),
        "user": clean_text(get_message_content(sample.get("messages", []), "user")),
    }


def audit_cross_split_duplicates(
    split_records: dict[str, list[dict[str, Any]]],
    near_duplicate_ratio: float = NEAR_DUPLICATE_RATIO,
    max_examples: int = 25,
) -> dict[str, Any]:
    split_names = list(split_records)
    audit = {
        "exact_duplicate_count": 0,
        "near_duplicate_count": 0,
        "exact_duplicates": [],
        "near_duplicates": [],
    }

    flattened = {
        split_name: [
            _split_record_for_audit(sample, split_name, index)
            for index, sample in enumerate(records)
        ]
        for split_name, records in split_records.items()
    }

    text_index: dict[str, list[dict[str, Any]]] = {}
    for split_name in split_names:
        for record in flattened[split_name]:
            if record["user"]:
                text_index.setdefault(record["user"].casefold(), []).append(record)

    for user_text, matches in text_index.items():
        del user_text
        split_set = {match["split"] for match in matches}
        if len(split_set) < 2:
            continue
        audit["exact_duplicate_count"] += 1
        if len(audit["exact_duplicates"]) < max_examples:
            audit["exact_duplicates"].append(matches)

    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            for left_record in flattened[left_name]:
                if not left_record["user"]:
                    continue
                for right_record in flattened[right_name]:
                    if not right_record["user"]:
                        continue
                    if left_record["species"] != right_record["species"]:
                        continue
                    similarity = SequenceMatcher(
                        None, left_record["user"].casefold(), right_record["user"].casefold()
                    ).ratio()
                    if similarity < near_duplicate_ratio or similarity >= 1.0:
                        continue
                    audit["near_duplicate_count"] += 1
                    if len(audit["near_duplicates"]) < max_examples:
                        audit["near_duplicates"].append(
                            {
                                "left": left_record,
                                "right": right_record,
                                "similarity": round(similarity, 4),
                            }
                        )

    return audit


def stratified_split(
    samples: list[dict],
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified 80/10/10 split using stable metadata.
    Stratification key: normalized species + condition.
    """
    random.seed(seed)

    buckets: dict[tuple[str, str], list[dict]] = {}
    for sample in samples:
        meta = sample.get("_meta", {})
        species = meta.get("species", "unknown")
        condition = meta.get("condition") or "__unlabeled__"
        buckets.setdefault((species, condition), []).append(sample)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for (species, condition), group in buckets.items():
        random.shuffle(group)
        total = len(group)
        n_train, n_val, n_test = split_counts_for_bucket(total, ratios)

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val : n_train + n_val + n_test])

        log.info(
            "  %s / %s: %s train / %s val / %s test",
            species,
            condition,
            n_train,
            n_val,
            n_test,
        )

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def collect_samples(
    raw_source_records: dict[str, list[dict]],
    synthetic_records: list[dict] | None = None,
) -> list[dict[str, Any]]:
    all_samples: list[dict[str, Any]] = []

    for repo_id in HF_SOURCES:
        records = raw_source_records.get(repo_id, [])
        normalized = normalize_source_records(repo_id, records)
        log.info("  Normalized: %s usable samples from %s", len(normalized), repo_id)
        all_samples.extend(normalized)

    if synthetic_records:
        normalized_synth = [normalize_synthetic_sample(record) for record in synthetic_records]
        log.info("Loaded %s synthetic samples", len(normalized_synth))
        all_samples.extend(normalized_synth)

    return all_samples


def log_species_counts(samples: list[dict[str, Any]], label: str) -> None:
    species_counts = Counter(sample["_meta"]["species"] for sample in samples)
    counts_str = ", ".join(
        f"{species}={count}" for species, count in sorted(species_counts.items())
    )
    log.info("%s species counts: %s", label, counts_str or "none")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)

    raw_source_records = {repo_id: load_hf_dataset(repo_id) for repo_id in HF_SOURCES}

    synthetic_records: list[dict] | None = None
    if args.synthetic:
        synth_path = Path(args.synthetic)
        if synth_path.exists():
            synthetic_records = load_jsonl(synth_path)
            log.info("Loaded %s synthetic samples from %s", len(synthetic_records), synth_path)
        else:
            log.warning("Synthetic file not found: %s", synth_path)

    all_samples = collect_samples(raw_source_records, synthetic_records)
    log.info("Total samples before species filtering: %s", len(all_samples))
    log_species_counts(all_samples, "Pre-filter")

    all_samples, dropped = filter_target_species(all_samples, allowed_species=TARGET_SPECIES)
    if dropped:
        for species, count in sorted(dropped.items()):
            log.info("Dropped %s out-of-scope samples for species=%s", count, species)
    log.info("Total samples after species filtering: %s", len(all_samples))
    log_species_counts(all_samples, "Post-filter")
    log_condition_quality(all_samples, "Post-filter")

    all_samples = deduplicate(all_samples, threshold=DEDUP_THRESHOLD)

    log.info("Splitting dataset ...")
    train, val, test = stratified_split(all_samples)
    train, val, test = ensure_eval_split_coverage(train, val, test, seed=RANDOM_SEED)
    duplicate_audit = audit_cross_split_duplicates(
        {"train": train, "val": val, "test": test},
        near_duplicate_ratio=NEAR_DUPLICATE_RATIO,
    )
    train = oversample_urgent_train(train, target_ratio=URGENT_TARGET_TRAIN_RATIO)
    log.info("Final split — train: %s, val: %s, test: %s", len(train), len(val), len(test))
    log_condition_quality(train, "Train")
    log_condition_quality(val, "Val")
    log_condition_quality(test, "Test")

    for split_name, records in [("train", train), ("val", val), ("test", test)]:
        output_path = output_dir / f"{split_name}.jsonl"
        write_jsonl(records, output_path)
        log.info("Wrote %s records to %s", len(records), output_path)
    with (output_dir / "duplicate_audit.json").open("w") as handle:
        json.dump(duplicate_audit, handle, indent=2)
    log.info(
        "Duplicate audit saved to %s (exact=%s, near=%s)",
        output_dir / "duplicate_audit.json",
        duplicate_audit["exact_duplicate_count"],
        duplicate_audit["near_duplicate_count"],
    )

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
    parsed_args = parser.parse_args()
    main(parsed_args)
