# VetQwen Dataset Card

## Dataset Summary

A curated and augmented instruction-tuning dataset for veterinary differential diagnosis,
formatted in ChatML for fine-tuning Qwen2.5-3B-Instruct.

**Task:** Symptom → Structured Differential Diagnosis (open-ended reasoning)
**Format:** ChatML (system / user / assistant message triples)
**Species:** Dogs, Cats, Cattle, Pigs, Sheep

---

## Data Sources

### Tier 1 — Hugging Face Hub

| Dataset | Records (approx.) | Language | Species | Notes |
|---|---|---|---|---|
| `karenwky/pet-health-symptoms-dataset` | ~2,000 | EN | Dogs, Cats | Labeled symptoms → conditions; lay + clinical language |
| `infinite-dataset-hub/VetHealthAssessment` | TBD | EN | Mixed | Structured Q&A: symptoms, diagnostics, treatment |
| `infinite-dataset-hub/VetPetCare` | TBD | EN | Dogs, Cats | Clinic records with diagnosis + treatment plan |

### Tier 2 — Synthetic (Ollama / Qwen2.5:9b)

- Generated using `scripts/generate_synthetic.py`
- Target: 300–500 livestock cases (Cattle, Pig, Sheep)
- Generation prompt: structured case → signalment, complaint, signs, differentials, triage
- Validation: random sample of 50 cases reviewed manually for clinical plausibility

### Tier 3 — Augmentation (optional)

- Paraphrased entries (formal ↔ lay language)
- Species/breed variation added to companion animal cases

---

## Processing Pipeline

1. Download all Tier 1 datasets from HuggingFace Hub (`scripts/build_dataset.py`)
2. Normalize each source to ChatML instruction format
3. Combine with synthetic Tier 2 data
4. Deduplicate using sentence-transformer cosine similarity (threshold: 0.95)
5. Stratified 80/10/10 split by species and condition

---

## Splits

| Split | File | Records | Notes |
|---|---|---|---|
| Train | `data/processed/train.jsonl` | TBD | 80% stratified |
| Val | `data/processed/val.jsonl` | TBD | 10% stratified |
| Test | `data/processed/test.jsonl` | TBD | 10% stratified — held out for final evaluation |

---

## ChatML Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are VetQwen, an expert veterinary diagnostic assistant..."
    },
    {
      "role": "user",
      "content": "Species: Dog\nAge: 3 years | Sex: Male (neutered) | Breed: Golden Retriever\nPresenting complaint: ..."
    },
    {
      "role": "assistant",
      "content": "**Species & Signalment:** ...\n**Presenting Symptoms:** ...\n\n**Assessment:**\n...\n\n**Differential Diagnoses (ranked by likelihood):**\n1. ...\n2. ...\n\n**Triage Recommendation:** ...\n**Suggested Next Steps:** ..."
    }
  ],
  "_meta": {
    "source": "karenwky/pet-health-symptoms-dataset"
  }
}
```

---

## Known Limitations

- **Livestock underrepresentation:** Public veterinary datasets skew heavily toward companion animals.
  Synthetic generation (Tier 2) partially addresses this but synthetic cases may contain errors.
- **No radiograph/lab data:** Dataset is symptom-text-only; no imaging or bloodwork context.
- **English only:** All data is in English; no multilingual coverage.
- **Synthetic quality:** LLM-generated cases have not been validated by a licensed veterinarian.
  Do not use for clinical decision support without expert review.
- **Class imbalance:** Condition frequency mirrors real-world prevalence (common conditions over-represented).

---

## Intended Use

- Fine-tuning and evaluation of VetQwen for **research and educational purposes only**
- Benchmarking veterinary NLP models

## Out-of-Scope Use

- Clinical decision support without veterinary oversight
- Replacing professional veterinary examination or diagnosis

---

## Statistics

_Statistics are populated after running `python scripts/build_dataset.py`:_

- Total samples: Depends on synthetic data generation count
- Post-deduplication: Reduced by ~5-10% via cosine similarity (threshold: 0.95)
- Species distribution: Skews toward companion animals; synthetic data fills livestock gaps
- Median token length: Varies by source dataset
- P95 token length: See `configs/train_default.yaml` `max_seq_length` for training cutoff
