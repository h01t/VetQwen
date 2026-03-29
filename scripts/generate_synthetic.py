"""
generate_synthetic.py — Synthetic veterinary case generation via Ollama

Uses a locally running Ollama instance with Qwen2.5:7b to generate
structured veterinary cases for livestock species (cattle, pigs, sheep)
where public datasets are thin.

Prerequisites:
    - Ollama running locally: `ollama serve`
    - Model pulled: `ollama pull qwen2.5:7b`

Usage:
    python scripts/generate_synthetic.py --n 400 --output data/raw/synthetic.jsonl
    python scripts/generate_synthetic.py --n 100 --species cattle --output data/raw/synthetic_cattle.jsonl
"""

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

from vetqwen_core.records import build_structured_response, make_chatml_record
from vetqwen_core.text import canonicalize_triage, clean_text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

SPECIES_POOL = ["Cattle", "Pig", "Sheep"]

GENERATION_PROMPT_TEMPLATE = """Generate a realistic and clinically plausible veterinary case for {species}.

Include:
- Signalment (age, sex, breed)
- Presenting complaint (owner's description in lay language)
- Clinical signs observed
- A structured differential diagnosis with reasoning (2-3 differentials, ranked by likelihood)
- A triage recommendation (Urgent / Schedule within 48h / Monitor at home)

Output ONLY valid JSON with exactly these keys:
{{
  "signalment": "...",
  "complaint": "...",
  "signs": "...",
  "differentials": [
    {{"rank": 1, "diagnosis": "...", "rationale": "..."}},
    {{"rank": 2, "diagnosis": "...", "rationale": "..."}},
    {{"rank": 3, "diagnosis": "...", "rationale": "..."}}
  ],
  "triage": "..."
}}

Species: {species}
"""


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


def check_ollama(base_url: str, model: str) -> bool:
    """Verify Ollama is running and the target model is available."""
    import requests

    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(model in m for m in models):
            log.error(f"Model '{model}' not found in Ollama. Run: ollama pull {model}")
            return False
        return True
    except requests.ConnectionError:
        log.error(f"Cannot connect to Ollama at {base_url}. Is it running?")
        return False


def canonicalize_condition(value: object) -> str | None:
    text = clean_text(value).casefold()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9\s/&-]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text or None


def generate_via_ollama(
    prompt: str, base_url: str, model: str, temperature: float = 0.8
) -> str | None:
    """Send a generation request to Ollama and return the raw text response."""
    import requests

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": 600,
        },
    }
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        log.warning(f"Ollama request failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Case generation & parsing
# ---------------------------------------------------------------------------


def generate_case(species: str, base_url: str, model: str) -> dict | None:
    """Generate one synthetic case and return as a parsed dict, or None on failure."""
    prompt = GENERATION_PROMPT_TEMPLATE.format(species=species)
    raw = generate_via_ollama(prompt, base_url=base_url, model=model)
    if not raw:
        return None

    # Extract JSON from response (model may wrap it in markdown code fences)
    json_str = raw
    if "```" in raw:
        # Strip markdown code fences
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                json_str = part
                break

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        log.warning(f"Failed to parse JSON for {species}: {e}\nRaw: {raw[:200]}")
        return None

    normalized = normalize_generated_case(data)
    if normalized is None:
        log.warning("Discarding malformed generated case for %s", species)
        return None

    return normalized


def _coerce_rank(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return max(value, 1)
    if isinstance(value, float):
        return max(int(value), 1)
    text = clean_text(value)
    if not text:
        return fallback
    match = re.search(r"\d+", text)
    if not match:
        return fallback
    return max(int(match.group(0)), 1)


def _clean_case_field(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(clean_text(item) for item in value if clean_text(item))
    elif isinstance(value, dict):
        return ""
    return clean_text(value)


def normalize_generated_case(data: dict[str, Any]) -> dict[str, Any] | None:
    required = {"signalment", "complaint", "signs", "differentials", "triage"}
    if not required.issubset(data.keys()):
        log.warning("Missing keys in generated case: %s", required - set(data.keys()))
        return None

    signalment = _clean_case_field(data.get("signalment"))
    complaint = _clean_case_field(data.get("complaint"))
    signs = _clean_case_field(data.get("signs"))
    triage = canonicalize_triage(data.get("triage")) or "Schedule within 48h"
    if not signalment or not complaint or not signs:
        log.warning("Generated case is missing one or more core text fields.")
        return None

    raw_differentials = data.get("differentials")
    if not isinstance(raw_differentials, list):
        log.warning("Generated case differentials field is not a list.")
        return None

    normalized_differentials: list[dict[str, Any]] = []
    for index, differential in enumerate(raw_differentials, start=1):
        diagnosis = ""
        rationale = ""
        rank = index

        if isinstance(differential, dict):
            diagnosis = _clean_case_field(
                differential.get("diagnosis")
                or differential.get("condition")
                or differential.get("name")
            )
            rationale = _clean_case_field(
                differential.get("rationale")
                or differential.get("reasoning")
                or differential.get("reason")
                or differential.get("justification")
            )
            rank = _coerce_rank(differential.get("rank"), fallback=index)
        elif isinstance(differential, str):
            diagnosis = clean_text(differential)

        if not diagnosis:
            continue
        if not rationale:
            rationale = "Compatible with the reported signalment and clinical signs."

        normalized_differentials.append(
            {
                "rank": rank,
                "diagnosis": diagnosis,
                "rationale": rationale,
            }
        )

    normalized_differentials.sort(key=lambda item: (item["rank"], item["diagnosis"]))
    normalized_differentials = normalized_differentials[:3]
    for index, differential in enumerate(normalized_differentials, start=1):
        differential["rank"] = index

    if not normalized_differentials:
        normalized_differentials.append(
            {
                "rank": 1,
                "diagnosis": "Requires further evaluation",
                "rationale": "The generated case did not include a usable ranked differential.",
            }
        )

    return {
        "signalment": signalment,
        "complaint": complaint,
        "signs": signs,
        "differentials": normalized_differentials,
        "triage": triage,
    }


def case_to_chatml(species: str, case: dict) -> dict:
    """Convert a parsed synthetic case dict into ChatML format."""
    user_content = (
        f"Species: {species}\n"
        f"Signalment: {case['signalment']}\n"
        f"Presenting complaint: {case['complaint']}\n"
        f"Clinical signs: {case['signs']}"
    )

    # Build ranked differential list
    diffs = case.get("differentials", [])
    diff_lines = []
    for d in sorted(
        [d for d in diffs if isinstance(d, dict) and d.get("diagnosis")],
        key=lambda x: x.get("rank", 99),
    ):
        rank = _coerce_rank(d.get("rank"), fallback=len(diff_lines) + 1)
        diagnosis = _clean_case_field(d.get("diagnosis"))
        rationale = _clean_case_field(d.get("rationale"))
        if not diagnosis:
            continue
        if not rationale:
            rationale = "Compatible with the reported signalment and clinical signs."
        diff_lines.append(f"{rank}. {diagnosis} — {rationale}")
    triage = canonicalize_triage(case.get("triage")) or "Schedule within 48h"
    assistant_content = build_structured_response(
        species=species.strip().lower(),
        presenting_symptoms=case["complaint"],
        signalment=case["signalment"],
        assessment=(
            f"Based on the reported clinical signs ({case['signs']}), "
            "the following differentials are considered."
        ),
        differential_lines=diff_lines or ["1. Requires further evaluation — More diagnostics are needed."],
        triage=triage,
        next_steps="Consult a large animal veterinarian for physical examination and targeted diagnostics.",
    )

    primary_condition = None
    if diffs:
        primary_condition = canonicalize_condition(
            sorted(diffs, key=lambda x: x.get("rank", 99))[0].get("diagnosis", "")
        )
    return make_chatml_record(
        user_content=user_content,
        assistant_content=assistant_content,
        source="synthetic_ollama",
        species=species.strip().lower(),
        condition=primary_condition,
        triage=triage,
        source_labels=[],
        condition_source="synthetic_primary_differential",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    base_url = args.ollama_url
    model = args.model

    if not check_ollama(base_url, model):
        raise SystemExit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    species_pool = [args.species] if args.species else SPECIES_POOL
    target_n = args.n

    log.info(f"Generating {target_n} synthetic cases for species: {species_pool}")
    log.info(f"Using Ollama at {base_url} with model {model}")
    log.info(f"Output: {output_path}")

    generated = []
    failed = 0
    attempt = 0

    with output_path.open("w") as f_out:
        while len(generated) < target_n:
            attempt += 1
            species = random.choice(species_pool)

            case = generate_case(species, base_url=base_url, model=model)
            if case is None:
                failed += 1
                if failed > target_n * 0.3:
                    log.error(
                        "Too many failures — check Ollama and model availability."
                    )
                    break
                time.sleep(1)
                continue

            try:
                chatml = case_to_chatml(species, case)
            except Exception as exc:
                failed += 1
                log.warning(
                    "Failed to convert generated case for %s into ChatML: %s",
                    species,
                    exc,
                )
                if failed > target_n * 0.3:
                    log.error(
                        "Too many failures — check Ollama output quality and model stability."
                    )
                    break
                time.sleep(1)
                continue
            f_out.write(json.dumps(chatml, ensure_ascii=False) + "\n")
            f_out.flush()
            generated.append(chatml)

            if len(generated) % 50 == 0:
                log.info(
                    f"  Progress: {len(generated)}/{target_n} (attempts: {attempt}, failed: {failed})"
                )

            # Small pause to avoid hammering Ollama
            time.sleep(0.2)

    log.info(
        f"Done. Generated {len(generated)} cases in {attempt} attempts "
        f"({failed} failures). Written to {output_path}"
    )

    # Remind user to validate a sample
    if len(generated) >= 50:
        log.info(
            "IMPORTANT: Manually review a random sample of 50 cases for clinical plausibility "
            "before including in training data."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic livestock veterinary cases via Ollama"
    )
    parser.add_argument(
        "--n", type=int, default=400, help="Number of cases to generate"
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        choices=["Cattle", "Pig", "Sheep"],
        help="Restrict to a single species (default: all three)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/synthetic.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_OLLAMA_MODEL})",
    )
    args = parser.parse_args()

    main(args)
