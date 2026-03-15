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
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You are VetQwen, an expert veterinary diagnostic assistant. "
    "Given a patient signalment and clinical symptoms, provide a structured "
    "differential diagnosis with clinical reasoning, ranked differentials, and "
    "a triage recommendation. Always remind the user that your output is not a "
    "substitute for professional veterinary examination."
)

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


def generate_via_ollama(
    prompt: str, base_url: str, model: str, temperature: float = 0.8
) -> str | None:
    """Send a generation request to Ollama and return the raw text response."""
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

    # Validate required keys
    required = {"signalment", "complaint", "signs", "differentials", "triage"}
    if not required.issubset(data.keys()):
        log.warning(f"Missing keys in generated case: {required - set(data.keys())}")
        return None

    return data


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
    for d in sorted(diffs, key=lambda x: x.get("rank", 99)):
        diff_lines.append(f"{d['rank']}. {d['diagnosis']} — {d['rationale']}")
    diff_str = "\n".join(diff_lines) if diff_lines else "1. Requires further evaluation"

    assistant_content = (
        f"**Species & Signalment:** {species}, {case['signalment']}\n"
        f"**Presenting Symptoms:** {case['complaint']}\n\n"
        f"**Assessment:**\n"
        f"Based on the reported clinical signs ({case['signs']}), the following differentials are considered.\n\n"
        f"**Differential Diagnoses (ranked by likelihood):**\n{diff_str}\n\n"
        f"**Triage Recommendation:** {case['triage']}\n"
        f"**Suggested Next Steps:** Consult a large animal veterinarian for physical examination and targeted diagnostics.\n\n"
        f"*Note: This output is not a substitute for professional veterinary examination.*"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "_meta": {"source": "synthetic_ollama", "species": species},
    }


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

            chatml = case_to_chatml(species, case)
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
